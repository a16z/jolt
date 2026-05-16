#![allow(dead_code)]

use bolt_verifier_runtime::{batch_claims, find_batch, find_plan};
use jolt_field::{Field, Fr};
use jolt_sumcheck::SumcheckError;
use jolt_transcript::{Blake2bTranscript, LabelWithCount, Transcript};

pub type Stage4NamedEval<F> = bolt_verifier_runtime::StageNamedEval<F>;
pub type Stage4SumcheckOutput<F> = bolt_verifier_runtime::StageSumcheckOutput<F>;
pub type Stage4ChallengeVector<F> = bolt_verifier_runtime::StageChallengeVector<F>;
pub type Stage4ExecutionArtifacts<F> = bolt_verifier_runtime::StageExecutionArtifacts<F>;
pub type Stage4Proof<F> = bolt_verifier_runtime::StageProof<F>;
pub type Stage4OpeningInputValue<F> = bolt_verifier_runtime::StageOpeningInputValue<F>;
pub type Stage4CpuProgramPlan = bolt_verifier_runtime::StageProgramPlanNoPointZeros<Stage4RelationKind>;
pub type Stage4SumcheckClaimPlan = bolt_verifier_runtime::SumcheckClaimPlan<Stage4RelationKind>;
pub type Stage4SumcheckDriverPlan = bolt_verifier_runtime::SumcheckDriverPlan<Stage4RelationKind>;
pub type Stage4SumcheckInstanceResultPlan = bolt_verifier_runtime::SumcheckInstanceResultPlan<Stage4RelationKind>;
pub type Stage4RelationOutputPlan = bolt_verifier_runtime::RelationOutputPlan<Stage4RelationKind>;
pub type Stage4StructuredPolynomialEvalPlan = bolt_verifier_runtime::StructuredPolynomialEvalPlan;

pub use super::jolt_relations::JoltRelationKind as Stage4RelationKind;
pub use bolt_verifier_runtime::{
    ClaimKind as Stage4ClaimKind, FieldConstantPlan as Stage4FieldConstantPlan,
    FieldExprKind as Stage4FieldExprKind,
    FieldExprPlan as Stage4FieldExprPlan,
    KernelPlan as Stage4KernelPlan, OpeningBatchPlan as Stage4OpeningBatchPlan,
    OpeningClaimEqualityPlan as Stage4OpeningClaimEqualityPlan,
    OpeningClaimPlan as Stage4OpeningClaimPlan, OpeningInputPlan as Stage4OpeningInputPlan,
    OpeningEqualityMode as Stage4OpeningEqualityMode,
    PointConcatPlan as Stage4PointConcatPlan, PointSlicePlan as Stage4PointSlicePlan,
    ProgramStepKind as Stage4ProgramStepKind,
    ProgramStepPlan as Stage4ProgramStepPlan, StageParams as Stage4Params,
    SumcheckBatchPlan as Stage4SumcheckBatchPlan,
    SumcheckEvalPlan as Stage4SumcheckEvalPlan,
    StructuredPolynomialPointLength as Stage4StructuredPolynomialPointLength,
    StructuredPolynomialPointOrder as Stage4StructuredPolynomialPointOrder,
    StructuredPolynomialPointPlan as Stage4StructuredPolynomialPointPlan,
    StructuredPolynomialPointSegment as Stage4StructuredPolynomialPointSegment,
    StructuredPolynomialKind as Stage4StructuredPolynomialKind,
    TranscriptAbsorbBytesPlan as Stage4TranscriptAbsorbBytesPlan,
    TranscriptSqueezeKind as Stage4TranscriptSqueezeKind,
    TranscriptSqueezePlan as Stage4TranscriptSqueezePlan,
};

pub type DefaultStage4Transcript = Blake2bTranscript<Fr>;
pub type Stage4VerifierProgramPlan = Stage4CpuProgramPlan;

#[derive(Debug)]
pub enum VerifyStage4Error {
    UnexpectedProofCount { expected: usize, got: usize },
    MissingProof { driver: &'static str },
    MissingBatch { driver: &'static str, batch: &'static str },
    MissingClaim { batch: &'static str, claim: &'static str },
    MissingValue { symbol: &'static str },
    InvalidInputLength { input: &'static str, expected: usize, actual: usize },
    InvalidProof { driver: &'static str, reason: &'static str },
    UnsupportedRelation { relation: Stage4RelationKind },
    Sumcheck { driver: &'static str, error: SumcheckError<Fr> },
}

bolt_verifier_runtime::impl_runtime_plan_error_conversion!(VerifyStage4Error);

pub const STAGE4_PARAMS: Stage4Params = Stage4Params { field: "bn254_fr", pcs: "dory", transcript: "blake2b_transcript" };
pub const STAGE4_PROGRAM_STEPS: &[Stage4ProgramStepPlan] = &[
    Stage4ProgramStepPlan { kind: Stage4ProgramStepKind::TranscriptSqueeze, symbol: "stage4.registers_read_write.gamma" },
    Stage4ProgramStepPlan { kind: Stage4ProgramStepKind::TranscriptAbsorbBytes, symbol: "stage4.ram_val_check.domain_separator" },
    Stage4ProgramStepPlan { kind: Stage4ProgramStepKind::TranscriptSqueeze, symbol: "stage4.ram_val_check.gamma" },
    Stage4ProgramStepPlan { kind: Stage4ProgramStepKind::SumcheckDriver, symbol: "stage4.sumcheck" },
];

pub const STAGE4_TRANSCRIPT_SQUEEZES: &[Stage4TranscriptSqueezePlan] = &[
    Stage4TranscriptSqueezePlan { symbol: "stage4.registers_read_write.gamma", label: "registers_read_write_gamma", kind: Stage4TranscriptSqueezeKind::ChallengeScalar, count: 1 },
    Stage4TranscriptSqueezePlan { symbol: "stage4.ram_val_check.gamma", label: "ram_val_check_gamma", kind: Stage4TranscriptSqueezeKind::ChallengeScalar, count: 1 },
];

pub const STAGE4_TRANSCRIPT_ABSORB_BYTES: &[Stage4TranscriptAbsorbBytesPlan] = &[
    Stage4TranscriptAbsorbBytesPlan { symbol: "stage4.ram_val_check.domain_separator", label: "ram_val_check_gamma", payload: "" },
];

pub const STAGE4_OPENING_INPUTS: &[Stage4OpeningInputPlan] = &[
    Stage4OpeningInputPlan { symbol: "stage4.input.stage3.registers.RdWriteValue", source_stage: "stage3", source_claim: "stage3.registers_claim_reduction.opening.RdWriteValue", oracle: "RdWriteValue", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage4ClaimKind::Virtual },
    Stage4OpeningInputPlan { symbol: "stage4.input.stage3.registers.Rs1Value", source_stage: "stage3", source_claim: "stage3.registers_claim_reduction.opening.Rs1Value", oracle: "Rs1Value", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage4ClaimKind::Virtual },
    Stage4OpeningInputPlan { symbol: "stage4.input.stage3.registers.Rs2Value", source_stage: "stage3", source_claim: "stage3.registers_claim_reduction.opening.Rs2Value", oracle: "Rs2Value", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage4ClaimKind::Virtual },
    Stage4OpeningInputPlan { symbol: "stage4.input.stage3.instruction.Rs1Value", source_stage: "stage3", source_claim: "stage3.instruction_input.opening.Rs1Value", oracle: "Rs1Value", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage4ClaimKind::Virtual },
    Stage4OpeningInputPlan { symbol: "stage4.input.stage3.instruction.Rs2Value", source_stage: "stage3", source_claim: "stage3.instruction_input.opening.Rs2Value", oracle: "Rs2Value", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage4ClaimKind::Virtual },
    Stage4OpeningInputPlan { symbol: "stage4.input.stage2.RamVal", source_stage: "stage2", source_claim: "stage2.ram_read_write.opening.RamVal", oracle: "RamVal", domain: "jolt.stage2_ram_rw_domain", point_arity: 32, claim_kind: Stage4ClaimKind::Virtual },
    Stage4OpeningInputPlan { symbol: "stage4.input.stage2.RamValFinal", source_stage: "stage2", source_claim: "stage2.ram_output.opening.RamValFinal", oracle: "RamValFinal", domain: "jolt.ram_address_domain", point_arity: 16, claim_kind: Stage4ClaimKind::Virtual },
    Stage4OpeningInputPlan { symbol: "stage4.input.initial_ram.RamValInit", source_stage: "stage4_precomputed", source_claim: "stage4.ram_val_check.initial_ram_eval", oracle: "RamValInit", domain: "jolt.ram_address_domain", point_arity: 16, claim_kind: Stage4ClaimKind::Virtual },
];

pub const STAGE4_FIELD_CONSTANTS: &[Stage4FieldConstantPlan] = &[

];

pub const STAGE4_FIELD_EXPRS: &[Stage4FieldExprPlan] = &[
    Stage4FieldExprPlan { symbol: "stage4.registers_read_write.gamma2", kind: Stage4FieldExprKind::Pow(2), operands: &["stage4.registers_read_write.gamma"] },
    Stage4FieldExprPlan { symbol: "stage4.registers_read_write.term.Rs1Value", kind: Stage4FieldExprKind::Mul, operands: &["stage4.registers_read_write.gamma", "stage4.input.stage3.registers.Rs1Value"] },
    Stage4FieldExprPlan { symbol: "stage4.registers_read_write.term.Rs2Value", kind: Stage4FieldExprKind::Mul, operands: &["stage4.registers_read_write.gamma2", "stage4.input.stage3.registers.Rs2Value"] },
    Stage4FieldExprPlan { symbol: "stage4.registers_read_write.partial.RdWriteValueRs1Value", kind: Stage4FieldExprKind::Add, operands: &["stage4.input.stage3.registers.RdWriteValue", "stage4.registers_read_write.term.Rs1Value"] },
    Stage4FieldExprPlan { symbol: "stage4.registers_read_write.claim_expr", kind: Stage4FieldExprKind::Add, operands: &["stage4.registers_read_write.partial.RdWriteValueRs1Value", "stage4.registers_read_write.term.Rs2Value"] },
    Stage4FieldExprPlan { symbol: "stage4.ram_val_check.delta.RamVal", kind: Stage4FieldExprKind::Sub, operands: &["stage4.input.stage2.RamVal", "stage4.input.initial_ram.RamValInit"] },
    Stage4FieldExprPlan { symbol: "stage4.ram_val_check.delta.RamValFinal", kind: Stage4FieldExprKind::Sub, operands: &["stage4.input.stage2.RamValFinal", "stage4.input.initial_ram.RamValInit"] },
    Stage4FieldExprPlan { symbol: "stage4.ram_val_check.term.RamValFinal", kind: Stage4FieldExprKind::Mul, operands: &["stage4.ram_val_check.gamma", "stage4.ram_val_check.delta.RamValFinal"] },
    Stage4FieldExprPlan { symbol: "stage4.ram_val_check.claim_expr", kind: Stage4FieldExprKind::Add, operands: &["stage4.ram_val_check.delta.RamVal", "stage4.ram_val_check.term.RamValFinal"] },
    Stage4FieldExprPlan { symbol: "stage4.registers_read_write.output.sum.RegistersValRdInc", kind: Stage4FieldExprKind::Add, operands: &["stage4.registers_read_write.eval.RegistersVal", "stage4.registers_read_write.eval.RdInc"] },
    Stage4FieldExprPlan { symbol: "stage4.registers_read_write.output.term.RdWa", kind: Stage4FieldExprKind::Mul, operands: &["stage4.registers_read_write.eval.RdWa", "stage4.registers_read_write.output.sum.RegistersValRdInc"] },
    Stage4FieldExprPlan { symbol: "stage4.registers_read_write.output.term.Rs1Ra", kind: Stage4FieldExprKind::Mul, operands: &["stage4.registers_read_write.eval.Rs1Ra", "stage4.registers_read_write.eval.RegistersVal"] },
    Stage4FieldExprPlan { symbol: "stage4.registers_read_write.output.term.Rs2Ra", kind: Stage4FieldExprKind::Mul, operands: &["stage4.registers_read_write.eval.Rs2Ra", "stage4.registers_read_write.eval.RegistersVal"] },
    Stage4FieldExprPlan { symbol: "stage4.registers_read_write.output.weighted.Rs2Ra", kind: Stage4FieldExprKind::Mul, operands: &["stage4.registers_read_write.gamma", "stage4.registers_read_write.output.term.Rs2Ra"] },
    Stage4FieldExprPlan { symbol: "stage4.registers_read_write.output.read_terms", kind: Stage4FieldExprKind::Add, operands: &["stage4.registers_read_write.output.term.Rs1Ra", "stage4.registers_read_write.output.weighted.Rs2Ra"] },
    Stage4FieldExprPlan { symbol: "stage4.registers_read_write.output.weighted.read_terms", kind: Stage4FieldExprKind::Mul, operands: &["stage4.registers_read_write.gamma", "stage4.registers_read_write.output.read_terms"] },
    Stage4FieldExprPlan { symbol: "stage4.registers_read_write.output.weighted_values", kind: Stage4FieldExprKind::Add, operands: &["stage4.registers_read_write.output.term.RdWa", "stage4.registers_read_write.output.weighted.read_terms"] },
    Stage4FieldExprPlan { symbol: "stage4.registers_read_write.output.claim_expr", kind: Stage4FieldExprKind::Mul, operands: &["stage4.registers_read_write.output.eq.RdWriteValue", "stage4.registers_read_write.output.weighted_values"] },
    Stage4FieldExprPlan { symbol: "stage4.ram_val_check.output.lt_plus_gamma", kind: Stage4FieldExprKind::Add, operands: &["stage4.ram_val_check.output.lt.RamValCycle", "stage4.ram_val_check.gamma"] },
    Stage4FieldExprPlan { symbol: "stage4.ram_val_check.output.term.RamIncRamRa", kind: Stage4FieldExprKind::Mul, operands: &["stage4.ram_val_check.eval.RamInc", "stage4.ram_val_check.eval.RamRa"] },
    Stage4FieldExprPlan { symbol: "stage4.ram_val_check.output.claim_expr", kind: Stage4FieldExprKind::Mul, operands: &["stage4.ram_val_check.output.term.RamIncRamRa", "stage4.ram_val_check.output.lt_plus_gamma"] },
];
pub const STAGE4_KERNELS: &[Stage4KernelPlan] = &[

];

pub const STAGE4_SUMCHECK_CLAIMS: &[Stage4SumcheckClaimPlan] = &[
    Stage4SumcheckClaimPlan { symbol: "stage4.registers_read_write.input", stage: "stage4", domain: "jolt.stage4_registers_rw_domain", num_rounds: 23, degree: 3, claim: "stage4.registers_read_write.weighted_values", kernel: None, relation: Some(Stage4RelationKind::Stage4RegistersReadWrite), claim_value: "stage4.registers_read_write.claim_expr" },
    Stage4SumcheckClaimPlan { symbol: "stage4.ram_val_check.input", stage: "stage4", domain: "jolt.trace_domain", num_rounds: 16, degree: 3, claim: "stage4.ram_val_check.weighted_values", kernel: None, relation: Some(Stage4RelationKind::Stage4RamValCheck), claim_value: "stage4.ram_val_check.claim_expr" },
];
pub const STAGE4_SUMCHECK_BATCH_0_ROUND_SCHEDULE: &[usize] = &[16, 7];

pub const STAGE4_SUMCHECK_BATCHES: &[Stage4SumcheckBatchPlan] = &[
    Stage4SumcheckBatchPlan { symbol: "stage4.batch", stage: "stage4", proof_slot: "stage4.sumcheck", policy: "jolt_core_stage4_aligned", count: 2, claim_operands: &["stage4.registers_read_write.input", "stage4.ram_val_check.input"], claim_label: "sumcheck_claim", round_label: "sumcheck_poly", round_schedule: STAGE4_SUMCHECK_BATCH_0_ROUND_SCHEDULE },
];
pub const STAGE4_SUMCHECK_DRIVER_0_ROUND_SCHEDULE: &[usize] = &[16, 7];

pub const STAGE4_SUMCHECK_DRIVERS: &[Stage4SumcheckDriverPlan] = &[
    Stage4SumcheckDriverPlan { symbol: "stage4.sumcheck", stage: "stage4", proof_slot: "stage4.sumcheck", kernel: None, relation: Some(Stage4RelationKind::Stage4Batched), batch: "stage4.batch", policy: "jolt_core_stage4_aligned", round_schedule: STAGE4_SUMCHECK_DRIVER_0_ROUND_SCHEDULE, claim_label: "sumcheck_claim", round_label: "sumcheck_poly", num_rounds: 23, degree: 3 },
];
pub const STAGE4_SUMCHECK_INSTANCE_RESULTS: &[Stage4SumcheckInstanceResultPlan] = &[
    Stage4SumcheckInstanceResultPlan { symbol: "stage4.registers_read_write.instance", source: "stage4.sumcheck", claim: "stage4.registers_read_write.input", relation: Stage4RelationKind::Stage4RegistersReadWrite, index: 0, point_arity: 23, num_rounds: 23, round_offset: 0, point_order: bolt_verifier_runtime::SumcheckPointOrder::Stage4RegistersReadWrite, degree: 3 },
    Stage4SumcheckInstanceResultPlan { symbol: "stage4.ram_val_check.instance", source: "stage4.sumcheck", claim: "stage4.ram_val_check.input", relation: Stage4RelationKind::Stage4RamValCheck, index: 1, point_arity: 16, num_rounds: 16, round_offset: 7, point_order: bolt_verifier_runtime::SumcheckPointOrder::Reverse, degree: 3 },
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

pub const STAGE4_POINT_CONCATS: &[Stage4PointConcatPlan] = &[
    Stage4PointConcatPlan { symbol: "stage4.ram_val_check.point.RamRa", layout: "address_then_cycle", arity: 32, inputs: &["stage4.ram_val_check.point.RamAddress", "stage4.ram_val_check.instance"] },
];
pub const STAGE4_OPENING_CLAIMS: &[Stage4OpeningClaimPlan] = &[
    Stage4OpeningClaimPlan { symbol: "stage4.registers_read_write.opening.RegistersVal", oracle: "RegistersVal", domain: "jolt.stage4_registers_rw_domain", point_arity: 23, claim_kind: Stage4ClaimKind::Virtual, point_source: "stage4.registers_read_write.instance", eval_source: "stage4.registers_read_write.eval.RegistersVal" },
    Stage4OpeningClaimPlan { symbol: "stage4.registers_read_write.opening.Rs1Ra", oracle: "Rs1Ra", domain: "jolt.stage4_registers_rw_domain", point_arity: 23, claim_kind: Stage4ClaimKind::Virtual, point_source: "stage4.registers_read_write.instance", eval_source: "stage4.registers_read_write.eval.Rs1Ra" },
    Stage4OpeningClaimPlan { symbol: "stage4.registers_read_write.opening.Rs2Ra", oracle: "Rs2Ra", domain: "jolt.stage4_registers_rw_domain", point_arity: 23, claim_kind: Stage4ClaimKind::Virtual, point_source: "stage4.registers_read_write.instance", eval_source: "stage4.registers_read_write.eval.Rs2Ra" },
    Stage4OpeningClaimPlan { symbol: "stage4.registers_read_write.opening.RdWa", oracle: "RdWa", domain: "jolt.stage4_registers_rw_domain", point_arity: 23, claim_kind: Stage4ClaimKind::Virtual, point_source: "stage4.registers_read_write.instance", eval_source: "stage4.registers_read_write.eval.RdWa" },
    Stage4OpeningClaimPlan { symbol: "stage4.registers_read_write.opening.RdInc", oracle: "RdInc", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage4ClaimKind::Committed, point_source: "stage4.registers_read_write.point.RdInc", eval_source: "stage4.registers_read_write.eval.RdInc" },
    Stage4OpeningClaimPlan { symbol: "stage4.ram_val_check.opening.RamRa", oracle: "RamRa", domain: "jolt.stage2_ram_rw_domain", point_arity: 32, claim_kind: Stage4ClaimKind::Virtual, point_source: "stage4.ram_val_check.point.RamRa", eval_source: "stage4.ram_val_check.eval.RamRa" },
    Stage4OpeningClaimPlan { symbol: "stage4.ram_val_check.opening.RamInc", oracle: "RamInc", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage4ClaimKind::Committed, point_source: "stage4.ram_val_check.instance", eval_source: "stage4.ram_val_check.eval.RamInc" },
];

pub const STAGE4_OPENING_EQUALITIES: &[Stage4OpeningClaimEqualityPlan] = &[
    Stage4OpeningClaimEqualityPlan { symbol: "stage4.registers.rs1_claim_consistency", mode: Stage4OpeningEqualityMode::PointAndEval, lhs: "stage4.input.stage3.registers.Rs1Value", rhs: "stage4.input.stage3.instruction.Rs1Value" },
    Stage4OpeningClaimEqualityPlan { symbol: "stage4.registers.rs2_claim_consistency", mode: Stage4OpeningEqualityMode::PointAndEval, lhs: "stage4.input.stage3.registers.Rs2Value", rhs: "stage4.input.stage3.instruction.Rs2Value" },
];

pub const STAGE4_OPENING_BATCHES: &[Stage4OpeningBatchPlan] = &[
    Stage4OpeningBatchPlan { symbol: "stage4.openings", stage: "stage4", proof_slot: "stage4.openings", policy: "jolt_stage4_output_order", count: 7, ordered_claims: &["stage4.registers_read_write.opening.RegistersVal", "stage4.registers_read_write.opening.Rs1Ra", "stage4.registers_read_write.opening.Rs2Ra", "stage4.registers_read_write.opening.RdWa", "stage4.registers_read_write.opening.RdInc", "stage4.ram_val_check.opening.RamRa", "stage4.ram_val_check.opening.RamInc"], claim_operands: &["stage4.registers_read_write.opening.RegistersVal", "stage4.registers_read_write.opening.Rs1Ra", "stage4.registers_read_write.opening.Rs2Ra", "stage4.registers_read_write.opening.RdWa", "stage4.registers_read_write.opening.RdInc", "stage4.ram_val_check.opening.RamRa", "stage4.ram_val_check.opening.RamInc"] },
];
pub const STAGE4_RELATION_OUTPUT_VALUES: &[Stage4StructuredPolynomialEvalPlan] = &[
    Stage4StructuredPolynomialEvalPlan { symbol: "stage4.registers_read_write.output.eq.RdWriteValue", polynomial: Stage4StructuredPolynomialKind::Eq, x_point: Stage4StructuredPolynomialPointPlan { source: "stage4.registers_read_write.instance", segment: Stage4StructuredPolynomialPointSegment::Prefix, length: Stage4StructuredPolynomialPointLength::YPoint, order: Stage4StructuredPolynomialPointOrder::Reverse }, y_point: Stage4StructuredPolynomialPointPlan { source: "stage4.input.stage3.registers.RdWriteValue", segment: Stage4StructuredPolynomialPointSegment::Full, length: Stage4StructuredPolynomialPointLength::Full, order: Stage4StructuredPolynomialPointOrder::AsIs } },
    Stage4StructuredPolynomialEvalPlan { symbol: "stage4.ram_val_check.output.lt.RamValCycle", polynomial: Stage4StructuredPolynomialKind::Lt, x_point: Stage4StructuredPolynomialPointPlan { source: "stage4.ram_val_check.instance", segment: Stage4StructuredPolynomialPointSegment::Full, length: Stage4StructuredPolynomialPointLength::Full, order: Stage4StructuredPolynomialPointOrder::Reverse }, y_point: Stage4StructuredPolynomialPointPlan { source: "stage4.input.stage2.RamVal", segment: Stage4StructuredPolynomialPointSegment::Suffix, length: Stage4StructuredPolynomialPointLength::XPoint, order: Stage4StructuredPolynomialPointOrder::AsIs } },
];

pub const STAGE4_RELATION_OUTPUTS: &[Stage4RelationOutputPlan] = &[
    Stage4RelationOutputPlan { relation: Stage4RelationKind::Stage4RegistersReadWrite, structured_polynomial_evals: &[bolt_verifier_runtime::StructuredPolynomialEvalRef { symbol: "stage4.registers_read_write.output.eq.RdWriteValue", index: 0 }], eval_families: &[], local_scalars: &[], expected_output: "stage4.registers_read_write.output.claim_expr" },
    Stage4RelationOutputPlan { relation: Stage4RelationKind::Stage4RamValCheck, structured_polynomial_evals: &[bolt_verifier_runtime::StructuredPolynomialEvalRef { symbol: "stage4.ram_val_check.output.lt.RamValCycle", index: 1 }], eval_families: &[], local_scalars: &[], expected_output: "stage4.ram_val_check.output.claim_expr" },
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
    relation_output_values: STAGE4_RELATION_OUTPUT_VALUES,
    relation_outputs: STAGE4_RELATION_OUTPUTS,
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
    let mut store =
        bolt_verifier_runtime::ValueStore::with_opening_inputs(opening_inputs, program.opening_inputs)?;
    store.seed_constants(program.field_constants);
    let mut artifacts = Stage4ExecutionArtifacts::default();
    for step in program.steps {
        match step.kind {
            Stage4ProgramStepKind::TranscriptSqueeze => {
                let squeeze =
                    find_plan(program.transcript_squeezes, step.symbol).ok_or(VerifyStage4Error::MissingValue {
                        symbol: step.symbol,
                    })?;
                verify_stage4_squeeze(program, squeeze, &mut store, transcript, &mut artifacts)?;
            }
            Stage4ProgramStepKind::TranscriptAbsorbBytes => {
                let absorb = find_plan(program.transcript_absorb_bytes, step.symbol).ok_or(
                    VerifyStage4Error::MissingValue {
                        symbol: step.symbol,
                    },
                )?;
                absorb_stage4_bytes(absorb, transcript);
            }
            Stage4ProgramStepKind::SumcheckDriver => {
                let driver =
                    find_plan(program.drivers, step.symbol).ok_or(VerifyStage4Error::MissingProof {
                        driver: step.symbol,
                    })?;
                verify_stage4_driver(program, driver, proof, &mut store, transcript, &mut artifacts)?;
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
    store: &mut bolt_verifier_runtime::ValueStore<Fr>,
    transcript: &mut T,
    artifacts: &mut Stage4ExecutionArtifacts<Fr>,
) -> Result<(), VerifyStage4Error>
where
    T: Transcript<Challenge = Fr>,
{
    let values = transcript.challenge_vector(squeeze.count);
    store.observe_challenge_vector(squeeze, &values, |input, expected, actual| {
        VerifyStage4Error::InvalidInputLength {
            input,
            expected,
            actual,
        }
    })?;
    store
        .evaluate_available_field_exprs(program.field_exprs, bolt_verifier_runtime::evaluate_field_expr)
        .map_err(VerifyStage4Error::from)?;
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
    store: &mut bolt_verifier_runtime::ValueStore<Fr>,
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
    let Some(relation) = driver.relation else {
        return Err(VerifyStage4Error::InvalidProof {
            driver: driver.symbol,
            reason: "missing driver relation",
        });
    };
    let output = match relation {
        Stage4RelationKind::Stage4Batched => {
            verify_batched_stage4(program, driver, proof, store, transcript)?
        }
        relation => return Err(VerifyStage4Error::UnsupportedRelation { relation }),
    };
    artifacts.sumchecks.push(output);
    Ok(())
}

fn verify_batched_stage4<T>(
    program: &'static Stage4VerifierProgramPlan,
    driver: &'static Stage4SumcheckDriverPlan,
    proof: &Stage4SumcheckOutput<Fr>,
    store: &mut bolt_verifier_runtime::ValueStore<Fr>,
    transcript: &mut T,
) -> Result<Stage4SumcheckOutput<Fr>, VerifyStage4Error>
where
    T: Transcript<Challenge = Fr>,
{
    store.evaluate_available_points(
        program.point_slices,
        program.point_concats,
        |input, expected, actual| VerifyStage4Error::InvalidInputLength {
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
        |store, verified| observe_stage4_sumcheck_output(program, store, verified),
        |driver, error| VerifyStage4Error::Sumcheck { driver, error },
    )
}

fn observe_stage4_sumcheck_output<F: Field>(
    program: &'static Stage4VerifierProgramPlan,
    store: &mut bolt_verifier_runtime::ValueStore<F>,
    output: &Stage4SumcheckOutput<F>,
) -> Result<(), VerifyStage4Error> {
    store.observe_sumcheck_output(
        program.instance_results,
        program.evals,
        output,
        |instance, mut point| {
            match instance.point_order {
                bolt_verifier_runtime::SumcheckPointOrder::AsIs => {}
                bolt_verifier_runtime::SumcheckPointOrder::Reverse => point.reverse(),
                bolt_verifier_runtime::SumcheckPointOrder::Stage4RegistersReadWrite => {
                    point = normalize_stage4_registers_rw_point(program, output.driver, &point)?;
                }
                _ => {
                    return Err(VerifyStage4Error::InvalidProof {
                        driver: output.driver,
                        reason: "unsupported point order",
                    });
                }
            }
            Ok(point)
        },
        |input, expected, actual| VerifyStage4Error::InvalidInputLength {
            input,
            expected,
            actual,
        },
        |symbol| VerifyStage4Error::MissingValue { symbol },
    )?;
    store.evaluate_available_points(
        program.point_slices,
        program.point_concats,
        |input, expected, actual| VerifyStage4Error::InvalidInputLength {
            input,
            expected,
            actual,
        },
    )?;
    store
        .evaluate_available_field_exprs(program.field_exprs, bolt_verifier_runtime::evaluate_field_expr)
        .map_err(VerifyStage4Error::from)?;
    store.verify_opening_equalities(
        program.opening_equalities,
        |driver, reason| VerifyStage4Error::InvalidProof { driver, reason },
        |symbol| VerifyStage4Error::MissingValue { symbol },
    )
}

fn expected_batched_output_claim(
    program: &'static Stage4VerifierProgramPlan,
    driver: &'static Stage4SumcheckDriverPlan,
    store: &bolt_verifier_runtime::ValueStore<Fr>,
    evals: &[Stage4NamedEval<Fr>],
    point: &[Fr],
    batching_coeffs: &[Fr],
) -> Result<Fr, VerifyStage4Error> {
    let batch = find_batch(program.batches, driver.symbol, driver.batch)?;
    let claims = batch_claims(program.claims, batch)?;
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
        let value = bolt_verifier_runtime::evaluate_relation_output_for_instance(
            program.relation_outputs,
        program.relation_output_values,
            program.field_exprs,
            store,
            instance,
            evals, &[], &[], local_point,
        )?;
        expected += *coefficient * value;
    }
    Ok(expected)
}

fn normalize_stage4_registers_rw_point<F: Field>(
    program: &'static Stage4VerifierProgramPlan,
    driver: &'static str,
    point: &[F],
) -> Result<Vec<F>, VerifyStage4Error> {
    let driver_plan = find_plan(program.drivers, driver).ok_or(VerifyStage4Error::MissingProof {
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
