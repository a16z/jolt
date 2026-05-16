#![allow(dead_code)]

use bolt_verifier_runtime::find_plan;
use super::jolt_relations::{evaluate_stage5_instruction_read_raf_local_scalars, normalize_instruction_read_raf_point, JoltLocalScalarMleKind, JoltLocalScalarPlan, Stage5InstructionReadRafPlan};
use jolt_field::{Field, Fr};
use jolt_sumcheck::SumcheckError;
use jolt_transcript::{Blake2bTranscript, LabelWithCount, Transcript};

pub type Stage5NamedEval<F> = bolt_verifier_runtime::StageNamedEval<F>;
pub type Stage5SumcheckOutput<F> = bolt_verifier_runtime::StageSumcheckOutput<F>;
pub type Stage5ChallengeVector<F> = bolt_verifier_runtime::StageChallengeVector<F>;
pub type Stage5ExecutionArtifacts<F> = bolt_verifier_runtime::StageExecutionArtifacts<F>;
pub type Stage5Proof<F> = bolt_verifier_runtime::StageProof<F>;
pub type Stage5OpeningInputValue<F> = bolt_verifier_runtime::StageOpeningInputValue<F>;
pub type Stage5CpuProgramPlan = bolt_verifier_runtime::StageProgramPlan<Stage5RelationKind>;
pub type Stage5SumcheckClaimPlan = bolt_verifier_runtime::SumcheckClaimPlan<Stage5RelationKind>;
pub type Stage5SumcheckDriverPlan = bolt_verifier_runtime::SumcheckDriverPlan<Stage5RelationKind>;
pub type Stage5SumcheckInstanceResultPlan = bolt_verifier_runtime::SumcheckInstanceResultPlan<Stage5RelationKind>;
pub type Stage5RelationOutputPlan = bolt_verifier_runtime::RelationOutputPlan<Stage5RelationKind>;

pub use super::jolt_relations::JoltRelationKind as Stage5RelationKind;
pub use bolt_verifier_runtime::{
    ClaimKind as Stage5ClaimKind, FieldConstantPlan as Stage5FieldConstantPlan,
    FieldExprKind as Stage5FieldExprKind,
    FieldExprPlan as Stage5FieldExprPlan,
    ScalarExprKind as Stage5ScalarExprKind,
    ScalarExprPlan as Stage5ScalarExprPlan,
    KernelPlan as Stage5KernelPlan, OpeningBatchPlan as Stage5OpeningBatchPlan,
    OpeningClaimEqualityPlan as Stage5OpeningClaimEqualityPlan,
    OpeningClaimPlan as Stage5OpeningClaimPlan, OpeningInputPlan as Stage5OpeningInputPlan,
    OpeningEqualityMode as Stage5OpeningEqualityMode, PointExprKind as Stage5PointExprKind,
    PointExprPlan as Stage5PointExprPlan,
    ProgramStepKind as Stage5ProgramStepKind,
    ProgramStepPlan as Stage5ProgramStepPlan, StageParams as Stage5Params,
    SumcheckBatchPlan as Stage5SumcheckBatchPlan,
    SumcheckEvalPlan as Stage5SumcheckEvalPlan,
    TranscriptAbsorbBytesPlan as Stage5TranscriptAbsorbBytesPlan,
    TranscriptSqueezeKind as Stage5TranscriptSqueezeKind,
    TranscriptSqueezePlan as Stage5TranscriptSqueezePlan,
};

pub type DefaultStage5Transcript = Blake2bTranscript<Fr>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage5VerifierProgramPlan {
    pub base: Stage5CpuProgramPlan,
    pub instruction_read_raf_plan: Stage5InstructionReadRafPlan,
}

impl core::ops::Deref for Stage5VerifierProgramPlan {
    type Target = Stage5CpuProgramPlan;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
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
    UnsupportedRelation { relation: Stage5RelationKind },
    Sumcheck { driver: &'static str, error: SumcheckError<Fr> },
}

bolt_verifier_runtime::impl_runtime_plan_error_conversion!(VerifyStage5Error);

pub const STAGE5_PARAMS: Stage5Params = Stage5Params { field: "bn254_fr", pcs: "dory", transcript: "blake2b_transcript" };
pub const STAGE5_PROGRAM_STEPS: &[Stage5ProgramStepPlan] = &[
    Stage5ProgramStepPlan { kind: Stage5ProgramStepKind::TranscriptSqueeze, symbol: "stage5.instruction_read_raf.gamma" },
    Stage5ProgramStepPlan { kind: Stage5ProgramStepKind::TranscriptSqueeze, symbol: "stage5.ram_ra_claim_reduction.gamma" },
    Stage5ProgramStepPlan { kind: Stage5ProgramStepKind::SumcheckDriver, symbol: "stage5.sumcheck" },
];

pub const STAGE5_TRANSCRIPT_SQUEEZES: &[Stage5TranscriptSqueezePlan] = &[
    Stage5TranscriptSqueezePlan { symbol: "stage5.instruction_read_raf.gamma", label: "instruction_read_raf_gamma", kind: Stage5TranscriptSqueezeKind::ChallengeScalar, count: 1 },
    Stage5TranscriptSqueezePlan { symbol: "stage5.ram_ra_claim_reduction.gamma", label: "ram_ra_claim_reduction_gamma", kind: Stage5TranscriptSqueezeKind::ChallengeScalar, count: 1 },
];

pub const STAGE5_TRANSCRIPT_ABSORB_BYTES: &[Stage5TranscriptAbsorbBytesPlan] = &[

];

pub const STAGE5_OPENING_INPUTS: &[Stage5OpeningInputPlan] = &[
    Stage5OpeningInputPlan { symbol: "stage5.input.stage2.instruction.LookupOutput", source_stage: "stage2", source_claim: "stage2.instruction_lookup.claim_reduction.opening.LookupOutput", oracle: "LookupOutput", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage5ClaimKind::Virtual },
    Stage5OpeningInputPlan { symbol: "stage5.input.stage2.product_virtual.LookupOutput", source_stage: "stage2", source_claim: "stage2.product_virtual.remainder.opening.LookupOutput", oracle: "LookupOutput", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage5ClaimKind::Virtual },
    Stage5OpeningInputPlan { symbol: "stage5.input.stage2.instruction.LeftLookupOperand", source_stage: "stage2", source_claim: "stage2.instruction_lookup.claim_reduction.opening.LeftLookupOperand", oracle: "LeftLookupOperand", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage5ClaimKind::Virtual },
    Stage5OpeningInputPlan { symbol: "stage5.input.stage2.instruction.RightLookupOperand", source_stage: "stage2", source_claim: "stage2.instruction_lookup.claim_reduction.opening.RightLookupOperand", oracle: "RightLookupOperand", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage5ClaimKind::Virtual },
    Stage5OpeningInputPlan { symbol: "stage5.input.stage2.ram_raf.RamRa", source_stage: "stage2", source_claim: "stage2.ram_raf.opening.RamRa", oracle: "RamRa", domain: "jolt.stage2_ram_rw_domain", point_arity: 32, claim_kind: Stage5ClaimKind::Virtual },
    Stage5OpeningInputPlan { symbol: "stage5.input.stage2.ram_read_write.RamRa", source_stage: "stage2", source_claim: "stage2.ram_read_write.opening.RamRa", oracle: "RamRa", domain: "jolt.stage2_ram_rw_domain", point_arity: 32, claim_kind: Stage5ClaimKind::Virtual },
    Stage5OpeningInputPlan { symbol: "stage5.input.stage4.ram_val_check.RamRa", source_stage: "stage4", source_claim: "stage4.ram_val_check.opening.RamRa", oracle: "RamRa", domain: "jolt.stage2_ram_rw_domain", point_arity: 32, claim_kind: Stage5ClaimKind::Virtual },
    Stage5OpeningInputPlan { symbol: "stage5.input.stage4.registers.RegistersVal", source_stage: "stage4", source_claim: "stage4.registers_read_write.opening.RegistersVal", oracle: "RegistersVal", domain: "jolt.stage4_registers_rw_domain", point_arity: 23, claim_kind: Stage5ClaimKind::Virtual },
];

pub const STAGE5_FIELD_CONSTANTS: &[Stage5FieldConstantPlan] = &[

];

pub const STAGE5_FIELD_EXPRS: &[Stage5FieldExprPlan] = &[
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.gamma2", kind: Stage5FieldExprKind::Pow(2), operands: &["stage5.instruction_read_raf.gamma"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.term.LeftLookupOperand", kind: Stage5FieldExprKind::Mul, operands: &["stage5.instruction_read_raf.gamma", "stage5.input.stage2.instruction.LeftLookupOperand"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.term.RightLookupOperand", kind: Stage5FieldExprKind::Mul, operands: &["stage5.instruction_read_raf.gamma2", "stage5.input.stage2.instruction.RightLookupOperand"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.partial.LookupOutputLeftOperand", kind: Stage5FieldExprKind::Add, operands: &["stage5.input.stage2.instruction.LookupOutput", "stage5.instruction_read_raf.term.LeftLookupOperand"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.claim_expr", kind: Stage5FieldExprKind::Add, operands: &["stage5.instruction_read_raf.partial.LookupOutputLeftOperand", "stage5.instruction_read_raf.term.RightLookupOperand"] },
    Stage5FieldExprPlan { symbol: "stage5.ram_ra_claim_reduction.gamma2", kind: Stage5FieldExprKind::Pow(2), operands: &["stage5.ram_ra_claim_reduction.gamma"] },
    Stage5FieldExprPlan { symbol: "stage5.ram_ra_claim_reduction.term.RamRaReadWrite", kind: Stage5FieldExprKind::Mul, operands: &["stage5.ram_ra_claim_reduction.gamma", "stage5.input.stage2.ram_read_write.RamRa"] },
    Stage5FieldExprPlan { symbol: "stage5.ram_ra_claim_reduction.term.RamRaValCheck", kind: Stage5FieldExprKind::Mul, operands: &["stage5.ram_ra_claim_reduction.gamma2", "stage5.input.stage4.ram_val_check.RamRa"] },
    Stage5FieldExprPlan { symbol: "stage5.ram_ra_claim_reduction.partial.RafReadWrite", kind: Stage5FieldExprKind::Add, operands: &["stage5.input.stage2.ram_raf.RamRa", "stage5.ram_ra_claim_reduction.term.RamRaReadWrite"] },
    Stage5FieldExprPlan { symbol: "stage5.ram_ra_claim_reduction.claim_expr", kind: Stage5FieldExprKind::Add, operands: &["stage5.ram_ra_claim_reduction.partial.RafReadWrite", "stage5.ram_ra_claim_reduction.term.RamRaValCheck"] },
    Stage5FieldExprPlan { symbol: "stage5.ram_ra_claim_reduction.output.term.ReadWrite", kind: Stage5FieldExprKind::Mul, operands: &["stage5.ram_ra_claim_reduction.gamma", "stage5.ram_ra_claim_reduction.output.eq.ReadWrite"] },
    Stage5FieldExprPlan { symbol: "stage5.ram_ra_claim_reduction.output.term.ValCheck", kind: Stage5FieldExprKind::Mul, operands: &["stage5.ram_ra_claim_reduction.gamma2", "stage5.ram_ra_claim_reduction.output.eq.ValCheck"] },
    Stage5FieldExprPlan { symbol: "stage5.ram_ra_claim_reduction.output.partial.RafReadWrite", kind: Stage5FieldExprKind::Add, operands: &["stage5.ram_ra_claim_reduction.output.eq.Raf", "stage5.ram_ra_claim_reduction.output.term.ReadWrite"] },
    Stage5FieldExprPlan { symbol: "stage5.ram_ra_claim_reduction.output.eq_combined", kind: Stage5FieldExprKind::Add, operands: &["stage5.ram_ra_claim_reduction.output.partial.RafReadWrite", "stage5.ram_ra_claim_reduction.output.term.ValCheck"] },
    Stage5FieldExprPlan { symbol: "stage5.ram_ra_claim_reduction.output.claim_expr", kind: Stage5FieldExprKind::Mul, operands: &["stage5.ram_ra_claim_reduction.output.eq_combined", "stage5.ram_ra_claim_reduction.eval.RamRa"] },
    Stage5FieldExprPlan { symbol: "stage5.registers_val_evaluation.output.product.RdIncRdWa", kind: Stage5FieldExprKind::Mul, operands: &["stage5.registers_val_evaluation.eval.RdInc", "stage5.registers_val_evaluation.eval.RdWa"] },
    Stage5FieldExprPlan { symbol: "stage5.registers_val_evaluation.output.claim_expr", kind: Stage5FieldExprKind::Mul, operands: &["stage5.registers_val_evaluation.output.product.RdIncRdWa", "stage5.registers_val_evaluation.output.lt.RegistersValCycle"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.term.LookupTableValue_0", kind: Stage5FieldExprKind::Mul, operands: &["stage5.instruction_read_raf.local_scalar.LookupTable_0", "stage5.instruction_read_raf.eval.LookupTableFlag_0"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.term.LookupTableValue_1", kind: Stage5FieldExprKind::Mul, operands: &["stage5.instruction_read_raf.local_scalar.LookupTable_1", "stage5.instruction_read_raf.eval.LookupTableFlag_1"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.term.LookupTableValue_2", kind: Stage5FieldExprKind::Mul, operands: &["stage5.instruction_read_raf.local_scalar.LookupTable_2", "stage5.instruction_read_raf.eval.LookupTableFlag_2"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.term.LookupTableValue_3", kind: Stage5FieldExprKind::Mul, operands: &["stage5.instruction_read_raf.local_scalar.LookupTable_3", "stage5.instruction_read_raf.eval.LookupTableFlag_3"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.term.LookupTableValue_4", kind: Stage5FieldExprKind::Mul, operands: &["stage5.instruction_read_raf.local_scalar.LookupTable_4", "stage5.instruction_read_raf.eval.LookupTableFlag_4"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.term.LookupTableValue_5", kind: Stage5FieldExprKind::Mul, operands: &["stage5.instruction_read_raf.local_scalar.LookupTable_5", "stage5.instruction_read_raf.eval.LookupTableFlag_5"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.term.LookupTableValue_6", kind: Stage5FieldExprKind::Mul, operands: &["stage5.instruction_read_raf.local_scalar.LookupTable_6", "stage5.instruction_read_raf.eval.LookupTableFlag_6"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.term.LookupTableValue_7", kind: Stage5FieldExprKind::Mul, operands: &["stage5.instruction_read_raf.local_scalar.LookupTable_7", "stage5.instruction_read_raf.eval.LookupTableFlag_7"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.term.LookupTableValue_8", kind: Stage5FieldExprKind::Mul, operands: &["stage5.instruction_read_raf.local_scalar.LookupTable_8", "stage5.instruction_read_raf.eval.LookupTableFlag_8"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.term.LookupTableValue_9", kind: Stage5FieldExprKind::Mul, operands: &["stage5.instruction_read_raf.local_scalar.LookupTable_9", "stage5.instruction_read_raf.eval.LookupTableFlag_9"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.term.LookupTableValue_10", kind: Stage5FieldExprKind::Mul, operands: &["stage5.instruction_read_raf.local_scalar.LookupTable_10", "stage5.instruction_read_raf.eval.LookupTableFlag_10"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.term.LookupTableValue_11", kind: Stage5FieldExprKind::Mul, operands: &["stage5.instruction_read_raf.local_scalar.LookupTable_11", "stage5.instruction_read_raf.eval.LookupTableFlag_11"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.term.LookupTableValue_12", kind: Stage5FieldExprKind::Mul, operands: &["stage5.instruction_read_raf.local_scalar.LookupTable_12", "stage5.instruction_read_raf.eval.LookupTableFlag_12"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.term.LookupTableValue_13", kind: Stage5FieldExprKind::Mul, operands: &["stage5.instruction_read_raf.local_scalar.LookupTable_13", "stage5.instruction_read_raf.eval.LookupTableFlag_13"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.term.LookupTableValue_14", kind: Stage5FieldExprKind::Mul, operands: &["stage5.instruction_read_raf.local_scalar.LookupTable_14", "stage5.instruction_read_raf.eval.LookupTableFlag_14"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.term.LookupTableValue_15", kind: Stage5FieldExprKind::Mul, operands: &["stage5.instruction_read_raf.local_scalar.LookupTable_15", "stage5.instruction_read_raf.eval.LookupTableFlag_15"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.term.LookupTableValue_16", kind: Stage5FieldExprKind::Mul, operands: &["stage5.instruction_read_raf.local_scalar.LookupTable_16", "stage5.instruction_read_raf.eval.LookupTableFlag_16"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.term.LookupTableValue_17", kind: Stage5FieldExprKind::Mul, operands: &["stage5.instruction_read_raf.local_scalar.LookupTable_17", "stage5.instruction_read_raf.eval.LookupTableFlag_17"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.term.LookupTableValue_18", kind: Stage5FieldExprKind::Mul, operands: &["stage5.instruction_read_raf.local_scalar.LookupTable_18", "stage5.instruction_read_raf.eval.LookupTableFlag_18"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.term.LookupTableValue_19", kind: Stage5FieldExprKind::Mul, operands: &["stage5.instruction_read_raf.local_scalar.LookupTable_19", "stage5.instruction_read_raf.eval.LookupTableFlag_19"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.term.LookupTableValue_20", kind: Stage5FieldExprKind::Mul, operands: &["stage5.instruction_read_raf.local_scalar.LookupTable_20", "stage5.instruction_read_raf.eval.LookupTableFlag_20"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.term.LookupTableValue_21", kind: Stage5FieldExprKind::Mul, operands: &["stage5.instruction_read_raf.local_scalar.LookupTable_21", "stage5.instruction_read_raf.eval.LookupTableFlag_21"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.term.LookupTableValue_22", kind: Stage5FieldExprKind::Mul, operands: &["stage5.instruction_read_raf.local_scalar.LookupTable_22", "stage5.instruction_read_raf.eval.LookupTableFlag_22"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.term.LookupTableValue_23", kind: Stage5FieldExprKind::Mul, operands: &["stage5.instruction_read_raf.local_scalar.LookupTable_23", "stage5.instruction_read_raf.eval.LookupTableFlag_23"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.term.LookupTableValue_24", kind: Stage5FieldExprKind::Mul, operands: &["stage5.instruction_read_raf.local_scalar.LookupTable_24", "stage5.instruction_read_raf.eval.LookupTableFlag_24"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.term.LookupTableValue_25", kind: Stage5FieldExprKind::Mul, operands: &["stage5.instruction_read_raf.local_scalar.LookupTable_25", "stage5.instruction_read_raf.eval.LookupTableFlag_25"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.term.LookupTableValue_26", kind: Stage5FieldExprKind::Mul, operands: &["stage5.instruction_read_raf.local_scalar.LookupTable_26", "stage5.instruction_read_raf.eval.LookupTableFlag_26"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.term.LookupTableValue_27", kind: Stage5FieldExprKind::Mul, operands: &["stage5.instruction_read_raf.local_scalar.LookupTable_27", "stage5.instruction_read_raf.eval.LookupTableFlag_27"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.term.LookupTableValue_28", kind: Stage5FieldExprKind::Mul, operands: &["stage5.instruction_read_raf.local_scalar.LookupTable_28", "stage5.instruction_read_raf.eval.LookupTableFlag_28"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.term.LookupTableValue_29", kind: Stage5FieldExprKind::Mul, operands: &["stage5.instruction_read_raf.local_scalar.LookupTable_29", "stage5.instruction_read_raf.eval.LookupTableFlag_29"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.term.LookupTableValue_30", kind: Stage5FieldExprKind::Mul, operands: &["stage5.instruction_read_raf.local_scalar.LookupTable_30", "stage5.instruction_read_raf.eval.LookupTableFlag_30"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.term.LookupTableValue_31", kind: Stage5FieldExprKind::Mul, operands: &["stage5.instruction_read_raf.local_scalar.LookupTable_31", "stage5.instruction_read_raf.eval.LookupTableFlag_31"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.term.LookupTableValue_32", kind: Stage5FieldExprKind::Mul, operands: &["stage5.instruction_read_raf.local_scalar.LookupTable_32", "stage5.instruction_read_raf.eval.LookupTableFlag_32"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.term.LookupTableValue_33", kind: Stage5FieldExprKind::Mul, operands: &["stage5.instruction_read_raf.local_scalar.LookupTable_33", "stage5.instruction_read_raf.eval.LookupTableFlag_33"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.term.LookupTableValue_34", kind: Stage5FieldExprKind::Mul, operands: &["stage5.instruction_read_raf.local_scalar.LookupTable_34", "stage5.instruction_read_raf.eval.LookupTableFlag_34"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.term.LookupTableValue_35", kind: Stage5FieldExprKind::Mul, operands: &["stage5.instruction_read_raf.local_scalar.LookupTable_35", "stage5.instruction_read_raf.eval.LookupTableFlag_35"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.term.LookupTableValue_36", kind: Stage5FieldExprKind::Mul, operands: &["stage5.instruction_read_raf.local_scalar.LookupTable_36", "stage5.instruction_read_raf.eval.LookupTableFlag_36"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.term.LookupTableValue_37", kind: Stage5FieldExprKind::Mul, operands: &["stage5.instruction_read_raf.local_scalar.LookupTable_37", "stage5.instruction_read_raf.eval.LookupTableFlag_37"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.term.LookupTableValue_38", kind: Stage5FieldExprKind::Mul, operands: &["stage5.instruction_read_raf.local_scalar.LookupTable_38", "stage5.instruction_read_raf.eval.LookupTableFlag_38"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.term.LookupTableValue_39", kind: Stage5FieldExprKind::Mul, operands: &["stage5.instruction_read_raf.local_scalar.LookupTable_39", "stage5.instruction_read_raf.eval.LookupTableFlag_39"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.term.LookupTableValue_40", kind: Stage5FieldExprKind::Mul, operands: &["stage5.instruction_read_raf.local_scalar.LookupTable_40", "stage5.instruction_read_raf.eval.LookupTableFlag_40"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.product.LookupTableValues", kind: Stage5FieldExprKind::Sum, operands: &["stage5.instruction_read_raf.output.term.LookupTableValue_0", "stage5.instruction_read_raf.output.term.LookupTableValue_1", "stage5.instruction_read_raf.output.term.LookupTableValue_2", "stage5.instruction_read_raf.output.term.LookupTableValue_3", "stage5.instruction_read_raf.output.term.LookupTableValue_4", "stage5.instruction_read_raf.output.term.LookupTableValue_5", "stage5.instruction_read_raf.output.term.LookupTableValue_6", "stage5.instruction_read_raf.output.term.LookupTableValue_7", "stage5.instruction_read_raf.output.term.LookupTableValue_8", "stage5.instruction_read_raf.output.term.LookupTableValue_9", "stage5.instruction_read_raf.output.term.LookupTableValue_10", "stage5.instruction_read_raf.output.term.LookupTableValue_11", "stage5.instruction_read_raf.output.term.LookupTableValue_12", "stage5.instruction_read_raf.output.term.LookupTableValue_13", "stage5.instruction_read_raf.output.term.LookupTableValue_14", "stage5.instruction_read_raf.output.term.LookupTableValue_15", "stage5.instruction_read_raf.output.term.LookupTableValue_16", "stage5.instruction_read_raf.output.term.LookupTableValue_17", "stage5.instruction_read_raf.output.term.LookupTableValue_18", "stage5.instruction_read_raf.output.term.LookupTableValue_19", "stage5.instruction_read_raf.output.term.LookupTableValue_20", "stage5.instruction_read_raf.output.term.LookupTableValue_21", "stage5.instruction_read_raf.output.term.LookupTableValue_22", "stage5.instruction_read_raf.output.term.LookupTableValue_23", "stage5.instruction_read_raf.output.term.LookupTableValue_24", "stage5.instruction_read_raf.output.term.LookupTableValue_25", "stage5.instruction_read_raf.output.term.LookupTableValue_26", "stage5.instruction_read_raf.output.term.LookupTableValue_27", "stage5.instruction_read_raf.output.term.LookupTableValue_28", "stage5.instruction_read_raf.output.term.LookupTableValue_29", "stage5.instruction_read_raf.output.term.LookupTableValue_30", "stage5.instruction_read_raf.output.term.LookupTableValue_31", "stage5.instruction_read_raf.output.term.LookupTableValue_32", "stage5.instruction_read_raf.output.term.LookupTableValue_33", "stage5.instruction_read_raf.output.term.LookupTableValue_34", "stage5.instruction_read_raf.output.term.LookupTableValue_35", "stage5.instruction_read_raf.output.term.LookupTableValue_36", "stage5.instruction_read_raf.output.term.LookupTableValue_37", "stage5.instruction_read_raf.output.term.LookupTableValue_38", "stage5.instruction_read_raf.output.term.LookupTableValue_39", "stage5.instruction_read_raf.output.term.LookupTableValue_40"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.term.GammaRightLookupOperand", kind: Stage5FieldExprKind::Mul, operands: &["stage5.instruction_read_raf.gamma", "stage5.instruction_read_raf.local_scalar.RightLookupOperand"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.partial.LeftPlusGammaRight", kind: Stage5FieldExprKind::Add, operands: &["stage5.instruction_read_raf.local_scalar.LeftLookupOperand", "stage5.instruction_read_raf.output.term.GammaRightLookupOperand"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.term.RafFlagLeftPlusGammaRight", kind: Stage5FieldExprKind::Mul, operands: &["stage5.instruction_read_raf.eval.InstructionRafFlag", "stage5.instruction_read_raf.output.partial.LeftPlusGammaRight"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.partial.NonRafLookupOperands", kind: Stage5FieldExprKind::Sub, operands: &["stage5.instruction_read_raf.output.partial.LeftPlusGammaRight", "stage5.instruction_read_raf.output.term.RafFlagLeftPlusGammaRight"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.term.GammaIdentity", kind: Stage5FieldExprKind::Mul, operands: &["stage5.instruction_read_raf.gamma", "stage5.instruction_read_raf.local_scalar.Identity"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.term.RafFlagGammaIdentity", kind: Stage5FieldExprKind::Mul, operands: &["stage5.instruction_read_raf.eval.InstructionRafFlag", "stage5.instruction_read_raf.output.term.GammaIdentity"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.partial.RafClaim", kind: Stage5FieldExprKind::Add, operands: &["stage5.instruction_read_raf.output.partial.NonRafLookupOperands", "stage5.instruction_read_raf.output.term.RafFlagGammaIdentity"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.term.GammaRafClaim", kind: Stage5FieldExprKind::Mul, operands: &["stage5.instruction_read_raf.gamma", "stage5.instruction_read_raf.output.partial.RafClaim"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.partial.LookupOrRaf", kind: Stage5FieldExprKind::Add, operands: &["stage5.instruction_read_raf.output.product.LookupTableValues", "stage5.instruction_read_raf.output.term.GammaRafClaim"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.partial.EqRa", kind: Stage5FieldExprKind::Mul, operands: &["stage5.instruction_read_raf.output.eq.LookupOutputCycle", "stage5.instruction_read_raf.output.product.InstructionRa"] },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.output.claim_expr", kind: Stage5FieldExprKind::Mul, operands: &["stage5.instruction_read_raf.output.partial.EqRa", "stage5.instruction_read_raf.output.partial.LookupOrRaf"] },
];
pub const STAGE5_SCALAR_EXPRS: &[Stage5ScalarExprPlan] = &[
    Stage5ScalarExprPlan { symbol: "stage5.instruction_read_raf.output.product.InstructionRa", kind: Stage5ScalarExprKind::FieldVectorProduct, operands: &["stage5.instruction_read_raf.eval.InstructionRa"] },
    Stage5ScalarExprPlan { symbol: "stage5.ram_ra_claim_reduction.output.eq.Raf", kind: Stage5ScalarExprKind::StructuredPolynomial { polynomial: bolt_verifier_runtime::StructuredPolynomialKind::Eq, x_point: bolt_verifier_runtime::StructuredPolynomialPointTransform { segment: bolt_verifier_runtime::StructuredPolynomialPointSegment::Full, length: bolt_verifier_runtime::StructuredPolynomialPointLength::Full, order: bolt_verifier_runtime::StructuredPolynomialPointOrder::Reverse }, y_point: bolt_verifier_runtime::StructuredPolynomialPointTransform { segment: bolt_verifier_runtime::StructuredPolynomialPointSegment::Suffix, length: bolt_verifier_runtime::StructuredPolynomialPointLength::XPoint, order: bolt_verifier_runtime::StructuredPolynomialPointOrder::AsIs } }, operands: &["stage5.ram_ra_claim_reduction.instance", "stage5.input.stage2.ram_raf.RamRa"] },
    Stage5ScalarExprPlan { symbol: "stage5.ram_ra_claim_reduction.output.eq.ReadWrite", kind: Stage5ScalarExprKind::StructuredPolynomial { polynomial: bolt_verifier_runtime::StructuredPolynomialKind::Eq, x_point: bolt_verifier_runtime::StructuredPolynomialPointTransform { segment: bolt_verifier_runtime::StructuredPolynomialPointSegment::Full, length: bolt_verifier_runtime::StructuredPolynomialPointLength::Full, order: bolt_verifier_runtime::StructuredPolynomialPointOrder::Reverse }, y_point: bolt_verifier_runtime::StructuredPolynomialPointTransform { segment: bolt_verifier_runtime::StructuredPolynomialPointSegment::Suffix, length: bolt_verifier_runtime::StructuredPolynomialPointLength::XPoint, order: bolt_verifier_runtime::StructuredPolynomialPointOrder::AsIs } }, operands: &["stage5.ram_ra_claim_reduction.instance", "stage5.input.stage2.ram_read_write.RamRa"] },
    Stage5ScalarExprPlan { symbol: "stage5.ram_ra_claim_reduction.output.eq.ValCheck", kind: Stage5ScalarExprKind::StructuredPolynomial { polynomial: bolt_verifier_runtime::StructuredPolynomialKind::Eq, x_point: bolt_verifier_runtime::StructuredPolynomialPointTransform { segment: bolt_verifier_runtime::StructuredPolynomialPointSegment::Full, length: bolt_verifier_runtime::StructuredPolynomialPointLength::Full, order: bolt_verifier_runtime::StructuredPolynomialPointOrder::Reverse }, y_point: bolt_verifier_runtime::StructuredPolynomialPointTransform { segment: bolt_verifier_runtime::StructuredPolynomialPointSegment::Suffix, length: bolt_verifier_runtime::StructuredPolynomialPointLength::XPoint, order: bolt_verifier_runtime::StructuredPolynomialPointOrder::AsIs } }, operands: &["stage5.ram_ra_claim_reduction.instance", "stage5.input.stage4.ram_val_check.RamRa"] },
    Stage5ScalarExprPlan { symbol: "stage5.registers_val_evaluation.output.lt.RegistersValCycle", kind: Stage5ScalarExprKind::StructuredPolynomial { polynomial: bolt_verifier_runtime::StructuredPolynomialKind::Lt, x_point: bolt_verifier_runtime::StructuredPolynomialPointTransform { segment: bolt_verifier_runtime::StructuredPolynomialPointSegment::Full, length: bolt_verifier_runtime::StructuredPolynomialPointLength::Full, order: bolt_verifier_runtime::StructuredPolynomialPointOrder::Reverse }, y_point: bolt_verifier_runtime::StructuredPolynomialPointTransform { segment: bolt_verifier_runtime::StructuredPolynomialPointSegment::Suffix, length: bolt_verifier_runtime::StructuredPolynomialPointLength::XPoint, order: bolt_verifier_runtime::StructuredPolynomialPointOrder::AsIs } }, operands: &["stage5.registers_val_evaluation.instance", "stage5.input.stage4.registers.RegistersVal"] },
    Stage5ScalarExprPlan { symbol: "stage5.instruction_read_raf.output.eq.LookupOutputCycle", kind: Stage5ScalarExprKind::StructuredPolynomial { polynomial: bolt_verifier_runtime::StructuredPolynomialKind::Eq, x_point: bolt_verifier_runtime::StructuredPolynomialPointTransform { segment: bolt_verifier_runtime::StructuredPolynomialPointSegment::Suffix, length: bolt_verifier_runtime::StructuredPolynomialPointLength::YPoint, order: bolt_verifier_runtime::StructuredPolynomialPointOrder::Reverse }, y_point: bolt_verifier_runtime::StructuredPolynomialPointTransform { segment: bolt_verifier_runtime::StructuredPolynomialPointSegment::Full, length: bolt_verifier_runtime::StructuredPolynomialPointLength::Full, order: bolt_verifier_runtime::StructuredPolynomialPointOrder::AsIs } }, operands: &["stage5.instruction_read_raf.instance", "stage5.input.stage2.instruction.LookupOutput"] },
];
pub const STAGE5_KERNELS: &[Stage5KernelPlan] = &[

];

pub const STAGE5_SUMCHECK_CLAIMS: &[Stage5SumcheckClaimPlan] = &[
    Stage5SumcheckClaimPlan { symbol: "stage5.instruction_read_raf.input", stage: "stage5", domain: "jolt.stage5_instruction_read_raf_domain", num_rounds: 144, degree: 10, claim: "stage5.instruction_read_raf.weighted_lookup_values", kernel: None, relation: Some(Stage5RelationKind::Stage5InstructionReadRaf), claim_value: "stage5.instruction_read_raf.claim_expr" },
    Stage5SumcheckClaimPlan { symbol: "stage5.ram_ra_claim_reduction.input", stage: "stage5", domain: "jolt.trace_domain", num_rounds: 16, degree: 2, claim: "stage5.ram_ra_claim_reduction.weighted_ram_ra", kernel: None, relation: Some(Stage5RelationKind::Stage5RamRaClaimReduction), claim_value: "stage5.ram_ra_claim_reduction.claim_expr" },
    Stage5SumcheckClaimPlan { symbol: "stage5.registers_val_evaluation.input", stage: "stage5", domain: "jolt.trace_domain", num_rounds: 16, degree: 3, claim: "stage5.registers_val_evaluation.registers_val", kernel: None, relation: Some(Stage5RelationKind::Stage5RegistersValEvaluation), claim_value: "stage5.input.stage4.registers.RegistersVal" },
];
pub const STAGE5_SUMCHECK_BATCH_0_ROUND_SCHEDULE: &[usize] = &[128, 16];

pub const STAGE5_SUMCHECK_BATCHES: &[Stage5SumcheckBatchPlan] = &[
    Stage5SumcheckBatchPlan { symbol: "stage5.batch", stage: "stage5", proof_slot: "stage5.sumcheck", policy: "jolt_core_stage5_aligned", count: 3, claim_operands: &["stage5.instruction_read_raf.input", "stage5.ram_ra_claim_reduction.input", "stage5.registers_val_evaluation.input"], claim_label: "sumcheck_claim", round_label: "sumcheck_poly", round_schedule: STAGE5_SUMCHECK_BATCH_0_ROUND_SCHEDULE },
];
pub const STAGE5_SUMCHECK_DRIVER_0_ROUND_SCHEDULE: &[usize] = &[128, 16];

pub const STAGE5_SUMCHECK_DRIVERS: &[Stage5SumcheckDriverPlan] = &[
    Stage5SumcheckDriverPlan { symbol: "stage5.sumcheck", stage: "stage5", proof_slot: "stage5.sumcheck", kernel: None, relation: Some(Stage5RelationKind::Stage5Batched), batch: "stage5.batch", policy: "jolt_core_stage5_aligned", round_schedule: STAGE5_SUMCHECK_DRIVER_0_ROUND_SCHEDULE, claim_label: "sumcheck_claim", round_label: "sumcheck_poly", num_rounds: 144, degree: 10 },
];
pub const STAGE5_SUMCHECK_INSTANCE_RESULTS: &[Stage5SumcheckInstanceResultPlan] = &[
    Stage5SumcheckInstanceResultPlan { symbol: "stage5.instruction_read_raf.instance", source: "stage5.sumcheck", claim: "stage5.instruction_read_raf.input", relation: Stage5RelationKind::Stage5InstructionReadRaf, index: 0, point_arity: 144, num_rounds: 144, round_offset: 0, point_order: bolt_verifier_runtime::SumcheckPointOrder::InstructionReadRaf, degree: 10 },
    Stage5SumcheckInstanceResultPlan { symbol: "stage5.ram_ra_claim_reduction.instance", source: "stage5.sumcheck", claim: "stage5.ram_ra_claim_reduction.input", relation: Stage5RelationKind::Stage5RamRaClaimReduction, index: 1, point_arity: 16, num_rounds: 16, round_offset: 128, point_order: bolt_verifier_runtime::SumcheckPointOrder::Reverse, degree: 2 },
    Stage5SumcheckInstanceResultPlan { symbol: "stage5.registers_val_evaluation.instance", source: "stage5.sumcheck", claim: "stage5.registers_val_evaluation.input", relation: Stage5RelationKind::Stage5RegistersValEvaluation, index: 2, point_arity: 16, num_rounds: 16, round_offset: 128, point_order: bolt_verifier_runtime::SumcheckPointOrder::Reverse, degree: 3 },
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
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_40", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_40", index: 40, oracle: "LookupTableFlag_40" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.InstructionRa_0", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.InstructionRa_0", index: 41, oracle: "InstructionRa_0" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.InstructionRa_1", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.InstructionRa_1", index: 42, oracle: "InstructionRa_1" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.InstructionRa_2", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.InstructionRa_2", index: 43, oracle: "InstructionRa_2" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.InstructionRa_3", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.InstructionRa_3", index: 44, oracle: "InstructionRa_3" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.InstructionRa_4", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.InstructionRa_4", index: 45, oracle: "InstructionRa_4" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.InstructionRa_5", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.InstructionRa_5", index: 46, oracle: "InstructionRa_5" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.InstructionRa_6", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.InstructionRa_6", index: 47, oracle: "InstructionRa_6" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.InstructionRa_7", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.InstructionRa_7", index: 48, oracle: "InstructionRa_7" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.InstructionRafFlag", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.InstructionRafFlag", index: 49, oracle: "InstructionRafFlag" },
    Stage5SumcheckEvalPlan { symbol: "stage5.ram_ra_claim_reduction.eval.RamRa", source: "stage5.sumcheck", name: "stage5.ram_ra_claim_reduction.eval.RamRa", index: 0, oracle: "RamRa" },
    Stage5SumcheckEvalPlan { symbol: "stage5.registers_val_evaluation.eval.RdInc", source: "stage5.sumcheck", name: "stage5.registers_val_evaluation.eval.RdInc", index: 0, oracle: "RdInc" },
    Stage5SumcheckEvalPlan { symbol: "stage5.registers_val_evaluation.eval.RdWa", source: "stage5.sumcheck", name: "stage5.registers_val_evaluation.eval.RdWa", index: 1, oracle: "RdWa" },
];

pub const STAGE5_INSTRUCTION_READ_RAF_LOCAL_SCALARS: &[JoltLocalScalarPlan] = &[
    JoltLocalScalarPlan { symbol: "stage5.instruction_read_raf.local_scalar.LookupTable_0", kind: JoltLocalScalarMleKind::LookupTable { index: 0 } },
    JoltLocalScalarPlan { symbol: "stage5.instruction_read_raf.local_scalar.LookupTable_1", kind: JoltLocalScalarMleKind::LookupTable { index: 1 } },
    JoltLocalScalarPlan { symbol: "stage5.instruction_read_raf.local_scalar.LookupTable_2", kind: JoltLocalScalarMleKind::LookupTable { index: 2 } },
    JoltLocalScalarPlan { symbol: "stage5.instruction_read_raf.local_scalar.LookupTable_3", kind: JoltLocalScalarMleKind::LookupTable { index: 3 } },
    JoltLocalScalarPlan { symbol: "stage5.instruction_read_raf.local_scalar.LookupTable_4", kind: JoltLocalScalarMleKind::LookupTable { index: 4 } },
    JoltLocalScalarPlan { symbol: "stage5.instruction_read_raf.local_scalar.LookupTable_5", kind: JoltLocalScalarMleKind::LookupTable { index: 5 } },
    JoltLocalScalarPlan { symbol: "stage5.instruction_read_raf.local_scalar.LookupTable_6", kind: JoltLocalScalarMleKind::LookupTable { index: 6 } },
    JoltLocalScalarPlan { symbol: "stage5.instruction_read_raf.local_scalar.LookupTable_7", kind: JoltLocalScalarMleKind::LookupTable { index: 7 } },
    JoltLocalScalarPlan { symbol: "stage5.instruction_read_raf.local_scalar.LookupTable_8", kind: JoltLocalScalarMleKind::LookupTable { index: 8 } },
    JoltLocalScalarPlan { symbol: "stage5.instruction_read_raf.local_scalar.LookupTable_9", kind: JoltLocalScalarMleKind::LookupTable { index: 9 } },
    JoltLocalScalarPlan { symbol: "stage5.instruction_read_raf.local_scalar.LookupTable_10", kind: JoltLocalScalarMleKind::LookupTable { index: 10 } },
    JoltLocalScalarPlan { symbol: "stage5.instruction_read_raf.local_scalar.LookupTable_11", kind: JoltLocalScalarMleKind::LookupTable { index: 11 } },
    JoltLocalScalarPlan { symbol: "stage5.instruction_read_raf.local_scalar.LookupTable_12", kind: JoltLocalScalarMleKind::LookupTable { index: 12 } },
    JoltLocalScalarPlan { symbol: "stage5.instruction_read_raf.local_scalar.LookupTable_13", kind: JoltLocalScalarMleKind::LookupTable { index: 13 } },
    JoltLocalScalarPlan { symbol: "stage5.instruction_read_raf.local_scalar.LookupTable_14", kind: JoltLocalScalarMleKind::LookupTable { index: 14 } },
    JoltLocalScalarPlan { symbol: "stage5.instruction_read_raf.local_scalar.LookupTable_15", kind: JoltLocalScalarMleKind::LookupTable { index: 15 } },
    JoltLocalScalarPlan { symbol: "stage5.instruction_read_raf.local_scalar.LookupTable_16", kind: JoltLocalScalarMleKind::LookupTable { index: 16 } },
    JoltLocalScalarPlan { symbol: "stage5.instruction_read_raf.local_scalar.LookupTable_17", kind: JoltLocalScalarMleKind::LookupTable { index: 17 } },
    JoltLocalScalarPlan { symbol: "stage5.instruction_read_raf.local_scalar.LookupTable_18", kind: JoltLocalScalarMleKind::LookupTable { index: 18 } },
    JoltLocalScalarPlan { symbol: "stage5.instruction_read_raf.local_scalar.LookupTable_19", kind: JoltLocalScalarMleKind::LookupTable { index: 19 } },
    JoltLocalScalarPlan { symbol: "stage5.instruction_read_raf.local_scalar.LookupTable_20", kind: JoltLocalScalarMleKind::LookupTable { index: 20 } },
    JoltLocalScalarPlan { symbol: "stage5.instruction_read_raf.local_scalar.LookupTable_21", kind: JoltLocalScalarMleKind::LookupTable { index: 21 } },
    JoltLocalScalarPlan { symbol: "stage5.instruction_read_raf.local_scalar.LookupTable_22", kind: JoltLocalScalarMleKind::LookupTable { index: 22 } },
    JoltLocalScalarPlan { symbol: "stage5.instruction_read_raf.local_scalar.LookupTable_23", kind: JoltLocalScalarMleKind::LookupTable { index: 23 } },
    JoltLocalScalarPlan { symbol: "stage5.instruction_read_raf.local_scalar.LookupTable_24", kind: JoltLocalScalarMleKind::LookupTable { index: 24 } },
    JoltLocalScalarPlan { symbol: "stage5.instruction_read_raf.local_scalar.LookupTable_25", kind: JoltLocalScalarMleKind::LookupTable { index: 25 } },
    JoltLocalScalarPlan { symbol: "stage5.instruction_read_raf.local_scalar.LookupTable_26", kind: JoltLocalScalarMleKind::LookupTable { index: 26 } },
    JoltLocalScalarPlan { symbol: "stage5.instruction_read_raf.local_scalar.LookupTable_27", kind: JoltLocalScalarMleKind::LookupTable { index: 27 } },
    JoltLocalScalarPlan { symbol: "stage5.instruction_read_raf.local_scalar.LookupTable_28", kind: JoltLocalScalarMleKind::LookupTable { index: 28 } },
    JoltLocalScalarPlan { symbol: "stage5.instruction_read_raf.local_scalar.LookupTable_29", kind: JoltLocalScalarMleKind::LookupTable { index: 29 } },
    JoltLocalScalarPlan { symbol: "stage5.instruction_read_raf.local_scalar.LookupTable_30", kind: JoltLocalScalarMleKind::LookupTable { index: 30 } },
    JoltLocalScalarPlan { symbol: "stage5.instruction_read_raf.local_scalar.LookupTable_31", kind: JoltLocalScalarMleKind::LookupTable { index: 31 } },
    JoltLocalScalarPlan { symbol: "stage5.instruction_read_raf.local_scalar.LookupTable_32", kind: JoltLocalScalarMleKind::LookupTable { index: 32 } },
    JoltLocalScalarPlan { symbol: "stage5.instruction_read_raf.local_scalar.LookupTable_33", kind: JoltLocalScalarMleKind::LookupTable { index: 33 } },
    JoltLocalScalarPlan { symbol: "stage5.instruction_read_raf.local_scalar.LookupTable_34", kind: JoltLocalScalarMleKind::LookupTable { index: 34 } },
    JoltLocalScalarPlan { symbol: "stage5.instruction_read_raf.local_scalar.LookupTable_35", kind: JoltLocalScalarMleKind::LookupTable { index: 35 } },
    JoltLocalScalarPlan { symbol: "stage5.instruction_read_raf.local_scalar.LookupTable_36", kind: JoltLocalScalarMleKind::LookupTable { index: 36 } },
    JoltLocalScalarPlan { symbol: "stage5.instruction_read_raf.local_scalar.LookupTable_37", kind: JoltLocalScalarMleKind::LookupTable { index: 37 } },
    JoltLocalScalarPlan { symbol: "stage5.instruction_read_raf.local_scalar.LookupTable_38", kind: JoltLocalScalarMleKind::LookupTable { index: 38 } },
    JoltLocalScalarPlan { symbol: "stage5.instruction_read_raf.local_scalar.LookupTable_39", kind: JoltLocalScalarMleKind::LookupTable { index: 39 } },
    JoltLocalScalarPlan { symbol: "stage5.instruction_read_raf.local_scalar.LookupTable_40", kind: JoltLocalScalarMleKind::LookupTable { index: 40 } },
    JoltLocalScalarPlan { symbol: "stage5.instruction_read_raf.local_scalar.LeftLookupOperand", kind: JoltLocalScalarMleKind::LeftOperand },
    JoltLocalScalarPlan { symbol: "stage5.instruction_read_raf.local_scalar.RightLookupOperand", kind: JoltLocalScalarMleKind::RightOperand },
    JoltLocalScalarPlan { symbol: "stage5.instruction_read_raf.local_scalar.Identity", kind: JoltLocalScalarMleKind::Identity },
];

pub const STAGE5_INSTRUCTION_READ_RAF_PLAN: Stage5InstructionReadRafPlan = Stage5InstructionReadRafPlan {
    point: "stage5.instruction_read_raf.instance",
    lookup_output_point: "stage5.input.stage2.instruction.LookupOutput",
    table_flag_evals: &STAGE5_INDEXED_EVAL_FAMILIES[0],
    instruction_ra_evals: &STAGE5_INDEXED_EVAL_FAMILIES[1],
    raf_flag_eval: "stage5.instruction_read_raf.eval.InstructionRafFlag",
    gamma: "stage5.instruction_read_raf.gamma",
    local_scalars: STAGE5_INSTRUCTION_READ_RAF_LOCAL_SCALARS,
    log_k: 128,
};

pub const STAGE5_POINT_EXPRS: &[Stage5PointExprPlan] = &[
    Stage5PointExprPlan { symbol: "stage5.instruction_read_raf.point.Cycle", kind: Stage5PointExprKind::Slice { offset: 128, length: 16 }, operands: &["stage5.instruction_read_raf.instance"] },
    Stage5PointExprPlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_0.address", kind: Stage5PointExprKind::Slice { offset: 0, length: 16 }, operands: &["stage5.instruction_read_raf.instance"] },
    Stage5PointExprPlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_1.address", kind: Stage5PointExprKind::Slice { offset: 16, length: 16 }, operands: &["stage5.instruction_read_raf.instance"] },
    Stage5PointExprPlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_2.address", kind: Stage5PointExprKind::Slice { offset: 32, length: 16 }, operands: &["stage5.instruction_read_raf.instance"] },
    Stage5PointExprPlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_3.address", kind: Stage5PointExprKind::Slice { offset: 48, length: 16 }, operands: &["stage5.instruction_read_raf.instance"] },
    Stage5PointExprPlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_4.address", kind: Stage5PointExprKind::Slice { offset: 64, length: 16 }, operands: &["stage5.instruction_read_raf.instance"] },
    Stage5PointExprPlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_5.address", kind: Stage5PointExprKind::Slice { offset: 80, length: 16 }, operands: &["stage5.instruction_read_raf.instance"] },
    Stage5PointExprPlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_6.address", kind: Stage5PointExprKind::Slice { offset: 96, length: 16 }, operands: &["stage5.instruction_read_raf.instance"] },
    Stage5PointExprPlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_7.address", kind: Stage5PointExprKind::Slice { offset: 112, length: 16 }, operands: &["stage5.instruction_read_raf.instance"] },
    Stage5PointExprPlan { symbol: "stage5.ram_ra_claim_reduction.point.RamAddress", kind: Stage5PointExprKind::Slice { offset: 0, length: 16 }, operands: &["stage5.input.stage2.ram_raf.RamRa"] },
    Stage5PointExprPlan { symbol: "stage5.registers_val_evaluation.point.RegisterAddress", kind: Stage5PointExprKind::Slice { offset: 0, length: 7 }, operands: &["stage5.input.stage4.registers.RegistersVal"] },
    Stage5PointExprPlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_0", kind: Stage5PointExprKind::Concat { layout: "address_chunk_then_cycle", arity: 32 }, operands: &["stage5.instruction_read_raf.point.InstructionRa_0.address", "stage5.instruction_read_raf.point.Cycle"] },
    Stage5PointExprPlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_1", kind: Stage5PointExprKind::Concat { layout: "address_chunk_then_cycle", arity: 32 }, operands: &["stage5.instruction_read_raf.point.InstructionRa_1.address", "stage5.instruction_read_raf.point.Cycle"] },
    Stage5PointExprPlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_2", kind: Stage5PointExprKind::Concat { layout: "address_chunk_then_cycle", arity: 32 }, operands: &["stage5.instruction_read_raf.point.InstructionRa_2.address", "stage5.instruction_read_raf.point.Cycle"] },
    Stage5PointExprPlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_3", kind: Stage5PointExprKind::Concat { layout: "address_chunk_then_cycle", arity: 32 }, operands: &["stage5.instruction_read_raf.point.InstructionRa_3.address", "stage5.instruction_read_raf.point.Cycle"] },
    Stage5PointExprPlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_4", kind: Stage5PointExprKind::Concat { layout: "address_chunk_then_cycle", arity: 32 }, operands: &["stage5.instruction_read_raf.point.InstructionRa_4.address", "stage5.instruction_read_raf.point.Cycle"] },
    Stage5PointExprPlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_5", kind: Stage5PointExprKind::Concat { layout: "address_chunk_then_cycle", arity: 32 }, operands: &["stage5.instruction_read_raf.point.InstructionRa_5.address", "stage5.instruction_read_raf.point.Cycle"] },
    Stage5PointExprPlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_6", kind: Stage5PointExprKind::Concat { layout: "address_chunk_then_cycle", arity: 32 }, operands: &["stage5.instruction_read_raf.point.InstructionRa_6.address", "stage5.instruction_read_raf.point.Cycle"] },
    Stage5PointExprPlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_7", kind: Stage5PointExprKind::Concat { layout: "address_chunk_then_cycle", arity: 32 }, operands: &["stage5.instruction_read_raf.point.InstructionRa_7.address", "stage5.instruction_read_raf.point.Cycle"] },
    Stage5PointExprPlan { symbol: "stage5.ram_ra_claim_reduction.point.RamRa", kind: Stage5PointExprKind::Concat { layout: "address_then_cycle", arity: 32 }, operands: &["stage5.ram_ra_claim_reduction.point.RamAddress", "stage5.ram_ra_claim_reduction.instance"] },
    Stage5PointExprPlan { symbol: "stage5.registers_val_evaluation.point.RdWa", kind: Stage5PointExprKind::Concat { layout: "register_address_then_cycle", arity: 23 }, operands: &["stage5.registers_val_evaluation.point.RegisterAddress", "stage5.registers_val_evaluation.instance"] },
];

pub const STAGE5_OPENING_CLAIMS: &[Stage5OpeningClaimPlan] = &[
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_0", oracle: "LookupTableFlag_0", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_0" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_1", oracle: "LookupTableFlag_1", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_1" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_2", oracle: "LookupTableFlag_2", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_2" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_3", oracle: "LookupTableFlag_3", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_3" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_4", oracle: "LookupTableFlag_4", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_4" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_5", oracle: "LookupTableFlag_5", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_5" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_6", oracle: "LookupTableFlag_6", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_6" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_7", oracle: "LookupTableFlag_7", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_7" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_8", oracle: "LookupTableFlag_8", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_8" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_9", oracle: "LookupTableFlag_9", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_9" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_10", oracle: "LookupTableFlag_10", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_10" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_11", oracle: "LookupTableFlag_11", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_11" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_12", oracle: "LookupTableFlag_12", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_12" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_13", oracle: "LookupTableFlag_13", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_13" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_14", oracle: "LookupTableFlag_14", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_14" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_15", oracle: "LookupTableFlag_15", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_15" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_16", oracle: "LookupTableFlag_16", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_16" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_17", oracle: "LookupTableFlag_17", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_17" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_18", oracle: "LookupTableFlag_18", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_18" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_19", oracle: "LookupTableFlag_19", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_19" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_20", oracle: "LookupTableFlag_20", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_20" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_21", oracle: "LookupTableFlag_21", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_21" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_22", oracle: "LookupTableFlag_22", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_22" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_23", oracle: "LookupTableFlag_23", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_23" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_24", oracle: "LookupTableFlag_24", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_24" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_25", oracle: "LookupTableFlag_25", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_25" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_26", oracle: "LookupTableFlag_26", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_26" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_27", oracle: "LookupTableFlag_27", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_27" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_28", oracle: "LookupTableFlag_28", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_28" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_29", oracle: "LookupTableFlag_29", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_29" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_30", oracle: "LookupTableFlag_30", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_30" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_31", oracle: "LookupTableFlag_31", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_31" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_32", oracle: "LookupTableFlag_32", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_32" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_33", oracle: "LookupTableFlag_33", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_33" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_34", oracle: "LookupTableFlag_34", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_34" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_35", oracle: "LookupTableFlag_35", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_35" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_36", oracle: "LookupTableFlag_36", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_36" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_37", oracle: "LookupTableFlag_37", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_37" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_38", oracle: "LookupTableFlag_38", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_38" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_39", oracle: "LookupTableFlag_39", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_39" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_40", oracle: "LookupTableFlag_40", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_40" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.InstructionRa_0", oracle: "InstructionRa_0", domain: "jolt.stage5_instruction_ra_chunk_domain", point_arity: 32, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.instruction_read_raf.point.InstructionRa_0", eval_source: "stage5.instruction_read_raf.eval.InstructionRa_0" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.InstructionRa_1", oracle: "InstructionRa_1", domain: "jolt.stage5_instruction_ra_chunk_domain", point_arity: 32, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.instruction_read_raf.point.InstructionRa_1", eval_source: "stage5.instruction_read_raf.eval.InstructionRa_1" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.InstructionRa_2", oracle: "InstructionRa_2", domain: "jolt.stage5_instruction_ra_chunk_domain", point_arity: 32, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.instruction_read_raf.point.InstructionRa_2", eval_source: "stage5.instruction_read_raf.eval.InstructionRa_2" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.InstructionRa_3", oracle: "InstructionRa_3", domain: "jolt.stage5_instruction_ra_chunk_domain", point_arity: 32, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.instruction_read_raf.point.InstructionRa_3", eval_source: "stage5.instruction_read_raf.eval.InstructionRa_3" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.InstructionRa_4", oracle: "InstructionRa_4", domain: "jolt.stage5_instruction_ra_chunk_domain", point_arity: 32, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.instruction_read_raf.point.InstructionRa_4", eval_source: "stage5.instruction_read_raf.eval.InstructionRa_4" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.InstructionRa_5", oracle: "InstructionRa_5", domain: "jolt.stage5_instruction_ra_chunk_domain", point_arity: 32, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.instruction_read_raf.point.InstructionRa_5", eval_source: "stage5.instruction_read_raf.eval.InstructionRa_5" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.InstructionRa_6", oracle: "InstructionRa_6", domain: "jolt.stage5_instruction_ra_chunk_domain", point_arity: 32, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.instruction_read_raf.point.InstructionRa_6", eval_source: "stage5.instruction_read_raf.eval.InstructionRa_6" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.InstructionRa_7", oracle: "InstructionRa_7", domain: "jolt.stage5_instruction_ra_chunk_domain", point_arity: 32, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.instruction_read_raf.point.InstructionRa_7", eval_source: "stage5.instruction_read_raf.eval.InstructionRa_7" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.InstructionRafFlag", oracle: "InstructionRafFlag", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.InstructionRafFlag" },
    Stage5OpeningClaimPlan { symbol: "stage5.ram_ra_claim_reduction.opening.RamRa", oracle: "RamRa", domain: "jolt.stage2_ram_rw_domain", point_arity: 32, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.ram_ra_claim_reduction.point.RamRa", eval_source: "stage5.ram_ra_claim_reduction.eval.RamRa" },
    Stage5OpeningClaimPlan { symbol: "stage5.registers_val_evaluation.opening.RdInc", oracle: "RdInc", domain: "jolt.trace_domain", point_arity: 16, claim_kind: Stage5ClaimKind::Committed, point_source: "stage5.registers_val_evaluation.instance", eval_source: "stage5.registers_val_evaluation.eval.RdInc" },
    Stage5OpeningClaimPlan { symbol: "stage5.registers_val_evaluation.opening.RdWa", oracle: "RdWa", domain: "jolt.stage4_registers_rw_domain", point_arity: 23, claim_kind: Stage5ClaimKind::Virtual, point_source: "stage5.registers_val_evaluation.point.RdWa", eval_source: "stage5.registers_val_evaluation.eval.RdWa" },
];

pub const STAGE5_OPENING_EQUALITIES: &[Stage5OpeningClaimEqualityPlan] = &[
    Stage5OpeningClaimEqualityPlan { symbol: "stage5.instruction.lookup_output_claim_consistency", mode: Stage5OpeningEqualityMode::PointAndEval, lhs: "stage5.input.stage2.instruction.LookupOutput", rhs: "stage5.input.stage2.product_virtual.LookupOutput" },
];

pub const STAGE5_OPENING_BATCHES: &[Stage5OpeningBatchPlan] = &[
    Stage5OpeningBatchPlan { symbol: "stage5.openings", stage: "stage5", proof_slot: "stage5.openings", policy: "jolt_stage5_output_order", count: 53, ordered_claims: &["stage5.instruction_read_raf.opening.LookupTableFlag_0", "stage5.instruction_read_raf.opening.LookupTableFlag_1", "stage5.instruction_read_raf.opening.LookupTableFlag_2", "stage5.instruction_read_raf.opening.LookupTableFlag_3", "stage5.instruction_read_raf.opening.LookupTableFlag_4", "stage5.instruction_read_raf.opening.LookupTableFlag_5", "stage5.instruction_read_raf.opening.LookupTableFlag_6", "stage5.instruction_read_raf.opening.LookupTableFlag_7", "stage5.instruction_read_raf.opening.LookupTableFlag_8", "stage5.instruction_read_raf.opening.LookupTableFlag_9", "stage5.instruction_read_raf.opening.LookupTableFlag_10", "stage5.instruction_read_raf.opening.LookupTableFlag_11", "stage5.instruction_read_raf.opening.LookupTableFlag_12", "stage5.instruction_read_raf.opening.LookupTableFlag_13", "stage5.instruction_read_raf.opening.LookupTableFlag_14", "stage5.instruction_read_raf.opening.LookupTableFlag_15", "stage5.instruction_read_raf.opening.LookupTableFlag_16", "stage5.instruction_read_raf.opening.LookupTableFlag_17", "stage5.instruction_read_raf.opening.LookupTableFlag_18", "stage5.instruction_read_raf.opening.LookupTableFlag_19", "stage5.instruction_read_raf.opening.LookupTableFlag_20", "stage5.instruction_read_raf.opening.LookupTableFlag_21", "stage5.instruction_read_raf.opening.LookupTableFlag_22", "stage5.instruction_read_raf.opening.LookupTableFlag_23", "stage5.instruction_read_raf.opening.LookupTableFlag_24", "stage5.instruction_read_raf.opening.LookupTableFlag_25", "stage5.instruction_read_raf.opening.LookupTableFlag_26", "stage5.instruction_read_raf.opening.LookupTableFlag_27", "stage5.instruction_read_raf.opening.LookupTableFlag_28", "stage5.instruction_read_raf.opening.LookupTableFlag_29", "stage5.instruction_read_raf.opening.LookupTableFlag_30", "stage5.instruction_read_raf.opening.LookupTableFlag_31", "stage5.instruction_read_raf.opening.LookupTableFlag_32", "stage5.instruction_read_raf.opening.LookupTableFlag_33", "stage5.instruction_read_raf.opening.LookupTableFlag_34", "stage5.instruction_read_raf.opening.LookupTableFlag_35", "stage5.instruction_read_raf.opening.LookupTableFlag_36", "stage5.instruction_read_raf.opening.LookupTableFlag_37", "stage5.instruction_read_raf.opening.LookupTableFlag_38", "stage5.instruction_read_raf.opening.LookupTableFlag_39", "stage5.instruction_read_raf.opening.LookupTableFlag_40", "stage5.instruction_read_raf.opening.InstructionRa_0", "stage5.instruction_read_raf.opening.InstructionRa_1", "stage5.instruction_read_raf.opening.InstructionRa_2", "stage5.instruction_read_raf.opening.InstructionRa_3", "stage5.instruction_read_raf.opening.InstructionRa_4", "stage5.instruction_read_raf.opening.InstructionRa_5", "stage5.instruction_read_raf.opening.InstructionRa_6", "stage5.instruction_read_raf.opening.InstructionRa_7", "stage5.instruction_read_raf.opening.InstructionRafFlag", "stage5.ram_ra_claim_reduction.opening.RamRa", "stage5.registers_val_evaluation.opening.RdInc", "stage5.registers_val_evaluation.opening.RdWa"], claim_operands: &["stage5.instruction_read_raf.opening.LookupTableFlag_0", "stage5.instruction_read_raf.opening.LookupTableFlag_1", "stage5.instruction_read_raf.opening.LookupTableFlag_2", "stage5.instruction_read_raf.opening.LookupTableFlag_3", "stage5.instruction_read_raf.opening.LookupTableFlag_4", "stage5.instruction_read_raf.opening.LookupTableFlag_5", "stage5.instruction_read_raf.opening.LookupTableFlag_6", "stage5.instruction_read_raf.opening.LookupTableFlag_7", "stage5.instruction_read_raf.opening.LookupTableFlag_8", "stage5.instruction_read_raf.opening.LookupTableFlag_9", "stage5.instruction_read_raf.opening.LookupTableFlag_10", "stage5.instruction_read_raf.opening.LookupTableFlag_11", "stage5.instruction_read_raf.opening.LookupTableFlag_12", "stage5.instruction_read_raf.opening.LookupTableFlag_13", "stage5.instruction_read_raf.opening.LookupTableFlag_14", "stage5.instruction_read_raf.opening.LookupTableFlag_15", "stage5.instruction_read_raf.opening.LookupTableFlag_16", "stage5.instruction_read_raf.opening.LookupTableFlag_17", "stage5.instruction_read_raf.opening.LookupTableFlag_18", "stage5.instruction_read_raf.opening.LookupTableFlag_19", "stage5.instruction_read_raf.opening.LookupTableFlag_20", "stage5.instruction_read_raf.opening.LookupTableFlag_21", "stage5.instruction_read_raf.opening.LookupTableFlag_22", "stage5.instruction_read_raf.opening.LookupTableFlag_23", "stage5.instruction_read_raf.opening.LookupTableFlag_24", "stage5.instruction_read_raf.opening.LookupTableFlag_25", "stage5.instruction_read_raf.opening.LookupTableFlag_26", "stage5.instruction_read_raf.opening.LookupTableFlag_27", "stage5.instruction_read_raf.opening.LookupTableFlag_28", "stage5.instruction_read_raf.opening.LookupTableFlag_29", "stage5.instruction_read_raf.opening.LookupTableFlag_30", "stage5.instruction_read_raf.opening.LookupTableFlag_31", "stage5.instruction_read_raf.opening.LookupTableFlag_32", "stage5.instruction_read_raf.opening.LookupTableFlag_33", "stage5.instruction_read_raf.opening.LookupTableFlag_34", "stage5.instruction_read_raf.opening.LookupTableFlag_35", "stage5.instruction_read_raf.opening.LookupTableFlag_36", "stage5.instruction_read_raf.opening.LookupTableFlag_37", "stage5.instruction_read_raf.opening.LookupTableFlag_38", "stage5.instruction_read_raf.opening.LookupTableFlag_39", "stage5.instruction_read_raf.opening.LookupTableFlag_40", "stage5.instruction_read_raf.opening.InstructionRa_0", "stage5.instruction_read_raf.opening.InstructionRa_1", "stage5.instruction_read_raf.opening.InstructionRa_2", "stage5.instruction_read_raf.opening.InstructionRa_3", "stage5.instruction_read_raf.opening.InstructionRa_4", "stage5.instruction_read_raf.opening.InstructionRa_5", "stage5.instruction_read_raf.opening.InstructionRa_6", "stage5.instruction_read_raf.opening.InstructionRa_7", "stage5.instruction_read_raf.opening.InstructionRafFlag", "stage5.ram_ra_claim_reduction.opening.RamRa", "stage5.registers_val_evaluation.opening.RdInc", "stage5.registers_val_evaluation.opening.RdWa"] },
];
#[rustfmt::skip]
const STAGE5_INDEXED_EVAL_FAMILY_0_NAMES: &[&str] = &["stage5.instruction_read_raf.eval.LookupTableFlag_0", "stage5.instruction_read_raf.eval.LookupTableFlag_1", "stage5.instruction_read_raf.eval.LookupTableFlag_2", "stage5.instruction_read_raf.eval.LookupTableFlag_3", "stage5.instruction_read_raf.eval.LookupTableFlag_4", "stage5.instruction_read_raf.eval.LookupTableFlag_5", "stage5.instruction_read_raf.eval.LookupTableFlag_6", "stage5.instruction_read_raf.eval.LookupTableFlag_7", "stage5.instruction_read_raf.eval.LookupTableFlag_8", "stage5.instruction_read_raf.eval.LookupTableFlag_9", "stage5.instruction_read_raf.eval.LookupTableFlag_10", "stage5.instruction_read_raf.eval.LookupTableFlag_11", "stage5.instruction_read_raf.eval.LookupTableFlag_12", "stage5.instruction_read_raf.eval.LookupTableFlag_13", "stage5.instruction_read_raf.eval.LookupTableFlag_14", "stage5.instruction_read_raf.eval.LookupTableFlag_15", "stage5.instruction_read_raf.eval.LookupTableFlag_16", "stage5.instruction_read_raf.eval.LookupTableFlag_17", "stage5.instruction_read_raf.eval.LookupTableFlag_18", "stage5.instruction_read_raf.eval.LookupTableFlag_19", "stage5.instruction_read_raf.eval.LookupTableFlag_20", "stage5.instruction_read_raf.eval.LookupTableFlag_21", "stage5.instruction_read_raf.eval.LookupTableFlag_22", "stage5.instruction_read_raf.eval.LookupTableFlag_23", "stage5.instruction_read_raf.eval.LookupTableFlag_24", "stage5.instruction_read_raf.eval.LookupTableFlag_25", "stage5.instruction_read_raf.eval.LookupTableFlag_26", "stage5.instruction_read_raf.eval.LookupTableFlag_27", "stage5.instruction_read_raf.eval.LookupTableFlag_28", "stage5.instruction_read_raf.eval.LookupTableFlag_29", "stage5.instruction_read_raf.eval.LookupTableFlag_30", "stage5.instruction_read_raf.eval.LookupTableFlag_31", "stage5.instruction_read_raf.eval.LookupTableFlag_32", "stage5.instruction_read_raf.eval.LookupTableFlag_33", "stage5.instruction_read_raf.eval.LookupTableFlag_34", "stage5.instruction_read_raf.eval.LookupTableFlag_35", "stage5.instruction_read_raf.eval.LookupTableFlag_36", "stage5.instruction_read_raf.eval.LookupTableFlag_37", "stage5.instruction_read_raf.eval.LookupTableFlag_38", "stage5.instruction_read_raf.eval.LookupTableFlag_39", "stage5.instruction_read_raf.eval.LookupTableFlag_40"];
#[rustfmt::skip]
const STAGE5_INDEXED_EVAL_FAMILY_1_NAMES: &[&str] = &["stage5.instruction_read_raf.eval.InstructionRa_0", "stage5.instruction_read_raf.eval.InstructionRa_1", "stage5.instruction_read_raf.eval.InstructionRa_2", "stage5.instruction_read_raf.eval.InstructionRa_3", "stage5.instruction_read_raf.eval.InstructionRa_4", "stage5.instruction_read_raf.eval.InstructionRa_5", "stage5.instruction_read_raf.eval.InstructionRa_6", "stage5.instruction_read_raf.eval.InstructionRa_7"];
pub const STAGE5_INDEXED_EVAL_FAMILIES: &[bolt_verifier_runtime::NamedEvalFamilyPlan] = &[
    bolt_verifier_runtime::NamedEvalFamilyPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag", evals: STAGE5_INDEXED_EVAL_FAMILY_0_NAMES },
    bolt_verifier_runtime::NamedEvalFamilyPlan { symbol: "stage5.instruction_read_raf.eval.InstructionRa", evals: STAGE5_INDEXED_EVAL_FAMILY_1_NAMES },
];

pub const STAGE5_RELATION_OUTPUT_2_LOCAL_SCALARS: &[&str] = &["stage5.instruction_read_raf.local_scalar.LookupTable_0", "stage5.instruction_read_raf.local_scalar.LookupTable_1", "stage5.instruction_read_raf.local_scalar.LookupTable_2", "stage5.instruction_read_raf.local_scalar.LookupTable_3", "stage5.instruction_read_raf.local_scalar.LookupTable_4", "stage5.instruction_read_raf.local_scalar.LookupTable_5", "stage5.instruction_read_raf.local_scalar.LookupTable_6", "stage5.instruction_read_raf.local_scalar.LookupTable_7", "stage5.instruction_read_raf.local_scalar.LookupTable_8", "stage5.instruction_read_raf.local_scalar.LookupTable_9", "stage5.instruction_read_raf.local_scalar.LookupTable_10", "stage5.instruction_read_raf.local_scalar.LookupTable_11", "stage5.instruction_read_raf.local_scalar.LookupTable_12", "stage5.instruction_read_raf.local_scalar.LookupTable_13", "stage5.instruction_read_raf.local_scalar.LookupTable_14", "stage5.instruction_read_raf.local_scalar.LookupTable_15", "stage5.instruction_read_raf.local_scalar.LookupTable_16", "stage5.instruction_read_raf.local_scalar.LookupTable_17", "stage5.instruction_read_raf.local_scalar.LookupTable_18", "stage5.instruction_read_raf.local_scalar.LookupTable_19", "stage5.instruction_read_raf.local_scalar.LookupTable_20", "stage5.instruction_read_raf.local_scalar.LookupTable_21", "stage5.instruction_read_raf.local_scalar.LookupTable_22", "stage5.instruction_read_raf.local_scalar.LookupTable_23", "stage5.instruction_read_raf.local_scalar.LookupTable_24", "stage5.instruction_read_raf.local_scalar.LookupTable_25", "stage5.instruction_read_raf.local_scalar.LookupTable_26", "stage5.instruction_read_raf.local_scalar.LookupTable_27", "stage5.instruction_read_raf.local_scalar.LookupTable_28", "stage5.instruction_read_raf.local_scalar.LookupTable_29", "stage5.instruction_read_raf.local_scalar.LookupTable_30", "stage5.instruction_read_raf.local_scalar.LookupTable_31", "stage5.instruction_read_raf.local_scalar.LookupTable_32", "stage5.instruction_read_raf.local_scalar.LookupTable_33", "stage5.instruction_read_raf.local_scalar.LookupTable_34", "stage5.instruction_read_raf.local_scalar.LookupTable_35", "stage5.instruction_read_raf.local_scalar.LookupTable_36", "stage5.instruction_read_raf.local_scalar.LookupTable_37", "stage5.instruction_read_raf.local_scalar.LookupTable_38", "stage5.instruction_read_raf.local_scalar.LookupTable_39", "stage5.instruction_read_raf.local_scalar.LookupTable_40", "stage5.instruction_read_raf.local_scalar.LeftLookupOperand", "stage5.instruction_read_raf.local_scalar.RightLookupOperand", "stage5.instruction_read_raf.local_scalar.Identity"];
pub const STAGE5_RELATION_OUTPUTS: &[Stage5RelationOutputPlan] = &[
    Stage5RelationOutputPlan { relation: Stage5RelationKind::Stage5RamRaClaimReduction, local_scalars: &[], expected_output: "stage5.ram_ra_claim_reduction.output.claim_expr" },
    Stage5RelationOutputPlan { relation: Stage5RelationKind::Stage5RegistersValEvaluation, local_scalars: &[], expected_output: "stage5.registers_val_evaluation.output.claim_expr" },
    Stage5RelationOutputPlan { relation: Stage5RelationKind::Stage5InstructionReadRaf, local_scalars: STAGE5_RELATION_OUTPUT_2_LOCAL_SCALARS, expected_output: "stage5.instruction_read_raf.output.claim_expr" },
];

pub const STAGE5_PROGRAM: Stage5VerifierProgramPlan = Stage5VerifierProgramPlan {
    base: Stage5CpuProgramPlan {
        role: "verifier",
        params: STAGE5_PARAMS,
        steps: STAGE5_PROGRAM_STEPS,
        transcript_squeezes: STAGE5_TRANSCRIPT_SQUEEZES,
        transcript_absorb_bytes: STAGE5_TRANSCRIPT_ABSORB_BYTES,
        opening_inputs: STAGE5_OPENING_INPUTS,
        field_constants: STAGE5_FIELD_CONSTANTS,
        field_exprs: STAGE5_FIELD_EXPRS,
        scalar_exprs: STAGE5_SCALAR_EXPRS,
        kernels: STAGE5_KERNELS,
        claims: STAGE5_SUMCHECK_CLAIMS,
        batches: STAGE5_SUMCHECK_BATCHES,
        drivers: STAGE5_SUMCHECK_DRIVERS,
        instance_results: STAGE5_SUMCHECK_INSTANCE_RESULTS,
        evals: STAGE5_SUMCHECK_EVALS,
        indexed_eval_families: STAGE5_INDEXED_EVAL_FAMILIES,
        relation_outputs: STAGE5_RELATION_OUTPUTS,
        point_exprs: STAGE5_POINT_EXPRS,
        opening_claims: STAGE5_OPENING_CLAIMS,
        opening_equalities: STAGE5_OPENING_EQUALITIES,
        opening_batches: STAGE5_OPENING_BATCHES,
    },
    instruction_read_raf_plan: STAGE5_INSTRUCTION_READ_RAF_PLAN,
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
        bolt_verifier_runtime::ValueStore::with_opening_inputs(opening_inputs, program.opening_inputs)?;
    store.seed_constants(program.field_constants);
    let mut artifacts = Stage5ExecutionArtifacts::default();
    for step in program.steps {
        match step.kind {
            Stage5ProgramStepKind::TranscriptSqueeze => {
                let squeeze =
                    find_plan(program.transcript_squeezes, step.symbol).ok_or(VerifyStage5Error::MissingValue {
                        symbol: step.symbol,
                    })?;
                verify_stage5_squeeze(program, squeeze, &mut store, transcript, &mut artifacts)?;
            }
            Stage5ProgramStepKind::TranscriptAbsorbBytes => {
                let absorb = find_plan(program.transcript_absorb_bytes, step.symbol).ok_or(
                    VerifyStage5Error::MissingValue {
                        symbol: step.symbol,
                    },
                )?;
                absorb_stage5_bytes(absorb, transcript);
            }
            Stage5ProgramStepKind::SumcheckDriver => {
                let driver =
                    find_plan(program.drivers, step.symbol).ok_or(VerifyStage5Error::MissingProof {
                        driver: step.symbol,
                    })?;
                verify_stage5_driver(program, driver, proof, &mut store, transcript, &mut artifacts)?;
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
    store: &mut bolt_verifier_runtime::ValueStore<Fr>,
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
        .evaluate_available_exprs(program.field_exprs, program.scalar_exprs)
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
    store: &mut bolt_verifier_runtime::ValueStore<Fr>,
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
    let Some(relation) = driver.relation else {
        return Err(VerifyStage5Error::InvalidProof {
            driver: driver.symbol,
            reason: "missing driver relation",
        });
    };
    let output = match relation {
        Stage5RelationKind::Stage5Batched => {
            verify_batched_stage5(program, driver, proof, store, transcript)?
        }
        relation => return Err(VerifyStage5Error::UnsupportedRelation { relation }),
    };
    artifacts.sumchecks.push(output);
    Ok(())
}

fn verify_batched_stage5<T>(
    program: &'static Stage5VerifierProgramPlan,
    driver: &'static Stage5SumcheckDriverPlan,
    proof: &Stage5SumcheckOutput<Fr>,
    store: &mut bolt_verifier_runtime::ValueStore<Fr>,
    transcript: &mut T,
) -> Result<Stage5SumcheckOutput<Fr>, VerifyStage5Error>
where
    T: Transcript<Challenge = Fr>,
{
    store.evaluate_available_points(
        program.point_exprs,
        |input, expected, actual| VerifyStage5Error::InvalidInputLength {
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
        program.scalar_exprs,
        program.opening_inputs,
        program.opening_claims,
        program.opening_batches,
        store,
        transcript,
        |store, evals, point, batching_coeffs| {
            bolt_verifier_runtime::evaluate_relation_output_batch(
                driver,
                program.batches,
                program.claims,
                program.instance_results,
                program.relation_outputs,
                program.field_exprs,
                program.scalar_exprs,
                program.indexed_eval_families,
                store,
                evals,
                point,
                batching_coeffs,
                |instance, relation_output, local_point| {
                    stage5_relation_output_inputs(program, instance, relation_output, local_point)
                },
            )
        },
        |store, verified| observe_stage5_sumcheck_output(program, store, verified),
        |driver, error| VerifyStage5Error::Sumcheck { driver, error },
    )
}

fn observe_stage5_sumcheck_output<F: Field>(
    program: &'static Stage5VerifierProgramPlan,
    store: &mut bolt_verifier_runtime::ValueStore<F>,
    output: &Stage5SumcheckOutput<F>,
) -> Result<(), VerifyStage5Error> {
    store.observe_sumcheck_output(
        program.instance_results,
        program.evals,
        output,
        |instance, mut point| {
            match instance.point_order {
                bolt_verifier_runtime::SumcheckPointOrder::AsIs => {}
                bolt_verifier_runtime::SumcheckPointOrder::Reverse => point.reverse(),
                bolt_verifier_runtime::SumcheckPointOrder::InstructionReadRaf => {
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
    store.evaluate_named_eval_families(program.indexed_eval_families)?;
    store.evaluate_available_points(
        program.point_exprs,
        |input, expected, actual| VerifyStage5Error::InvalidInputLength {
            input,
            expected,
            actual,
        },
    )?;
    store
        .evaluate_available_exprs(program.field_exprs, program.scalar_exprs)
        .map_err(VerifyStage5Error::from)?;
    store.verify_opening_equalities(
        program.opening_equalities,
        |driver, reason| VerifyStage5Error::InvalidProof { driver, reason },
        |symbol| VerifyStage5Error::MissingValue { symbol },
    )
}

fn stage5_relation_output_inputs<'a>(
    program: &'static Stage5VerifierProgramPlan,
    instance: &Stage5SumcheckInstanceResultPlan,
    relation_output: &Stage5RelationOutputPlan,
    local_point: &'a [Fr],
) -> Result<bolt_verifier_runtime::RelationOutputInputs<'a, Fr>, VerifyStage5Error> {
    if instance.relation != Stage5RelationKind::Stage5InstructionReadRaf {
        return Ok(bolt_verifier_runtime::RelationOutputInputs::empty());
    }

    Ok(bolt_verifier_runtime::RelationOutputInputs {
        scalars: bolt_verifier_runtime::select_named_scalars(
            relation_output.local_scalars,
            evaluate_stage5_instruction_read_raf_local_scalars(
                &program.instruction_read_raf_plan,
                local_point,
            )?,
        )?,
        points: Vec::new(),
    })
}
