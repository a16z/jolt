#![allow(dead_code)]

use super::common::append_labeled_scalar;
use jolt_field::Fr;
use jolt_sumcheck::{CompressedLabeledRoundPoly, LabeledRoundPoly, SumcheckClaim, SumcheckError, SumcheckVerifier};
use jolt_transcript::{Blake2bTranscript, Transcript};

pub type DefaultStage1Transcript = Blake2bTranscript<Fr>;

pub type Stage1Params = super::common::StageParams;
pub type Stage1NamedEval<F> = super::common::StageNamedEval<F>;
pub type Stage1SumcheckOutput<F> = super::common::StageSumcheckOutput<F>;
pub type Stage1ChallengeVector<F> = super::common::StageChallengeVector<F>;
pub type Stage1ExecutionArtifacts<F> = super::common::StageExecutionArtifacts<F>;
pub type Stage1Proof<F> = super::common::StageProof<F>;
pub type Stage1VerifierProgramPlan = super::common::VerifierProgramPlanMinimal;

pub use super::common::{
    OpeningBatchPlan as Stage1OpeningBatchPlan, OpeningClaimPlan as Stage1OpeningClaimPlan,
    SumcheckBatchPlan as Stage1SumcheckBatchPlan, SumcheckEvalPlan as Stage1SumcheckEvalPlan,
    SumcheckInstanceResultPlan as Stage1SumcheckInstanceResultPlan,
    TranscriptSqueezePlan as Stage1TranscriptSqueezePlan,
    SumcheckClaimPlan as Stage1SumcheckClaimPlan,
    SumcheckDriverPlan as Stage1SumcheckDriverPlan,
};

#[derive(Debug)]
pub enum VerifyStage1Error {
    UnexpectedProofCount { expected: usize, got: usize },
    MissingProof { driver: &'static str },
    MissingBatch { driver: &'static str, batch: &'static str },
    MissingClaim { driver: &'static str, claim: &'static str },
    MissingDependency { driver: &'static str, dependency: &'static str },
    InvalidProof { driver: &'static str, reason: &'static str },
    UnsupportedRelation { relation: &'static str },
    Sumcheck { driver: &'static str, error: SumcheckError<Fr> },
}

pub const STAGE1_PARAMS: Stage1Params = Stage1Params {
    field: "bn254_fr",
    pcs: "dory",
    transcript: "blake2b_transcript",
};
pub const STAGE1_TRANSCRIPT_SQUEEZES: &[Stage1TranscriptSqueezePlan] = &[
    Stage1TranscriptSqueezePlan { symbol: "stage1.tau", label: "outer_tau", kind: "challenge_vector", count: 18 },
];

pub const STAGE1_SUMCHECK_CLAIMS: &[Stage1SumcheckClaimPlan] = &[
    Stage1SumcheckClaimPlan { symbol: "stage1.uniskip.input", stage: "stage1", domain: "jolt.stage1_uniskip_domain", num_rounds: 1, degree: 27, claim: "stage1.zero", kernel: None, relation: Some("jolt.stage1.outer.uniskip"), claim_value: "stage1.zero", input_openings: "" },
    Stage1SumcheckClaimPlan { symbol: "stage1.outer_remaining.input", stage: "stage1", domain: "jolt.trace_domain", num_rounds: 17, degree: 3, claim: "stage1.uniskip.eval", kernel: None, relation: Some("jolt.stage1.outer.remaining"), claim_value: "stage1.uniskip.eval", input_openings: "stage1.uniskip.opening" },
];
pub const STAGE1_SUMCHECK_BATCH_0_ROUND_SCHEDULE: &[usize] = &[
    1,
];

pub const STAGE1_SUMCHECK_BATCH_1_ROUND_SCHEDULE: &[usize] = &[
    17,
];

pub const STAGE1_SUMCHECK_BATCHES: &[Stage1SumcheckBatchPlan] = &[
    Stage1SumcheckBatchPlan { symbol: "stage1.uniskip.batch", stage: "stage1", proof_slot: "stage1.uni_skip_first_round", policy: "single_instance", count: 1, ordered_claims: "stage1.uniskip.input", claim_operands: "stage1.uniskip.input", claim_label: "uniskip_claim", round_label: "uniskip_poly", round_schedule: STAGE1_SUMCHECK_BATCH_0_ROUND_SCHEDULE },
    Stage1SumcheckBatchPlan { symbol: "stage1.outer_remaining.batch", stage: "stage1", proof_slot: "stage1.sumcheck", policy: "jolt_core_front_loaded", count: 1, ordered_claims: "stage1.outer_remaining.input", claim_operands: "stage1.outer_remaining.input", claim_label: "sumcheck_claim", round_label: "sumcheck_poly", round_schedule: STAGE1_SUMCHECK_BATCH_1_ROUND_SCHEDULE },
];
pub const STAGE1_SUMCHECK_DRIVER_0_ROUND_SCHEDULE: &[usize] = &[
    1,
];

pub const STAGE1_SUMCHECK_DRIVER_1_ROUND_SCHEDULE: &[usize] = &[
    17,
];

pub const STAGE1_SUMCHECK_DRIVERS: &[Stage1SumcheckDriverPlan] = &[
    Stage1SumcheckDriverPlan { symbol: "stage1.uniskip.sumcheck", stage: "stage1", proof_slot: "stage1.uni_skip_first_round", kernel: None, relation: Some("jolt.stage1.outer.uniskip"), batch: "stage1.uniskip.batch", policy: "univariate_skip", round_schedule: STAGE1_SUMCHECK_DRIVER_0_ROUND_SCHEDULE, claim_label: "uniskip_claim", round_label: "uniskip_poly", num_rounds: 1, degree: 27 },
    Stage1SumcheckDriverPlan { symbol: "stage1.outer_remaining.sumcheck", stage: "stage1", proof_slot: "stage1.sumcheck", kernel: None, relation: Some("jolt.stage1.outer.remaining"), batch: "stage1.outer_remaining.batch", policy: "jolt_core_front_loaded", round_schedule: STAGE1_SUMCHECK_DRIVER_1_ROUND_SCHEDULE, claim_label: "sumcheck_claim", round_label: "sumcheck_poly", num_rounds: 17, degree: 3 },
];
pub const STAGE1_SUMCHECK_INSTANCE_RESULTS: &[Stage1SumcheckInstanceResultPlan] = &[
    Stage1SumcheckInstanceResultPlan { symbol: "stage1.uniskip.instance", source: "stage1.uniskip.sumcheck", claim: "stage1.uniskip.input", relation: "jolt.stage1.outer.uniskip", index: 0, point_arity: 1, num_rounds: 1, round_offset: 0, point_order: "as_is", degree: 27 },
    Stage1SumcheckInstanceResultPlan { symbol: "stage1.outer_remaining.instance", source: "stage1.outer_remaining.sumcheck", claim: "stage1.outer_remaining.input", relation: "jolt.stage1.outer.remaining", index: 0, point_arity: 16, num_rounds: 17, round_offset: 1, point_order: "reverse", degree: 3 },
];

pub const STAGE1_SUMCHECK_EVALS: &[Stage1SumcheckEvalPlan] = &[
    Stage1SumcheckEvalPlan { symbol: "stage1.uniskip.eval", source: "stage1.uniskip.sumcheck", name: "stage1.uniskip.eval", index: 0, oracle: "UnivariateSkip" },
    Stage1SumcheckEvalPlan { symbol: "stage1.outer_remaining.eval.LeftInstructionInput", source: "stage1.outer_remaining.sumcheck", name: "stage1.outer_remaining.eval.LeftInstructionInput", index: 0, oracle: "LeftInstructionInput" },
    Stage1SumcheckEvalPlan { symbol: "stage1.outer_remaining.eval.RightInstructionInput", source: "stage1.outer_remaining.sumcheck", name: "stage1.outer_remaining.eval.RightInstructionInput", index: 1, oracle: "RightInstructionInput" },
    Stage1SumcheckEvalPlan { symbol: "stage1.outer_remaining.eval.Product", source: "stage1.outer_remaining.sumcheck", name: "stage1.outer_remaining.eval.Product", index: 2, oracle: "Product" },
    Stage1SumcheckEvalPlan { symbol: "stage1.outer_remaining.eval.ShouldBranch", source: "stage1.outer_remaining.sumcheck", name: "stage1.outer_remaining.eval.ShouldBranch", index: 3, oracle: "ShouldBranch" },
    Stage1SumcheckEvalPlan { symbol: "stage1.outer_remaining.eval.PC", source: "stage1.outer_remaining.sumcheck", name: "stage1.outer_remaining.eval.PC", index: 4, oracle: "PC" },
    Stage1SumcheckEvalPlan { symbol: "stage1.outer_remaining.eval.UnexpandedPC", source: "stage1.outer_remaining.sumcheck", name: "stage1.outer_remaining.eval.UnexpandedPC", index: 5, oracle: "UnexpandedPC" },
    Stage1SumcheckEvalPlan { symbol: "stage1.outer_remaining.eval.Imm", source: "stage1.outer_remaining.sumcheck", name: "stage1.outer_remaining.eval.Imm", index: 6, oracle: "Imm" },
    Stage1SumcheckEvalPlan { symbol: "stage1.outer_remaining.eval.RamAddress", source: "stage1.outer_remaining.sumcheck", name: "stage1.outer_remaining.eval.RamAddress", index: 7, oracle: "RamAddress" },
    Stage1SumcheckEvalPlan { symbol: "stage1.outer_remaining.eval.Rs1Value", source: "stage1.outer_remaining.sumcheck", name: "stage1.outer_remaining.eval.Rs1Value", index: 8, oracle: "Rs1Value" },
    Stage1SumcheckEvalPlan { symbol: "stage1.outer_remaining.eval.Rs2Value", source: "stage1.outer_remaining.sumcheck", name: "stage1.outer_remaining.eval.Rs2Value", index: 9, oracle: "Rs2Value" },
    Stage1SumcheckEvalPlan { symbol: "stage1.outer_remaining.eval.RdWriteValue", source: "stage1.outer_remaining.sumcheck", name: "stage1.outer_remaining.eval.RdWriteValue", index: 10, oracle: "RdWriteValue" },
    Stage1SumcheckEvalPlan { symbol: "stage1.outer_remaining.eval.RamReadValue", source: "stage1.outer_remaining.sumcheck", name: "stage1.outer_remaining.eval.RamReadValue", index: 11, oracle: "RamReadValue" },
    Stage1SumcheckEvalPlan { symbol: "stage1.outer_remaining.eval.RamWriteValue", source: "stage1.outer_remaining.sumcheck", name: "stage1.outer_remaining.eval.RamWriteValue", index: 12, oracle: "RamWriteValue" },
    Stage1SumcheckEvalPlan { symbol: "stage1.outer_remaining.eval.LeftLookupOperand", source: "stage1.outer_remaining.sumcheck", name: "stage1.outer_remaining.eval.LeftLookupOperand", index: 13, oracle: "LeftLookupOperand" },
    Stage1SumcheckEvalPlan { symbol: "stage1.outer_remaining.eval.RightLookupOperand", source: "stage1.outer_remaining.sumcheck", name: "stage1.outer_remaining.eval.RightLookupOperand", index: 14, oracle: "RightLookupOperand" },
    Stage1SumcheckEvalPlan { symbol: "stage1.outer_remaining.eval.NextUnexpandedPC", source: "stage1.outer_remaining.sumcheck", name: "stage1.outer_remaining.eval.NextUnexpandedPC", index: 15, oracle: "NextUnexpandedPC" },
    Stage1SumcheckEvalPlan { symbol: "stage1.outer_remaining.eval.NextPC", source: "stage1.outer_remaining.sumcheck", name: "stage1.outer_remaining.eval.NextPC", index: 16, oracle: "NextPC" },
    Stage1SumcheckEvalPlan { symbol: "stage1.outer_remaining.eval.NextIsVirtual", source: "stage1.outer_remaining.sumcheck", name: "stage1.outer_remaining.eval.NextIsVirtual", index: 17, oracle: "NextIsVirtual" },
    Stage1SumcheckEvalPlan { symbol: "stage1.outer_remaining.eval.NextIsFirstInSequence", source: "stage1.outer_remaining.sumcheck", name: "stage1.outer_remaining.eval.NextIsFirstInSequence", index: 18, oracle: "NextIsFirstInSequence" },
    Stage1SumcheckEvalPlan { symbol: "stage1.outer_remaining.eval.LookupOutput", source: "stage1.outer_remaining.sumcheck", name: "stage1.outer_remaining.eval.LookupOutput", index: 19, oracle: "LookupOutput" },
    Stage1SumcheckEvalPlan { symbol: "stage1.outer_remaining.eval.ShouldJump", source: "stage1.outer_remaining.sumcheck", name: "stage1.outer_remaining.eval.ShouldJump", index: 20, oracle: "ShouldJump" },
    Stage1SumcheckEvalPlan { symbol: "stage1.outer_remaining.eval.OpFlagAddOperands", source: "stage1.outer_remaining.sumcheck", name: "stage1.outer_remaining.eval.OpFlagAddOperands", index: 21, oracle: "OpFlagAddOperands" },
    Stage1SumcheckEvalPlan { symbol: "stage1.outer_remaining.eval.OpFlagSubtractOperands", source: "stage1.outer_remaining.sumcheck", name: "stage1.outer_remaining.eval.OpFlagSubtractOperands", index: 22, oracle: "OpFlagSubtractOperands" },
    Stage1SumcheckEvalPlan { symbol: "stage1.outer_remaining.eval.OpFlagMultiplyOperands", source: "stage1.outer_remaining.sumcheck", name: "stage1.outer_remaining.eval.OpFlagMultiplyOperands", index: 23, oracle: "OpFlagMultiplyOperands" },
    Stage1SumcheckEvalPlan { symbol: "stage1.outer_remaining.eval.OpFlagLoad", source: "stage1.outer_remaining.sumcheck", name: "stage1.outer_remaining.eval.OpFlagLoad", index: 24, oracle: "OpFlagLoad" },
    Stage1SumcheckEvalPlan { symbol: "stage1.outer_remaining.eval.OpFlagStore", source: "stage1.outer_remaining.sumcheck", name: "stage1.outer_remaining.eval.OpFlagStore", index: 25, oracle: "OpFlagStore" },
    Stage1SumcheckEvalPlan { symbol: "stage1.outer_remaining.eval.OpFlagJump", source: "stage1.outer_remaining.sumcheck", name: "stage1.outer_remaining.eval.OpFlagJump", index: 26, oracle: "OpFlagJump" },
    Stage1SumcheckEvalPlan { symbol: "stage1.outer_remaining.eval.OpFlagWriteLookupOutputToRD", source: "stage1.outer_remaining.sumcheck", name: "stage1.outer_remaining.eval.OpFlagWriteLookupOutputToRD", index: 27, oracle: "OpFlagWriteLookupOutputToRD" },
    Stage1SumcheckEvalPlan { symbol: "stage1.outer_remaining.eval.OpFlagVirtualInstruction", source: "stage1.outer_remaining.sumcheck", name: "stage1.outer_remaining.eval.OpFlagVirtualInstruction", index: 28, oracle: "OpFlagVirtualInstruction" },
    Stage1SumcheckEvalPlan { symbol: "stage1.outer_remaining.eval.OpFlagAssert", source: "stage1.outer_remaining.sumcheck", name: "stage1.outer_remaining.eval.OpFlagAssert", index: 29, oracle: "OpFlagAssert" },
    Stage1SumcheckEvalPlan { symbol: "stage1.outer_remaining.eval.OpFlagDoNotUpdateUnexpandedPC", source: "stage1.outer_remaining.sumcheck", name: "stage1.outer_remaining.eval.OpFlagDoNotUpdateUnexpandedPC", index: 30, oracle: "OpFlagDoNotUpdateUnexpandedPC" },
    Stage1SumcheckEvalPlan { symbol: "stage1.outer_remaining.eval.OpFlagAdvice", source: "stage1.outer_remaining.sumcheck", name: "stage1.outer_remaining.eval.OpFlagAdvice", index: 31, oracle: "OpFlagAdvice" },
    Stage1SumcheckEvalPlan { symbol: "stage1.outer_remaining.eval.OpFlagIsCompressed", source: "stage1.outer_remaining.sumcheck", name: "stage1.outer_remaining.eval.OpFlagIsCompressed", index: 32, oracle: "OpFlagIsCompressed" },
    Stage1SumcheckEvalPlan { symbol: "stage1.outer_remaining.eval.OpFlagIsFirstInSequence", source: "stage1.outer_remaining.sumcheck", name: "stage1.outer_remaining.eval.OpFlagIsFirstInSequence", index: 33, oracle: "OpFlagIsFirstInSequence" },
    Stage1SumcheckEvalPlan { symbol: "stage1.outer_remaining.eval.OpFlagIsLastInSequence", source: "stage1.outer_remaining.sumcheck", name: "stage1.outer_remaining.eval.OpFlagIsLastInSequence", index: 34, oracle: "OpFlagIsLastInSequence" },
];

pub const STAGE1_OPENING_CLAIMS: &[Stage1OpeningClaimPlan] = &[
    Stage1OpeningClaimPlan { symbol: "stage1.uniskip.opening", oracle: "UnivariateSkip", domain: "jolt.stage1_uniskip_domain", point_arity: 1, claim_kind: "virtual", point_source: "stage1.uniskip.instance", eval_source: "stage1.uniskip.eval" },
    Stage1OpeningClaimPlan { symbol: "stage1.outer_remaining.opening.LeftInstructionInput", oracle: "LeftInstructionInput", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage1.outer_remaining.instance", eval_source: "stage1.outer_remaining.eval.LeftInstructionInput" },
    Stage1OpeningClaimPlan { symbol: "stage1.outer_remaining.opening.RightInstructionInput", oracle: "RightInstructionInput", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage1.outer_remaining.instance", eval_source: "stage1.outer_remaining.eval.RightInstructionInput" },
    Stage1OpeningClaimPlan { symbol: "stage1.outer_remaining.opening.Product", oracle: "Product", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage1.outer_remaining.instance", eval_source: "stage1.outer_remaining.eval.Product" },
    Stage1OpeningClaimPlan { symbol: "stage1.outer_remaining.opening.ShouldBranch", oracle: "ShouldBranch", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage1.outer_remaining.instance", eval_source: "stage1.outer_remaining.eval.ShouldBranch" },
    Stage1OpeningClaimPlan { symbol: "stage1.outer_remaining.opening.PC", oracle: "PC", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage1.outer_remaining.instance", eval_source: "stage1.outer_remaining.eval.PC" },
    Stage1OpeningClaimPlan { symbol: "stage1.outer_remaining.opening.UnexpandedPC", oracle: "UnexpandedPC", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage1.outer_remaining.instance", eval_source: "stage1.outer_remaining.eval.UnexpandedPC" },
    Stage1OpeningClaimPlan { symbol: "stage1.outer_remaining.opening.Imm", oracle: "Imm", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage1.outer_remaining.instance", eval_source: "stage1.outer_remaining.eval.Imm" },
    Stage1OpeningClaimPlan { symbol: "stage1.outer_remaining.opening.RamAddress", oracle: "RamAddress", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage1.outer_remaining.instance", eval_source: "stage1.outer_remaining.eval.RamAddress" },
    Stage1OpeningClaimPlan { symbol: "stage1.outer_remaining.opening.Rs1Value", oracle: "Rs1Value", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage1.outer_remaining.instance", eval_source: "stage1.outer_remaining.eval.Rs1Value" },
    Stage1OpeningClaimPlan { symbol: "stage1.outer_remaining.opening.Rs2Value", oracle: "Rs2Value", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage1.outer_remaining.instance", eval_source: "stage1.outer_remaining.eval.Rs2Value" },
    Stage1OpeningClaimPlan { symbol: "stage1.outer_remaining.opening.RdWriteValue", oracle: "RdWriteValue", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage1.outer_remaining.instance", eval_source: "stage1.outer_remaining.eval.RdWriteValue" },
    Stage1OpeningClaimPlan { symbol: "stage1.outer_remaining.opening.RamReadValue", oracle: "RamReadValue", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage1.outer_remaining.instance", eval_source: "stage1.outer_remaining.eval.RamReadValue" },
    Stage1OpeningClaimPlan { symbol: "stage1.outer_remaining.opening.RamWriteValue", oracle: "RamWriteValue", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage1.outer_remaining.instance", eval_source: "stage1.outer_remaining.eval.RamWriteValue" },
    Stage1OpeningClaimPlan { symbol: "stage1.outer_remaining.opening.LeftLookupOperand", oracle: "LeftLookupOperand", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage1.outer_remaining.instance", eval_source: "stage1.outer_remaining.eval.LeftLookupOperand" },
    Stage1OpeningClaimPlan { symbol: "stage1.outer_remaining.opening.RightLookupOperand", oracle: "RightLookupOperand", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage1.outer_remaining.instance", eval_source: "stage1.outer_remaining.eval.RightLookupOperand" },
    Stage1OpeningClaimPlan { symbol: "stage1.outer_remaining.opening.NextUnexpandedPC", oracle: "NextUnexpandedPC", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage1.outer_remaining.instance", eval_source: "stage1.outer_remaining.eval.NextUnexpandedPC" },
    Stage1OpeningClaimPlan { symbol: "stage1.outer_remaining.opening.NextPC", oracle: "NextPC", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage1.outer_remaining.instance", eval_source: "stage1.outer_remaining.eval.NextPC" },
    Stage1OpeningClaimPlan { symbol: "stage1.outer_remaining.opening.NextIsVirtual", oracle: "NextIsVirtual", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage1.outer_remaining.instance", eval_source: "stage1.outer_remaining.eval.NextIsVirtual" },
    Stage1OpeningClaimPlan { symbol: "stage1.outer_remaining.opening.NextIsFirstInSequence", oracle: "NextIsFirstInSequence", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage1.outer_remaining.instance", eval_source: "stage1.outer_remaining.eval.NextIsFirstInSequence" },
    Stage1OpeningClaimPlan { symbol: "stage1.outer_remaining.opening.LookupOutput", oracle: "LookupOutput", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage1.outer_remaining.instance", eval_source: "stage1.outer_remaining.eval.LookupOutput" },
    Stage1OpeningClaimPlan { symbol: "stage1.outer_remaining.opening.ShouldJump", oracle: "ShouldJump", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage1.outer_remaining.instance", eval_source: "stage1.outer_remaining.eval.ShouldJump" },
    Stage1OpeningClaimPlan { symbol: "stage1.outer_remaining.opening.OpFlagAddOperands", oracle: "OpFlagAddOperands", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage1.outer_remaining.instance", eval_source: "stage1.outer_remaining.eval.OpFlagAddOperands" },
    Stage1OpeningClaimPlan { symbol: "stage1.outer_remaining.opening.OpFlagSubtractOperands", oracle: "OpFlagSubtractOperands", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage1.outer_remaining.instance", eval_source: "stage1.outer_remaining.eval.OpFlagSubtractOperands" },
    Stage1OpeningClaimPlan { symbol: "stage1.outer_remaining.opening.OpFlagMultiplyOperands", oracle: "OpFlagMultiplyOperands", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage1.outer_remaining.instance", eval_source: "stage1.outer_remaining.eval.OpFlagMultiplyOperands" },
    Stage1OpeningClaimPlan { symbol: "stage1.outer_remaining.opening.OpFlagLoad", oracle: "OpFlagLoad", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage1.outer_remaining.instance", eval_source: "stage1.outer_remaining.eval.OpFlagLoad" },
    Stage1OpeningClaimPlan { symbol: "stage1.outer_remaining.opening.OpFlagStore", oracle: "OpFlagStore", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage1.outer_remaining.instance", eval_source: "stage1.outer_remaining.eval.OpFlagStore" },
    Stage1OpeningClaimPlan { symbol: "stage1.outer_remaining.opening.OpFlagJump", oracle: "OpFlagJump", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage1.outer_remaining.instance", eval_source: "stage1.outer_remaining.eval.OpFlagJump" },
    Stage1OpeningClaimPlan { symbol: "stage1.outer_remaining.opening.OpFlagWriteLookupOutputToRD", oracle: "OpFlagWriteLookupOutputToRD", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage1.outer_remaining.instance", eval_source: "stage1.outer_remaining.eval.OpFlagWriteLookupOutputToRD" },
    Stage1OpeningClaimPlan { symbol: "stage1.outer_remaining.opening.OpFlagVirtualInstruction", oracle: "OpFlagVirtualInstruction", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage1.outer_remaining.instance", eval_source: "stage1.outer_remaining.eval.OpFlagVirtualInstruction" },
    Stage1OpeningClaimPlan { symbol: "stage1.outer_remaining.opening.OpFlagAssert", oracle: "OpFlagAssert", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage1.outer_remaining.instance", eval_source: "stage1.outer_remaining.eval.OpFlagAssert" },
    Stage1OpeningClaimPlan { symbol: "stage1.outer_remaining.opening.OpFlagDoNotUpdateUnexpandedPC", oracle: "OpFlagDoNotUpdateUnexpandedPC", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage1.outer_remaining.instance", eval_source: "stage1.outer_remaining.eval.OpFlagDoNotUpdateUnexpandedPC" },
    Stage1OpeningClaimPlan { symbol: "stage1.outer_remaining.opening.OpFlagAdvice", oracle: "OpFlagAdvice", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage1.outer_remaining.instance", eval_source: "stage1.outer_remaining.eval.OpFlagAdvice" },
    Stage1OpeningClaimPlan { symbol: "stage1.outer_remaining.opening.OpFlagIsCompressed", oracle: "OpFlagIsCompressed", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage1.outer_remaining.instance", eval_source: "stage1.outer_remaining.eval.OpFlagIsCompressed" },
    Stage1OpeningClaimPlan { symbol: "stage1.outer_remaining.opening.OpFlagIsFirstInSequence", oracle: "OpFlagIsFirstInSequence", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage1.outer_remaining.instance", eval_source: "stage1.outer_remaining.eval.OpFlagIsFirstInSequence" },
    Stage1OpeningClaimPlan { symbol: "stage1.outer_remaining.opening.OpFlagIsLastInSequence", oracle: "OpFlagIsLastInSequence", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage1.outer_remaining.instance", eval_source: "stage1.outer_remaining.eval.OpFlagIsLastInSequence" },
];

pub const STAGE1_OPENING_BATCHES: &[Stage1OpeningBatchPlan] = &[
    Stage1OpeningBatchPlan { symbol: "stage1.outer_remaining.openings", stage: "stage1", proof_slot: "stage1.virtual_openings", policy: "jolt_r1cs_input_order", count: 35, ordered_claims: "stage1.outer_remaining.opening.LeftInstructionInput|stage1.outer_remaining.opening.RightInstructionInput|stage1.outer_remaining.opening.Product|stage1.outer_remaining.opening.ShouldBranch|stage1.outer_remaining.opening.PC|stage1.outer_remaining.opening.UnexpandedPC|stage1.outer_remaining.opening.Imm|stage1.outer_remaining.opening.RamAddress|stage1.outer_remaining.opening.Rs1Value|stage1.outer_remaining.opening.Rs2Value|stage1.outer_remaining.opening.RdWriteValue|stage1.outer_remaining.opening.RamReadValue|stage1.outer_remaining.opening.RamWriteValue|stage1.outer_remaining.opening.LeftLookupOperand|stage1.outer_remaining.opening.RightLookupOperand|stage1.outer_remaining.opening.NextUnexpandedPC|stage1.outer_remaining.opening.NextPC|stage1.outer_remaining.opening.NextIsVirtual|stage1.outer_remaining.opening.NextIsFirstInSequence|stage1.outer_remaining.opening.LookupOutput|stage1.outer_remaining.opening.ShouldJump|stage1.outer_remaining.opening.OpFlagAddOperands|stage1.outer_remaining.opening.OpFlagSubtractOperands|stage1.outer_remaining.opening.OpFlagMultiplyOperands|stage1.outer_remaining.opening.OpFlagLoad|stage1.outer_remaining.opening.OpFlagStore|stage1.outer_remaining.opening.OpFlagJump|stage1.outer_remaining.opening.OpFlagWriteLookupOutputToRD|stage1.outer_remaining.opening.OpFlagVirtualInstruction|stage1.outer_remaining.opening.OpFlagAssert|stage1.outer_remaining.opening.OpFlagDoNotUpdateUnexpandedPC|stage1.outer_remaining.opening.OpFlagAdvice|stage1.outer_remaining.opening.OpFlagIsCompressed|stage1.outer_remaining.opening.OpFlagIsFirstInSequence|stage1.outer_remaining.opening.OpFlagIsLastInSequence", claim_operands: "stage1.outer_remaining.opening.LeftInstructionInput|stage1.outer_remaining.opening.RightInstructionInput|stage1.outer_remaining.opening.Product|stage1.outer_remaining.opening.ShouldBranch|stage1.outer_remaining.opening.PC|stage1.outer_remaining.opening.UnexpandedPC|stage1.outer_remaining.opening.Imm|stage1.outer_remaining.opening.RamAddress|stage1.outer_remaining.opening.Rs1Value|stage1.outer_remaining.opening.Rs2Value|stage1.outer_remaining.opening.RdWriteValue|stage1.outer_remaining.opening.RamReadValue|stage1.outer_remaining.opening.RamWriteValue|stage1.outer_remaining.opening.LeftLookupOperand|stage1.outer_remaining.opening.RightLookupOperand|stage1.outer_remaining.opening.NextUnexpandedPC|stage1.outer_remaining.opening.NextPC|stage1.outer_remaining.opening.NextIsVirtual|stage1.outer_remaining.opening.NextIsFirstInSequence|stage1.outer_remaining.opening.LookupOutput|stage1.outer_remaining.opening.ShouldJump|stage1.outer_remaining.opening.OpFlagAddOperands|stage1.outer_remaining.opening.OpFlagSubtractOperands|stage1.outer_remaining.opening.OpFlagMultiplyOperands|stage1.outer_remaining.opening.OpFlagLoad|stage1.outer_remaining.opening.OpFlagStore|stage1.outer_remaining.opening.OpFlagJump|stage1.outer_remaining.opening.OpFlagWriteLookupOutputToRD|stage1.outer_remaining.opening.OpFlagVirtualInstruction|stage1.outer_remaining.opening.OpFlagAssert|stage1.outer_remaining.opening.OpFlagDoNotUpdateUnexpandedPC|stage1.outer_remaining.opening.OpFlagAdvice|stage1.outer_remaining.opening.OpFlagIsCompressed|stage1.outer_remaining.opening.OpFlagIsFirstInSequence|stage1.outer_remaining.opening.OpFlagIsLastInSequence" },
];
pub const STAGE1_PROGRAM: Stage1VerifierProgramPlan = Stage1VerifierProgramPlan {
    params: STAGE1_PARAMS,
    transcript_squeezes: STAGE1_TRANSCRIPT_SQUEEZES,
    claims: STAGE1_SUMCHECK_CLAIMS,
    batches: STAGE1_SUMCHECK_BATCHES,
    drivers: STAGE1_SUMCHECK_DRIVERS,
    instance_results: STAGE1_SUMCHECK_INSTANCE_RESULTS,
    evals: STAGE1_SUMCHECK_EVALS,
    opening_claims: STAGE1_OPENING_CLAIMS,
    opening_batches: STAGE1_OPENING_BATCHES,
};

pub fn verify_stage1_outer<T>(
    proof: &Stage1Proof<Fr>,
    transcript: &mut T,
) -> Result<Stage1ExecutionArtifacts<Fr>, VerifyStage1Error>
where
    T: Transcript<Challenge = Fr>,
{
    verify_stage1_outer_with_program(&STAGE1_PROGRAM, proof, transcript)
}

pub fn verify_stage1_outer_with_program<T>(
    program: &'static Stage1VerifierProgramPlan,
    proof: &Stage1Proof<Fr>,
    transcript: &mut T,
) -> Result<Stage1ExecutionArtifacts<Fr>, VerifyStage1Error>
where
    T: Transcript<Challenge = Fr>,
{
    if proof.sumchecks.len() != program.drivers.len() {
        return Err(VerifyStage1Error::UnexpectedProofCount {
            expected: program.drivers.len(),
            got: proof.sumchecks.len(),
        });
    }
    let mut artifacts = Stage1ExecutionArtifacts::default();
    for squeeze in program.transcript_squeezes {
        let values = transcript.challenge_vector_optimized(squeeze.count);
        artifacts.challenge_vectors.push(Stage1ChallengeVector {
            symbol: squeeze.symbol,
            values,
        });
    }
    for (index, driver) in program.drivers.iter().enumerate() {
        let proof = proof.sumchecks.get(index).ok_or(VerifyStage1Error::MissingProof {
            driver: driver.symbol,
        })?;
        let output = verify_stage1_driver(program, driver, proof, &artifacts.sumchecks, transcript)?;
        artifacts.sumchecks.push(output);
    }
    artifacts
        .opening_batches
        .extend(program.opening_batches.iter());
    Ok(artifacts)
}

pub fn stage1_outer_verifier_program() -> &'static Stage1VerifierProgramPlan {
    &STAGE1_PROGRAM
}

fn verify_stage1_driver<T>(
    program: &'static Stage1VerifierProgramPlan,
    driver: &'static Stage1SumcheckDriverPlan,
    proof: &Stage1SumcheckOutput<Fr>,
    completed: &[Stage1SumcheckOutput<Fr>],
    transcript: &mut T,
) -> Result<Stage1SumcheckOutput<Fr>, VerifyStage1Error>
where
    T: Transcript<Challenge = Fr>,
{
    if proof.driver != driver.symbol {
        return Err(VerifyStage1Error::InvalidProof {
            driver: driver.symbol,
            reason: "driver symbol mismatch",
        });
    }
    let relation = driver.relation.unwrap_or("<missing>");
    match relation {
        "jolt.stage1.outer.uniskip" => verify_outer_uniskip(program, driver, proof, transcript),
        "jolt.stage1.outer.remaining" => {
            verify_outer_remaining(program, driver, proof, completed, transcript)
        }
        relation => Err(VerifyStage1Error::UnsupportedRelation { relation }),
    }
}

fn verify_outer_uniskip<T>(
    program: &'static Stage1VerifierProgramPlan,
    driver: &'static Stage1SumcheckDriverPlan,
    proof: &Stage1SumcheckOutput<Fr>,
    transcript: &mut T,
) -> Result<Stage1SumcheckOutput<Fr>, VerifyStage1Error>
where
    T: Transcript<Challenge = Fr>,
{
    let claim = SumcheckClaim::new(driver.num_rounds, driver.degree, Fr::from_u64(0));
    let round_proofs = proof
        .proof
        .round_polynomials
        .iter()
        .map(|poly| LabeledRoundPoly::new(poly, driver.round_label.as_bytes()))
        .collect::<Vec<_>>();
    let output = SumcheckVerifier::verify_optimized(&claim, &round_proofs, transcript)
        .map_err(|error| VerifyStage1Error::Sumcheck {
            driver: driver.symbol,
            error,
        })?;
    let eval = output.value;
    let point = output.point;
    if !proof.point.is_empty() && proof.point != point {
        return Err(VerifyStage1Error::InvalidProof {
            driver: driver.symbol,
            reason: "uniskip point mismatch",
        });
    }
    validate_eval_shape(program, driver, &proof.evals, Some(eval))?;
    append_labeled_scalar(transcript, "opening_claim", &eval);
    Ok(Stage1SumcheckOutput {
        driver: driver.symbol,
        point,
        evals: driver_evals(program, driver.symbol, eval),
        proof: proof.proof.clone(),
    })
}

fn verify_outer_remaining<T>(
    program: &'static Stage1VerifierProgramPlan,
    driver: &'static Stage1SumcheckDriverPlan,
    proof: &Stage1SumcheckOutput<Fr>,
    completed: &[Stage1SumcheckOutput<Fr>],
    transcript: &mut T,
) -> Result<Stage1SumcheckOutput<Fr>, VerifyStage1Error>
where
    T: Transcript<Challenge = Fr>,
{
    let input_claim = completed
        .iter()
        .find(|output| output.driver == "stage1.uniskip.sumcheck")
        .and_then(|output| output.evals.first())
        .map(|eval| eval.value)
        .ok_or(VerifyStage1Error::MissingDependency {
            driver: driver.symbol,
            dependency: "stage1.uniskip.eval",
        })?;
    append_labeled_scalar(transcript, driver.claim_label, &input_claim);
    let batching_coeff = transcript.challenge();
    let claim = SumcheckClaim::new(
        driver.num_rounds,
        driver.degree,
        input_claim * batching_coeff,
    );
    let round_proofs = proof
        .proof
        .round_polynomials
        .iter()
        .map(|poly| CompressedLabeledRoundPoly::new(poly, driver.round_label.as_bytes()))
        .collect::<Vec<_>>();
    let output = SumcheckVerifier::verify_optimized(&claim, &round_proofs, transcript)
        .map_err(|error| VerifyStage1Error::Sumcheck {
            driver: driver.symbol,
            error,
        })?;
    let point = output.point;
    if !proof.point.is_empty() && proof.point != point {
        return Err(VerifyStage1Error::InvalidProof {
            driver: driver.symbol,
            reason: "outer remaining point mismatch",
        });
    }
    validate_eval_shape(program, driver, &proof.evals, None)?;
    append_opening_claims(transcript, &proof.evals);
    Ok(Stage1SumcheckOutput {
        driver: driver.symbol,
        point,
        evals: proof.evals.clone(),
        proof: proof.proof.clone(),
    })
}

fn driver_evals(
    program: &'static Stage1VerifierProgramPlan,
    driver: &'static str,
    value: Fr,
) -> Vec<Stage1NamedEval<Fr>> {
    program
        .evals
        .iter()
        .filter(|eval| eval.source == driver)
        .map(|eval| Stage1NamedEval {
            name: eval.name,
            oracle: eval.oracle,
            value,
        })
        .collect()
}

fn validate_eval_shape(
    program: &'static Stage1VerifierProgramPlan,
    driver: &'static Stage1SumcheckDriverPlan,
    actual: &[Stage1NamedEval<Fr>],
    expected_value: Option<Fr>,
) -> Result<(), VerifyStage1Error> {
    let expected = program
        .evals
        .iter()
        .filter(|eval| eval.source == driver.symbol)
        .collect::<Vec<_>>();
    if actual.len() != expected.len() {
        return Err(VerifyStage1Error::InvalidProof {
            driver: driver.symbol,
            reason: "eval count mismatch",
        });
    }
    for (actual, expected) in actual.iter().zip(expected) {
        if actual.name != expected.name {
            return Err(VerifyStage1Error::InvalidProof {
                driver: driver.symbol,
                reason: "eval name mismatch",
            });
        }
        if actual.oracle != expected.oracle {
            return Err(VerifyStage1Error::InvalidProof {
                driver: driver.symbol,
                reason: "eval oracle mismatch",
            });
        }
        if expected_value.is_some_and(|value| actual.value != value) {
            return Err(VerifyStage1Error::InvalidProof {
                driver: driver.symbol,
                reason: "eval value mismatch",
            });
        }
    }
    Ok(())
}

fn append_opening_claims<T>(transcript: &mut T, evals: &[Stage1NamedEval<Fr>])
where
    T: Transcript<Challenge = Fr>,
{
    for eval in evals {
        append_labeled_scalar(transcript, "opening_claim", &eval.value);
    }
}
