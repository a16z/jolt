#![allow(dead_code)]

use jolt_field::{Field, Fr};
use jolt_lookup_tables::LookupTableKind;
use jolt_poly::EqPolynomial;
use jolt_sumcheck::{CompressedLabeledRoundPoly, SumcheckClaim, SumcheckError, SumcheckProof, SumcheckVerifier};
use jolt_transcript::{Blake2bTranscript, Label, LabelWithCount, Transcript};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage7Params {
    pub field: &'static str,
    pub pcs: &'static str,
    pub transcript: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage7KernelPlan {
    pub symbol: &'static str,
    pub relation: &'static str,
    pub kind: &'static str,
    pub backend: &'static str,
    pub abi: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage7TranscriptSqueezePlan {
    pub symbol: &'static str,
    pub label: &'static str,
    pub kind: &'static str,
    pub count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage7TranscriptAbsorbBytesPlan {
    pub symbol: &'static str,
    pub label: &'static str,
    pub payload: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage7ProgramStepPlan {
    pub kind: &'static str,
    pub symbol: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage7OpeningInputPlan {
    pub symbol: &'static str,
    pub source_stage: &'static str,
    pub source_claim: &'static str,
    pub oracle: &'static str,
    pub domain: &'static str,
    pub point_arity: usize,
    pub claim_kind: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage7FieldConstantPlan {
    pub symbol: &'static str,
    pub field: &'static str,
    pub value: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage7FieldExprPlan {
    pub symbol: &'static str,
    pub kind: &'static str,
    pub formula: &'static str,
    pub operand_names: &'static [&'static str],
    pub operands: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage7SumcheckClaimPlan {
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
pub struct Stage7SumcheckBatchPlan {
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
pub struct Stage7SumcheckDriverPlan {
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
pub struct Stage7SumcheckInstanceResultPlan {
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
pub struct Stage7SumcheckEvalPlan {
    pub symbol: &'static str,
    pub source: &'static str,
    pub name: &'static str,
    pub index: usize,
    pub oracle: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage7PointZeroPlan {
    pub symbol: &'static str,
    pub field: &'static str,
    pub arity: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage7PointSlicePlan {
    pub symbol: &'static str,
    pub source: &'static str,
    pub offset: usize,
    pub length: usize,
    pub input: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage7PointConcatPlan {
    pub symbol: &'static str,
    pub layout: &'static str,
    pub arity: usize,
    pub inputs: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage7OpeningClaimPlan {
    pub symbol: &'static str,
    pub oracle: &'static str,
    pub domain: &'static str,
    pub point_arity: usize,
    pub claim_kind: &'static str,
    pub point_source: &'static str,
    pub eval_source: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage7OpeningClaimEqualityPlan {
    pub symbol: &'static str,
    pub mode: &'static str,
    pub lhs: &'static str,
    pub rhs: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage7OpeningBatchPlan {
    pub symbol: &'static str,
    pub stage: &'static str,
    pub proof_slot: &'static str,
    pub policy: &'static str,
    pub count: usize,
    pub ordered_claims: &'static [&'static str],
    pub claim_operands: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage7CpuProgramPlan {
    pub role: &'static str,
    pub params: Stage7Params,
    pub steps: &'static [Stage7ProgramStepPlan],
    pub transcript_squeezes: &'static [Stage7TranscriptSqueezePlan],
    pub transcript_absorb_bytes: &'static [Stage7TranscriptAbsorbBytesPlan],
    pub opening_inputs: &'static [Stage7OpeningInputPlan],
    pub field_constants: &'static [Stage7FieldConstantPlan],
    pub field_exprs: &'static [Stage7FieldExprPlan],
    pub kernels: &'static [Stage7KernelPlan],
    pub claims: &'static [Stage7SumcheckClaimPlan],
    pub batches: &'static [Stage7SumcheckBatchPlan],
    pub drivers: &'static [Stage7SumcheckDriverPlan],
    pub instance_results: &'static [Stage7SumcheckInstanceResultPlan],
    pub evals: &'static [Stage7SumcheckEvalPlan],
    pub point_zeros: &'static [Stage7PointZeroPlan],
    pub point_slices: &'static [Stage7PointSlicePlan],
    pub point_concats: &'static [Stage7PointConcatPlan],
    pub opening_claims: &'static [Stage7OpeningClaimPlan],
    pub opening_equalities: &'static [Stage7OpeningClaimEqualityPlan],
    pub opening_batches: &'static [Stage7OpeningBatchPlan],
}

pub type DefaultStage7Transcript = Blake2bTranscript<Fr>;
pub type Stage7VerifierProgramPlan = Stage7CpuProgramPlan;

#[derive(Clone, Debug)]
pub struct Stage7NamedEval<F: Field> {
    pub name: &'static str,
    pub oracle: &'static str,
    pub value: F,
}

#[derive(Clone, Debug)]
pub struct Stage7SumcheckOutput<F: Field> {
    pub driver: &'static str,
    pub point: Vec<F>,
    pub evals: Vec<Stage7NamedEval<F>>,
    pub proof: SumcheckProof<F>,
}

#[derive(Clone, Debug)]
pub struct Stage7ChallengeVector<F: Field> {
    pub symbol: &'static str,
    pub values: Vec<F>,
}

#[derive(Clone, Debug)]
pub struct Stage7ExecutionArtifacts<F: Field> {
    pub challenge_vectors: Vec<Stage7ChallengeVector<F>>,
    pub sumchecks: Vec<Stage7SumcheckOutput<F>>,
    pub opening_batches: Vec<&'static Stage7OpeningBatchPlan>,
}

impl<F: Field> Default for Stage7ExecutionArtifacts<F> {
    fn default() -> Self {
        Self {
            challenge_vectors: Vec::new(),
            sumchecks: Vec::new(),
            opening_batches: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Stage7Proof<F: Field> {
    pub sumchecks: Vec<Stage7SumcheckOutput<F>>,
}

#[derive(Clone, Debug)]
pub struct Stage7OpeningInputValue<F: Field> {
    pub symbol: &'static str,
    pub point: Vec<F>,
    pub eval: F,
}

#[derive(Clone, Debug)]
pub struct Stage7BytecodeEntry {
    pub address: Fr,
    pub imm: Fr,
    pub circuit_flags: [bool; 14],
    pub rd: Option<usize>,
    pub rs1: Option<usize>,
    pub rs2: Option<usize>,
    pub lookup_table: Option<usize>,
    pub is_interleaved: bool,
    pub is_branch: bool,
    pub left_is_rs1: bool,
    pub left_is_pc: bool,
    pub right_is_rs2: bool,
    pub right_is_imm: bool,
    pub is_noop: bool,
}

#[derive(Clone, Debug)]
pub struct Stage7BytecodeReadRafData {
    pub entries: Vec<Stage7BytecodeEntry>,
    pub entry_bytecode_index: usize,
    pub num_lookup_tables: usize,
}

#[derive(Clone, Debug, Default)]
struct Stage7ValueStore<F: Field> {
    scalars: Vec<(&'static str, F)>,
    points: Vec<(&'static str, Vec<F>)>,
}

#[derive(Debug)]
pub enum VerifyStage7Error {
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

pub const STAGE7_PARAMS: Stage7Params = Stage7Params {
    field: "bn254_fr",
    pcs: "dory",
    transcript: "blake2b_transcript",
};
pub const STAGE7_PROGRAM_STEPS: &[Stage7ProgramStepPlan] = &[
    Stage7ProgramStepPlan { kind: "transcript_squeeze", symbol: "stage7.hamming_weight_claim_reduction.gamma" },
    Stage7ProgramStepPlan { kind: "sumcheck_driver", symbol: "stage7.sumcheck" },
];

pub const STAGE7_TRANSCRIPT_SQUEEZES: &[Stage7TranscriptSqueezePlan] = &[
    Stage7TranscriptSqueezePlan { symbol: "stage7.hamming_weight_claim_reduction.gamma", label: "hamming_weight_claim_reduction_gamma", kind: "challenge_scalar", count: 1 },
];

pub const STAGE7_TRANSCRIPT_ABSORB_BYTES: &[Stage7TranscriptAbsorbBytesPlan] = &[

];

pub const STAGE7_OPENING_INPUTS: &[Stage7OpeningInputPlan] = &[
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.hamming_booleanity.HammingWeight", source_stage: "stage6", source_claim: "stage6.hamming_booleanity.opening.HammingWeight", oracle: "HammingWeight", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_0", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_0", oracle: "InstructionRa_0", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_0", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_0", oracle: "InstructionRa_0", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_1", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_1", oracle: "InstructionRa_1", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_1", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_1", oracle: "InstructionRa_1", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_2", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_2", oracle: "InstructionRa_2", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_2", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_2", oracle: "InstructionRa_2", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_3", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_3", oracle: "InstructionRa_3", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_3", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_3", oracle: "InstructionRa_3", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_4", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_4", oracle: "InstructionRa_4", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_4", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_4", oracle: "InstructionRa_4", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_5", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_5", oracle: "InstructionRa_5", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_5", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_5", oracle: "InstructionRa_5", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_6", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_6", oracle: "InstructionRa_6", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_6", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_6", oracle: "InstructionRa_6", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_7", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_7", oracle: "InstructionRa_7", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_7", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_7", oracle: "InstructionRa_7", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_8", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_8", oracle: "InstructionRa_8", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_8", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_8", oracle: "InstructionRa_8", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_9", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_9", oracle: "InstructionRa_9", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_9", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_9", oracle: "InstructionRa_9", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_10", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_10", oracle: "InstructionRa_10", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_10", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_10", oracle: "InstructionRa_10", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_11", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_11", oracle: "InstructionRa_11", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_11", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_11", oracle: "InstructionRa_11", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_12", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_12", oracle: "InstructionRa_12", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_12", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_12", oracle: "InstructionRa_12", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_13", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_13", oracle: "InstructionRa_13", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_13", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_13", oracle: "InstructionRa_13", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_14", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_14", oracle: "InstructionRa_14", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_14", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_14", oracle: "InstructionRa_14", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_15", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_15", oracle: "InstructionRa_15", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_15", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_15", oracle: "InstructionRa_15", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_16", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_16", oracle: "InstructionRa_16", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_16", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_16", oracle: "InstructionRa_16", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_17", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_17", oracle: "InstructionRa_17", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_17", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_17", oracle: "InstructionRa_17", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_18", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_18", oracle: "InstructionRa_18", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_18", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_18", oracle: "InstructionRa_18", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_19", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_19", oracle: "InstructionRa_19", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_19", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_19", oracle: "InstructionRa_19", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_20", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_20", oracle: "InstructionRa_20", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_20", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_20", oracle: "InstructionRa_20", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_21", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_21", oracle: "InstructionRa_21", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_21", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_21", oracle: "InstructionRa_21", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_22", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_22", oracle: "InstructionRa_22", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_22", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_22", oracle: "InstructionRa_22", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_23", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_23", oracle: "InstructionRa_23", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_23", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_23", oracle: "InstructionRa_23", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_24", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_24", oracle: "InstructionRa_24", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_24", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_24", oracle: "InstructionRa_24", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_25", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_25", oracle: "InstructionRa_25", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_25", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_25", oracle: "InstructionRa_25", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_26", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_26", oracle: "InstructionRa_26", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_26", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_26", oracle: "InstructionRa_26", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_27", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_27", oracle: "InstructionRa_27", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_27", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_27", oracle: "InstructionRa_27", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_28", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_28", oracle: "InstructionRa_28", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_28", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_28", oracle: "InstructionRa_28", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_29", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_29", oracle: "InstructionRa_29", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_29", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_29", oracle: "InstructionRa_29", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_30", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_30", oracle: "InstructionRa_30", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_30", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_30", oracle: "InstructionRa_30", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.InstructionRa_31", source_stage: "stage6", source_claim: "stage6.booleanity.opening.InstructionRa_31", oracle: "InstructionRa_31", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_31", source_stage: "stage6", source_claim: "stage6.instruction_ra_virtual.opening.InstructionRa_31", oracle: "InstructionRa_31", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.BytecodeRa_0", source_stage: "stage6", source_claim: "stage6.booleanity.opening.BytecodeRa_0", oracle: "BytecodeRa_0", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.bytecode_read_raf.BytecodeRa_0", source_stage: "stage6", source_claim: "stage6.bytecode_read_raf.opening.BytecodeRa_0", oracle: "BytecodeRa_0", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.BytecodeRa_1", source_stage: "stage6", source_claim: "stage6.booleanity.opening.BytecodeRa_1", oracle: "BytecodeRa_1", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.bytecode_read_raf.BytecodeRa_1", source_stage: "stage6", source_claim: "stage6.bytecode_read_raf.opening.BytecodeRa_1", oracle: "BytecodeRa_1", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.BytecodeRa_2", source_stage: "stage6", source_claim: "stage6.booleanity.opening.BytecodeRa_2", oracle: "BytecodeRa_2", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.bytecode_read_raf.BytecodeRa_2", source_stage: "stage6", source_claim: "stage6.bytecode_read_raf.opening.BytecodeRa_2", oracle: "BytecodeRa_2", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.RamRa_0", source_stage: "stage6", source_claim: "stage6.booleanity.opening.RamRa_0", oracle: "RamRa_0", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.ram_ra_virtual.RamRa_0", source_stage: "stage6", source_claim: "stage6.ram_ra_virtual.opening.RamRa_0", oracle: "RamRa_0", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.RamRa_1", source_stage: "stage6", source_claim: "stage6.booleanity.opening.RamRa_1", oracle: "RamRa_1", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.ram_ra_virtual.RamRa_1", source_stage: "stage6", source_claim: "stage6.ram_ra_virtual.opening.RamRa_1", oracle: "RamRa_1", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.RamRa_2", source_stage: "stage6", source_claim: "stage6.booleanity.opening.RamRa_2", oracle: "RamRa_2", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.ram_ra_virtual.RamRa_2", source_stage: "stage6", source_claim: "stage6.ram_ra_virtual.opening.RamRa_2", oracle: "RamRa_2", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.booleanity.RamRa_3", source_stage: "stage6", source_claim: "stage6.booleanity.opening.RamRa_3", oracle: "RamRa_3", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage7OpeningInputPlan { symbol: "stage7.input.stage6.ram_ra_virtual.RamRa_3", source_stage: "stage6", source_claim: "stage6.ram_ra_virtual.opening.RamRa_3", oracle: "RamRa_3", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
];

pub const STAGE7_FIELD_CONSTANTS: &[Stage7FieldConstantPlan] = &[
    Stage7FieldConstantPlan { symbol: "stage7.field.one", field: "bn254_fr", value: 1 },
];

pub const STAGE7_FIELD_EXPR_0_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_0_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_1_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.0.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_0",
];

pub const STAGE7_FIELD_EXPR_1_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.0.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_0",
];

pub const STAGE7_FIELD_EXPR_2_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_2_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_3_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.0.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_0",
];

pub const STAGE7_FIELD_EXPR_3_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.0.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_0",
];

pub const STAGE7_FIELD_EXPR_4_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_4_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_5_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.1.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_5_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.1.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_6_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_6_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_7_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.1.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_1",
];

pub const STAGE7_FIELD_EXPR_7_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.1.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_1",
];

pub const STAGE7_FIELD_EXPR_8_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_8_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_9_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.1.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_1",
];

pub const STAGE7_FIELD_EXPR_9_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.1.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_1",
];

pub const STAGE7_FIELD_EXPR_10_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_10_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_11_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.2.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_11_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.2.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_12_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_12_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_13_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.2.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_2",
];

pub const STAGE7_FIELD_EXPR_13_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.2.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_2",
];

pub const STAGE7_FIELD_EXPR_14_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_14_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_15_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.2.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_2",
];

pub const STAGE7_FIELD_EXPR_15_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.2.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_2",
];

pub const STAGE7_FIELD_EXPR_16_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_16_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_17_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.3.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_17_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.3.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_18_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_18_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_19_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.3.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_3",
];

pub const STAGE7_FIELD_EXPR_19_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.3.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_3",
];

pub const STAGE7_FIELD_EXPR_20_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_20_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_21_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.3.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_3",
];

pub const STAGE7_FIELD_EXPR_21_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.3.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_3",
];

pub const STAGE7_FIELD_EXPR_22_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_22_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_23_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.4.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_23_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.4.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_24_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_24_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_25_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.4.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_4",
];

pub const STAGE7_FIELD_EXPR_25_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.4.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_4",
];

pub const STAGE7_FIELD_EXPR_26_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_26_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_27_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.4.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_4",
];

pub const STAGE7_FIELD_EXPR_27_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.4.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_4",
];

pub const STAGE7_FIELD_EXPR_28_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_28_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_29_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.5.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_29_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.5.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_30_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_30_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_31_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.5.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_5",
];

pub const STAGE7_FIELD_EXPR_31_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.5.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_5",
];

pub const STAGE7_FIELD_EXPR_32_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_32_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_33_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.5.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_5",
];

pub const STAGE7_FIELD_EXPR_33_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.5.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_5",
];

pub const STAGE7_FIELD_EXPR_34_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_34_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_35_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.6.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_35_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.6.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_36_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_36_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_37_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.6.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_6",
];

pub const STAGE7_FIELD_EXPR_37_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.6.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_6",
];

pub const STAGE7_FIELD_EXPR_38_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_38_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_39_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.6.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_6",
];

pub const STAGE7_FIELD_EXPR_39_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.6.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_6",
];

pub const STAGE7_FIELD_EXPR_40_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_40_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_41_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.7.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_41_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.7.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_42_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_42_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_43_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.7.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_7",
];

pub const STAGE7_FIELD_EXPR_43_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.7.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_7",
];

pub const STAGE7_FIELD_EXPR_44_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_44_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_45_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.7.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_7",
];

pub const STAGE7_FIELD_EXPR_45_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.7.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_7",
];

pub const STAGE7_FIELD_EXPR_46_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_46_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_47_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.8.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_47_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.8.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_48_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_48_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_49_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.8.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_8",
];

pub const STAGE7_FIELD_EXPR_49_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.8.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_8",
];

pub const STAGE7_FIELD_EXPR_50_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_50_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_51_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.8.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_8",
];

pub const STAGE7_FIELD_EXPR_51_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.8.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_8",
];

pub const STAGE7_FIELD_EXPR_52_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_52_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_53_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.9.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_53_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.9.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_54_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_54_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_55_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.9.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_9",
];

pub const STAGE7_FIELD_EXPR_55_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.9.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_9",
];

pub const STAGE7_FIELD_EXPR_56_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_56_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_57_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.9.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_9",
];

pub const STAGE7_FIELD_EXPR_57_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.9.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_9",
];

pub const STAGE7_FIELD_EXPR_58_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_58_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_59_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.10.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_59_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.10.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_60_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_60_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_61_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.10.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_10",
];

pub const STAGE7_FIELD_EXPR_61_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.10.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_10",
];

pub const STAGE7_FIELD_EXPR_62_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_62_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_63_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.10.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_10",
];

pub const STAGE7_FIELD_EXPR_63_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.10.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_10",
];

pub const STAGE7_FIELD_EXPR_64_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_64_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_65_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.11.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_65_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.11.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_66_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_66_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_67_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.11.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_11",
];

pub const STAGE7_FIELD_EXPR_67_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.11.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_11",
];

pub const STAGE7_FIELD_EXPR_68_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_68_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_69_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.11.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_11",
];

pub const STAGE7_FIELD_EXPR_69_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.11.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_11",
];

pub const STAGE7_FIELD_EXPR_70_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_70_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_71_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.12.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_71_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.12.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_72_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_72_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_73_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.12.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_12",
];

pub const STAGE7_FIELD_EXPR_73_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.12.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_12",
];

pub const STAGE7_FIELD_EXPR_74_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_74_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_75_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.12.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_12",
];

pub const STAGE7_FIELD_EXPR_75_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.12.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_12",
];

pub const STAGE7_FIELD_EXPR_76_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_76_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_77_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.13.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_77_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.13.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_78_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_78_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_79_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.13.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_13",
];

pub const STAGE7_FIELD_EXPR_79_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.13.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_13",
];

pub const STAGE7_FIELD_EXPR_80_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_80_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_81_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.13.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_13",
];

pub const STAGE7_FIELD_EXPR_81_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.13.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_13",
];

pub const STAGE7_FIELD_EXPR_82_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_82_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_83_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.14.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_83_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.14.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_84_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_84_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_85_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.14.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_14",
];

pub const STAGE7_FIELD_EXPR_85_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.14.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_14",
];

pub const STAGE7_FIELD_EXPR_86_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_86_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_87_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.14.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_14",
];

pub const STAGE7_FIELD_EXPR_87_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.14.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_14",
];

pub const STAGE7_FIELD_EXPR_88_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_88_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_89_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.15.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_89_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.15.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_90_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_90_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_91_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.15.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_15",
];

pub const STAGE7_FIELD_EXPR_91_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.15.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_15",
];

pub const STAGE7_FIELD_EXPR_92_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_92_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_93_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.15.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_15",
];

pub const STAGE7_FIELD_EXPR_93_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.15.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_15",
];

pub const STAGE7_FIELD_EXPR_94_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_94_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_95_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.16.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_95_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.16.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_96_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_96_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_97_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.16.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_16",
];

pub const STAGE7_FIELD_EXPR_97_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.16.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_16",
];

pub const STAGE7_FIELD_EXPR_98_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_98_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_99_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.16.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_16",
];

pub const STAGE7_FIELD_EXPR_99_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.16.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_16",
];

pub const STAGE7_FIELD_EXPR_100_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_100_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_101_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.17.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_101_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.17.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_102_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_102_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_103_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.17.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_17",
];

pub const STAGE7_FIELD_EXPR_103_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.17.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_17",
];

pub const STAGE7_FIELD_EXPR_104_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_104_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_105_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.17.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_17",
];

pub const STAGE7_FIELD_EXPR_105_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.17.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_17",
];

pub const STAGE7_FIELD_EXPR_106_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_106_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_107_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.18.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_107_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.18.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_108_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_108_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_109_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.18.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_18",
];

pub const STAGE7_FIELD_EXPR_109_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.18.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_18",
];

pub const STAGE7_FIELD_EXPR_110_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_110_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_111_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.18.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_18",
];

pub const STAGE7_FIELD_EXPR_111_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.18.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_18",
];

pub const STAGE7_FIELD_EXPR_112_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_112_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_113_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.19.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_113_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.19.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_114_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_114_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_115_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.19.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_19",
];

pub const STAGE7_FIELD_EXPR_115_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.19.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_19",
];

pub const STAGE7_FIELD_EXPR_116_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_116_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_117_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.19.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_19",
];

pub const STAGE7_FIELD_EXPR_117_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.19.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_19",
];

pub const STAGE7_FIELD_EXPR_118_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_118_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_119_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.20.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_119_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.20.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_120_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_120_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_121_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.20.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_20",
];

pub const STAGE7_FIELD_EXPR_121_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.20.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_20",
];

pub const STAGE7_FIELD_EXPR_122_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_122_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_123_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.20.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_20",
];

pub const STAGE7_FIELD_EXPR_123_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.20.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_20",
];

pub const STAGE7_FIELD_EXPR_124_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_124_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_125_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.21.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_125_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.21.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_126_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_126_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_127_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.21.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_21",
];

pub const STAGE7_FIELD_EXPR_127_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.21.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_21",
];

pub const STAGE7_FIELD_EXPR_128_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_128_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_129_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.21.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_21",
];

pub const STAGE7_FIELD_EXPR_129_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.21.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_21",
];

pub const STAGE7_FIELD_EXPR_130_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_130_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_131_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.22.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_131_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.22.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_132_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_132_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_133_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.22.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_22",
];

pub const STAGE7_FIELD_EXPR_133_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.22.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_22",
];

pub const STAGE7_FIELD_EXPR_134_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_134_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_135_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.22.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_22",
];

pub const STAGE7_FIELD_EXPR_135_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.22.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_22",
];

pub const STAGE7_FIELD_EXPR_136_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_136_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_137_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.23.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_137_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.23.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_138_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_138_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_139_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.23.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_23",
];

pub const STAGE7_FIELD_EXPR_139_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.23.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_23",
];

pub const STAGE7_FIELD_EXPR_140_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_140_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_141_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.23.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_23",
];

pub const STAGE7_FIELD_EXPR_141_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.23.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_23",
];

pub const STAGE7_FIELD_EXPR_142_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_142_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_143_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.24.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_143_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.24.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_144_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_144_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_145_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.24.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_24",
];

pub const STAGE7_FIELD_EXPR_145_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.24.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_24",
];

pub const STAGE7_FIELD_EXPR_146_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_146_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_147_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.24.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_24",
];

pub const STAGE7_FIELD_EXPR_147_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.24.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_24",
];

pub const STAGE7_FIELD_EXPR_148_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_148_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_149_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.25.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_149_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.25.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_150_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_150_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_151_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.25.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_25",
];

pub const STAGE7_FIELD_EXPR_151_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.25.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_25",
];

pub const STAGE7_FIELD_EXPR_152_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_152_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_153_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.25.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_25",
];

pub const STAGE7_FIELD_EXPR_153_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.25.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_25",
];

pub const STAGE7_FIELD_EXPR_154_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_154_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_155_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.26.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_155_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.26.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_156_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_156_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_157_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.26.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_26",
];

pub const STAGE7_FIELD_EXPR_157_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.26.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_26",
];

pub const STAGE7_FIELD_EXPR_158_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_158_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_159_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.26.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_26",
];

pub const STAGE7_FIELD_EXPR_159_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.26.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_26",
];

pub const STAGE7_FIELD_EXPR_160_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_160_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_161_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.27.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_161_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.27.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_162_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_162_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_163_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.27.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_27",
];

pub const STAGE7_FIELD_EXPR_163_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.27.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_27",
];

pub const STAGE7_FIELD_EXPR_164_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_164_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_165_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.27.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_27",
];

pub const STAGE7_FIELD_EXPR_165_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.27.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_27",
];

pub const STAGE7_FIELD_EXPR_166_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_166_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_167_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.28.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_167_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.28.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_168_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_168_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_169_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.28.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_28",
];

pub const STAGE7_FIELD_EXPR_169_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.28.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_28",
];

pub const STAGE7_FIELD_EXPR_170_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_170_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_171_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.28.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_28",
];

pub const STAGE7_FIELD_EXPR_171_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.28.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_28",
];

pub const STAGE7_FIELD_EXPR_172_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_172_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_173_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.29.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_173_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.29.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_174_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_174_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_175_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.29.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_29",
];

pub const STAGE7_FIELD_EXPR_175_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.29.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_29",
];

pub const STAGE7_FIELD_EXPR_176_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_176_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_177_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.29.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_29",
];

pub const STAGE7_FIELD_EXPR_177_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.29.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_29",
];

pub const STAGE7_FIELD_EXPR_178_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_178_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_179_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.30.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_179_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.30.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_180_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_180_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_181_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.30.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_30",
];

pub const STAGE7_FIELD_EXPR_181_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.30.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_30",
];

pub const STAGE7_FIELD_EXPR_182_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_182_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_183_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.30.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_30",
];

pub const STAGE7_FIELD_EXPR_183_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.30.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_30",
];

pub const STAGE7_FIELD_EXPR_184_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_184_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_185_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.31.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_185_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.31.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_186_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_186_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_187_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.31.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_31",
];

pub const STAGE7_FIELD_EXPR_187_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.31.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.InstructionRa_31",
];

pub const STAGE7_FIELD_EXPR_188_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_188_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_189_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.31.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_31",
];

pub const STAGE7_FIELD_EXPR_189_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.31.virtualization.gamma_pow",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_31",
];

pub const STAGE7_FIELD_EXPR_190_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_190_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_191_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.32.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_191_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.32.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_192_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_192_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_193_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.32.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.BytecodeRa_0",
];

pub const STAGE7_FIELD_EXPR_193_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.32.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.BytecodeRa_0",
];

pub const STAGE7_FIELD_EXPR_194_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_194_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_195_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.32.virtualization.gamma_pow",
    "stage7.input.stage6.bytecode_read_raf.BytecodeRa_0",
];

pub const STAGE7_FIELD_EXPR_195_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.32.virtualization.gamma_pow",
    "stage7.input.stage6.bytecode_read_raf.BytecodeRa_0",
];

pub const STAGE7_FIELD_EXPR_196_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_196_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_197_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.33.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_197_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.33.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_198_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_198_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_199_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.33.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.BytecodeRa_1",
];

pub const STAGE7_FIELD_EXPR_199_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.33.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.BytecodeRa_1",
];

pub const STAGE7_FIELD_EXPR_200_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_200_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_201_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.33.virtualization.gamma_pow",
    "stage7.input.stage6.bytecode_read_raf.BytecodeRa_1",
];

pub const STAGE7_FIELD_EXPR_201_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.33.virtualization.gamma_pow",
    "stage7.input.stage6.bytecode_read_raf.BytecodeRa_1",
];

pub const STAGE7_FIELD_EXPR_202_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_202_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_203_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.34.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_203_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.34.hw.gamma_pow",
    "stage7.field.one",
];

pub const STAGE7_FIELD_EXPR_204_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_204_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_205_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.34.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.BytecodeRa_2",
];

pub const STAGE7_FIELD_EXPR_205_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.34.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.BytecodeRa_2",
];

pub const STAGE7_FIELD_EXPR_206_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_206_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_207_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.34.virtualization.gamma_pow",
    "stage7.input.stage6.bytecode_read_raf.BytecodeRa_2",
];

pub const STAGE7_FIELD_EXPR_207_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.34.virtualization.gamma_pow",
    "stage7.input.stage6.bytecode_read_raf.BytecodeRa_2",
];

pub const STAGE7_FIELD_EXPR_208_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_208_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_209_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.35.hw.gamma_pow",
    "stage7.input.stage6.hamming_booleanity.HammingWeight",
];

pub const STAGE7_FIELD_EXPR_209_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.35.hw.gamma_pow",
    "stage7.input.stage6.hamming_booleanity.HammingWeight",
];

pub const STAGE7_FIELD_EXPR_210_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_210_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_211_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.35.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.RamRa_0",
];

pub const STAGE7_FIELD_EXPR_211_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.35.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.RamRa_0",
];

pub const STAGE7_FIELD_EXPR_212_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_212_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_213_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.35.virtualization.gamma_pow",
    "stage7.input.stage6.ram_ra_virtual.RamRa_0",
];

pub const STAGE7_FIELD_EXPR_213_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.35.virtualization.gamma_pow",
    "stage7.input.stage6.ram_ra_virtual.RamRa_0",
];

pub const STAGE7_FIELD_EXPR_214_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_214_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_215_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.36.hw.gamma_pow",
    "stage7.input.stage6.hamming_booleanity.HammingWeight",
];

pub const STAGE7_FIELD_EXPR_215_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.36.hw.gamma_pow",
    "stage7.input.stage6.hamming_booleanity.HammingWeight",
];

pub const STAGE7_FIELD_EXPR_216_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_216_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_217_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.36.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.RamRa_1",
];

pub const STAGE7_FIELD_EXPR_217_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.36.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.RamRa_1",
];

pub const STAGE7_FIELD_EXPR_218_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_218_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_219_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.36.virtualization.gamma_pow",
    "stage7.input.stage6.ram_ra_virtual.RamRa_1",
];

pub const STAGE7_FIELD_EXPR_219_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.36.virtualization.gamma_pow",
    "stage7.input.stage6.ram_ra_virtual.RamRa_1",
];

pub const STAGE7_FIELD_EXPR_220_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_220_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_221_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.37.hw.gamma_pow",
    "stage7.input.stage6.hamming_booleanity.HammingWeight",
];

pub const STAGE7_FIELD_EXPR_221_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.37.hw.gamma_pow",
    "stage7.input.stage6.hamming_booleanity.HammingWeight",
];

pub const STAGE7_FIELD_EXPR_222_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_222_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_223_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.37.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.RamRa_2",
];

pub const STAGE7_FIELD_EXPR_223_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.37.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.RamRa_2",
];

pub const STAGE7_FIELD_EXPR_224_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_224_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_225_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.37.virtualization.gamma_pow",
    "stage7.input.stage6.ram_ra_virtual.RamRa_2",
];

pub const STAGE7_FIELD_EXPR_225_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.37.virtualization.gamma_pow",
    "stage7.input.stage6.ram_ra_virtual.RamRa_2",
];

pub const STAGE7_FIELD_EXPR_226_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_226_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_227_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.38.hw.gamma_pow",
    "stage7.input.stage6.hamming_booleanity.HammingWeight",
];

pub const STAGE7_FIELD_EXPR_227_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.38.hw.gamma_pow",
    "stage7.input.stage6.hamming_booleanity.HammingWeight",
];

pub const STAGE7_FIELD_EXPR_228_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_228_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_229_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.38.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.RamRa_3",
];

pub const STAGE7_FIELD_EXPR_229_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.38.booleanity.gamma_pow",
    "stage7.input.stage6.booleanity.RamRa_3",
];

pub const STAGE7_FIELD_EXPR_230_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_230_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.gamma",
];

pub const STAGE7_FIELD_EXPR_231_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.38.virtualization.gamma_pow",
    "stage7.input.stage6.ram_ra_virtual.RamRa_3",
];

pub const STAGE7_FIELD_EXPR_231_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim.38.virtualization.gamma_pow",
    "stage7.input.stage6.ram_ra_virtual.RamRa_3",
];

pub const STAGE7_FIELD_EXPR_232_OPERAND_NAMES: &[&str] = &[
    "stage7.field.one",
    "stage7.hamming_weight_claim_reduction.claim.0.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_232_OPERANDS: &[&str] = &[
    "stage7.field.one",
    "stage7.hamming_weight_claim_reduction.claim.0.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_233_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial0",
    "stage7.hamming_weight_claim_reduction.claim.0.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_233_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial0",
    "stage7.hamming_weight_claim_reduction.claim.0.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_234_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial1",
    "stage7.hamming_weight_claim_reduction.claim.1.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_234_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial1",
    "stage7.hamming_weight_claim_reduction.claim.1.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_235_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial2",
    "stage7.hamming_weight_claim_reduction.claim.1.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_235_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial2",
    "stage7.hamming_weight_claim_reduction.claim.1.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_236_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial3",
    "stage7.hamming_weight_claim_reduction.claim.1.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_236_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial3",
    "stage7.hamming_weight_claim_reduction.claim.1.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_237_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial4",
    "stage7.hamming_weight_claim_reduction.claim.2.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_237_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial4",
    "stage7.hamming_weight_claim_reduction.claim.2.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_238_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial5",
    "stage7.hamming_weight_claim_reduction.claim.2.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_238_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial5",
    "stage7.hamming_weight_claim_reduction.claim.2.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_239_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial6",
    "stage7.hamming_weight_claim_reduction.claim.2.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_239_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial6",
    "stage7.hamming_weight_claim_reduction.claim.2.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_240_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial7",
    "stage7.hamming_weight_claim_reduction.claim.3.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_240_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial7",
    "stage7.hamming_weight_claim_reduction.claim.3.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_241_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial8",
    "stage7.hamming_weight_claim_reduction.claim.3.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_241_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial8",
    "stage7.hamming_weight_claim_reduction.claim.3.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_242_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial9",
    "stage7.hamming_weight_claim_reduction.claim.3.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_242_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial9",
    "stage7.hamming_weight_claim_reduction.claim.3.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_243_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial10",
    "stage7.hamming_weight_claim_reduction.claim.4.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_243_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial10",
    "stage7.hamming_weight_claim_reduction.claim.4.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_244_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial11",
    "stage7.hamming_weight_claim_reduction.claim.4.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_244_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial11",
    "stage7.hamming_weight_claim_reduction.claim.4.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_245_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial12",
    "stage7.hamming_weight_claim_reduction.claim.4.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_245_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial12",
    "stage7.hamming_weight_claim_reduction.claim.4.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_246_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial13",
    "stage7.hamming_weight_claim_reduction.claim.5.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_246_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial13",
    "stage7.hamming_weight_claim_reduction.claim.5.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_247_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial14",
    "stage7.hamming_weight_claim_reduction.claim.5.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_247_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial14",
    "stage7.hamming_weight_claim_reduction.claim.5.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_248_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial15",
    "stage7.hamming_weight_claim_reduction.claim.5.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_248_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial15",
    "stage7.hamming_weight_claim_reduction.claim.5.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_249_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial16",
    "stage7.hamming_weight_claim_reduction.claim.6.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_249_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial16",
    "stage7.hamming_weight_claim_reduction.claim.6.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_250_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial17",
    "stage7.hamming_weight_claim_reduction.claim.6.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_250_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial17",
    "stage7.hamming_weight_claim_reduction.claim.6.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_251_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial18",
    "stage7.hamming_weight_claim_reduction.claim.6.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_251_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial18",
    "stage7.hamming_weight_claim_reduction.claim.6.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_252_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial19",
    "stage7.hamming_weight_claim_reduction.claim.7.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_252_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial19",
    "stage7.hamming_weight_claim_reduction.claim.7.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_253_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial20",
    "stage7.hamming_weight_claim_reduction.claim.7.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_253_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial20",
    "stage7.hamming_weight_claim_reduction.claim.7.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_254_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial21",
    "stage7.hamming_weight_claim_reduction.claim.7.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_254_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial21",
    "stage7.hamming_weight_claim_reduction.claim.7.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_255_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial22",
    "stage7.hamming_weight_claim_reduction.claim.8.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_255_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial22",
    "stage7.hamming_weight_claim_reduction.claim.8.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_256_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial23",
    "stage7.hamming_weight_claim_reduction.claim.8.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_256_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial23",
    "stage7.hamming_weight_claim_reduction.claim.8.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_257_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial24",
    "stage7.hamming_weight_claim_reduction.claim.8.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_257_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial24",
    "stage7.hamming_weight_claim_reduction.claim.8.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_258_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial25",
    "stage7.hamming_weight_claim_reduction.claim.9.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_258_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial25",
    "stage7.hamming_weight_claim_reduction.claim.9.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_259_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial26",
    "stage7.hamming_weight_claim_reduction.claim.9.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_259_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial26",
    "stage7.hamming_weight_claim_reduction.claim.9.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_260_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial27",
    "stage7.hamming_weight_claim_reduction.claim.9.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_260_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial27",
    "stage7.hamming_weight_claim_reduction.claim.9.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_261_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial28",
    "stage7.hamming_weight_claim_reduction.claim.10.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_261_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial28",
    "stage7.hamming_weight_claim_reduction.claim.10.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_262_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial29",
    "stage7.hamming_weight_claim_reduction.claim.10.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_262_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial29",
    "stage7.hamming_weight_claim_reduction.claim.10.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_263_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial30",
    "stage7.hamming_weight_claim_reduction.claim.10.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_263_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial30",
    "stage7.hamming_weight_claim_reduction.claim.10.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_264_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial31",
    "stage7.hamming_weight_claim_reduction.claim.11.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_264_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial31",
    "stage7.hamming_weight_claim_reduction.claim.11.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_265_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial32",
    "stage7.hamming_weight_claim_reduction.claim.11.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_265_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial32",
    "stage7.hamming_weight_claim_reduction.claim.11.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_266_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial33",
    "stage7.hamming_weight_claim_reduction.claim.11.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_266_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial33",
    "stage7.hamming_weight_claim_reduction.claim.11.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_267_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial34",
    "stage7.hamming_weight_claim_reduction.claim.12.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_267_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial34",
    "stage7.hamming_weight_claim_reduction.claim.12.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_268_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial35",
    "stage7.hamming_weight_claim_reduction.claim.12.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_268_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial35",
    "stage7.hamming_weight_claim_reduction.claim.12.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_269_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial36",
    "stage7.hamming_weight_claim_reduction.claim.12.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_269_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial36",
    "stage7.hamming_weight_claim_reduction.claim.12.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_270_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial37",
    "stage7.hamming_weight_claim_reduction.claim.13.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_270_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial37",
    "stage7.hamming_weight_claim_reduction.claim.13.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_271_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial38",
    "stage7.hamming_weight_claim_reduction.claim.13.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_271_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial38",
    "stage7.hamming_weight_claim_reduction.claim.13.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_272_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial39",
    "stage7.hamming_weight_claim_reduction.claim.13.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_272_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial39",
    "stage7.hamming_weight_claim_reduction.claim.13.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_273_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial40",
    "stage7.hamming_weight_claim_reduction.claim.14.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_273_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial40",
    "stage7.hamming_weight_claim_reduction.claim.14.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_274_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial41",
    "stage7.hamming_weight_claim_reduction.claim.14.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_274_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial41",
    "stage7.hamming_weight_claim_reduction.claim.14.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_275_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial42",
    "stage7.hamming_weight_claim_reduction.claim.14.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_275_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial42",
    "stage7.hamming_weight_claim_reduction.claim.14.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_276_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial43",
    "stage7.hamming_weight_claim_reduction.claim.15.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_276_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial43",
    "stage7.hamming_weight_claim_reduction.claim.15.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_277_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial44",
    "stage7.hamming_weight_claim_reduction.claim.15.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_277_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial44",
    "stage7.hamming_weight_claim_reduction.claim.15.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_278_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial45",
    "stage7.hamming_weight_claim_reduction.claim.15.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_278_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial45",
    "stage7.hamming_weight_claim_reduction.claim.15.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_279_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial46",
    "stage7.hamming_weight_claim_reduction.claim.16.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_279_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial46",
    "stage7.hamming_weight_claim_reduction.claim.16.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_280_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial47",
    "stage7.hamming_weight_claim_reduction.claim.16.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_280_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial47",
    "stage7.hamming_weight_claim_reduction.claim.16.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_281_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial48",
    "stage7.hamming_weight_claim_reduction.claim.16.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_281_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial48",
    "stage7.hamming_weight_claim_reduction.claim.16.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_282_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial49",
    "stage7.hamming_weight_claim_reduction.claim.17.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_282_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial49",
    "stage7.hamming_weight_claim_reduction.claim.17.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_283_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial50",
    "stage7.hamming_weight_claim_reduction.claim.17.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_283_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial50",
    "stage7.hamming_weight_claim_reduction.claim.17.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_284_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial51",
    "stage7.hamming_weight_claim_reduction.claim.17.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_284_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial51",
    "stage7.hamming_weight_claim_reduction.claim.17.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_285_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial52",
    "stage7.hamming_weight_claim_reduction.claim.18.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_285_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial52",
    "stage7.hamming_weight_claim_reduction.claim.18.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_286_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial53",
    "stage7.hamming_weight_claim_reduction.claim.18.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_286_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial53",
    "stage7.hamming_weight_claim_reduction.claim.18.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_287_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial54",
    "stage7.hamming_weight_claim_reduction.claim.18.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_287_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial54",
    "stage7.hamming_weight_claim_reduction.claim.18.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_288_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial55",
    "stage7.hamming_weight_claim_reduction.claim.19.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_288_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial55",
    "stage7.hamming_weight_claim_reduction.claim.19.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_289_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial56",
    "stage7.hamming_weight_claim_reduction.claim.19.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_289_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial56",
    "stage7.hamming_weight_claim_reduction.claim.19.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_290_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial57",
    "stage7.hamming_weight_claim_reduction.claim.19.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_290_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial57",
    "stage7.hamming_weight_claim_reduction.claim.19.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_291_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial58",
    "stage7.hamming_weight_claim_reduction.claim.20.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_291_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial58",
    "stage7.hamming_weight_claim_reduction.claim.20.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_292_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial59",
    "stage7.hamming_weight_claim_reduction.claim.20.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_292_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial59",
    "stage7.hamming_weight_claim_reduction.claim.20.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_293_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial60",
    "stage7.hamming_weight_claim_reduction.claim.20.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_293_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial60",
    "stage7.hamming_weight_claim_reduction.claim.20.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_294_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial61",
    "stage7.hamming_weight_claim_reduction.claim.21.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_294_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial61",
    "stage7.hamming_weight_claim_reduction.claim.21.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_295_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial62",
    "stage7.hamming_weight_claim_reduction.claim.21.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_295_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial62",
    "stage7.hamming_weight_claim_reduction.claim.21.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_296_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial63",
    "stage7.hamming_weight_claim_reduction.claim.21.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_296_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial63",
    "stage7.hamming_weight_claim_reduction.claim.21.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_297_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial64",
    "stage7.hamming_weight_claim_reduction.claim.22.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_297_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial64",
    "stage7.hamming_weight_claim_reduction.claim.22.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_298_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial65",
    "stage7.hamming_weight_claim_reduction.claim.22.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_298_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial65",
    "stage7.hamming_weight_claim_reduction.claim.22.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_299_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial66",
    "stage7.hamming_weight_claim_reduction.claim.22.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_299_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial66",
    "stage7.hamming_weight_claim_reduction.claim.22.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_300_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial67",
    "stage7.hamming_weight_claim_reduction.claim.23.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_300_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial67",
    "stage7.hamming_weight_claim_reduction.claim.23.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_301_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial68",
    "stage7.hamming_weight_claim_reduction.claim.23.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_301_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial68",
    "stage7.hamming_weight_claim_reduction.claim.23.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_302_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial69",
    "stage7.hamming_weight_claim_reduction.claim.23.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_302_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial69",
    "stage7.hamming_weight_claim_reduction.claim.23.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_303_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial70",
    "stage7.hamming_weight_claim_reduction.claim.24.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_303_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial70",
    "stage7.hamming_weight_claim_reduction.claim.24.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_304_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial71",
    "stage7.hamming_weight_claim_reduction.claim.24.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_304_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial71",
    "stage7.hamming_weight_claim_reduction.claim.24.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_305_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial72",
    "stage7.hamming_weight_claim_reduction.claim.24.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_305_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial72",
    "stage7.hamming_weight_claim_reduction.claim.24.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_306_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial73",
    "stage7.hamming_weight_claim_reduction.claim.25.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_306_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial73",
    "stage7.hamming_weight_claim_reduction.claim.25.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_307_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial74",
    "stage7.hamming_weight_claim_reduction.claim.25.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_307_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial74",
    "stage7.hamming_weight_claim_reduction.claim.25.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_308_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial75",
    "stage7.hamming_weight_claim_reduction.claim.25.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_308_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial75",
    "stage7.hamming_weight_claim_reduction.claim.25.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_309_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial76",
    "stage7.hamming_weight_claim_reduction.claim.26.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_309_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial76",
    "stage7.hamming_weight_claim_reduction.claim.26.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_310_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial77",
    "stage7.hamming_weight_claim_reduction.claim.26.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_310_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial77",
    "stage7.hamming_weight_claim_reduction.claim.26.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_311_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial78",
    "stage7.hamming_weight_claim_reduction.claim.26.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_311_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial78",
    "stage7.hamming_weight_claim_reduction.claim.26.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_312_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial79",
    "stage7.hamming_weight_claim_reduction.claim.27.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_312_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial79",
    "stage7.hamming_weight_claim_reduction.claim.27.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_313_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial80",
    "stage7.hamming_weight_claim_reduction.claim.27.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_313_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial80",
    "stage7.hamming_weight_claim_reduction.claim.27.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_314_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial81",
    "stage7.hamming_weight_claim_reduction.claim.27.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_314_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial81",
    "stage7.hamming_weight_claim_reduction.claim.27.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_315_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial82",
    "stage7.hamming_weight_claim_reduction.claim.28.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_315_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial82",
    "stage7.hamming_weight_claim_reduction.claim.28.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_316_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial83",
    "stage7.hamming_weight_claim_reduction.claim.28.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_316_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial83",
    "stage7.hamming_weight_claim_reduction.claim.28.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_317_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial84",
    "stage7.hamming_weight_claim_reduction.claim.28.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_317_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial84",
    "stage7.hamming_weight_claim_reduction.claim.28.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_318_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial85",
    "stage7.hamming_weight_claim_reduction.claim.29.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_318_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial85",
    "stage7.hamming_weight_claim_reduction.claim.29.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_319_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial86",
    "stage7.hamming_weight_claim_reduction.claim.29.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_319_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial86",
    "stage7.hamming_weight_claim_reduction.claim.29.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_320_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial87",
    "stage7.hamming_weight_claim_reduction.claim.29.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_320_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial87",
    "stage7.hamming_weight_claim_reduction.claim.29.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_321_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial88",
    "stage7.hamming_weight_claim_reduction.claim.30.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_321_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial88",
    "stage7.hamming_weight_claim_reduction.claim.30.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_322_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial89",
    "stage7.hamming_weight_claim_reduction.claim.30.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_322_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial89",
    "stage7.hamming_weight_claim_reduction.claim.30.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_323_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial90",
    "stage7.hamming_weight_claim_reduction.claim.30.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_323_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial90",
    "stage7.hamming_weight_claim_reduction.claim.30.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_324_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial91",
    "stage7.hamming_weight_claim_reduction.claim.31.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_324_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial91",
    "stage7.hamming_weight_claim_reduction.claim.31.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_325_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial92",
    "stage7.hamming_weight_claim_reduction.claim.31.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_325_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial92",
    "stage7.hamming_weight_claim_reduction.claim.31.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_326_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial93",
    "stage7.hamming_weight_claim_reduction.claim.31.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_326_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial93",
    "stage7.hamming_weight_claim_reduction.claim.31.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_327_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial94",
    "stage7.hamming_weight_claim_reduction.claim.32.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_327_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial94",
    "stage7.hamming_weight_claim_reduction.claim.32.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_328_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial95",
    "stage7.hamming_weight_claim_reduction.claim.32.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_328_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial95",
    "stage7.hamming_weight_claim_reduction.claim.32.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_329_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial96",
    "stage7.hamming_weight_claim_reduction.claim.32.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_329_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial96",
    "stage7.hamming_weight_claim_reduction.claim.32.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_330_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial97",
    "stage7.hamming_weight_claim_reduction.claim.33.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_330_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial97",
    "stage7.hamming_weight_claim_reduction.claim.33.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_331_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial98",
    "stage7.hamming_weight_claim_reduction.claim.33.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_331_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial98",
    "stage7.hamming_weight_claim_reduction.claim.33.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_332_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial99",
    "stage7.hamming_weight_claim_reduction.claim.33.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_332_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial99",
    "stage7.hamming_weight_claim_reduction.claim.33.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_333_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial100",
    "stage7.hamming_weight_claim_reduction.claim.34.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_333_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial100",
    "stage7.hamming_weight_claim_reduction.claim.34.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_334_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial101",
    "stage7.hamming_weight_claim_reduction.claim.34.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_334_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial101",
    "stage7.hamming_weight_claim_reduction.claim.34.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_335_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial102",
    "stage7.hamming_weight_claim_reduction.claim.34.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_335_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial102",
    "stage7.hamming_weight_claim_reduction.claim.34.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_336_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial103",
    "stage7.hamming_weight_claim_reduction.claim.35.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_336_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial103",
    "stage7.hamming_weight_claim_reduction.claim.35.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_337_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial104",
    "stage7.hamming_weight_claim_reduction.claim.35.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_337_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial104",
    "stage7.hamming_weight_claim_reduction.claim.35.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_338_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial105",
    "stage7.hamming_weight_claim_reduction.claim.35.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_338_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial105",
    "stage7.hamming_weight_claim_reduction.claim.35.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_339_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial106",
    "stage7.hamming_weight_claim_reduction.claim.36.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_339_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial106",
    "stage7.hamming_weight_claim_reduction.claim.36.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_340_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial107",
    "stage7.hamming_weight_claim_reduction.claim.36.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_340_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial107",
    "stage7.hamming_weight_claim_reduction.claim.36.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_341_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial108",
    "stage7.hamming_weight_claim_reduction.claim.36.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_341_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial108",
    "stage7.hamming_weight_claim_reduction.claim.36.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_342_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial109",
    "stage7.hamming_weight_claim_reduction.claim.37.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_342_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial109",
    "stage7.hamming_weight_claim_reduction.claim.37.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_343_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial110",
    "stage7.hamming_weight_claim_reduction.claim.37.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_343_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial110",
    "stage7.hamming_weight_claim_reduction.claim.37.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_344_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial111",
    "stage7.hamming_weight_claim_reduction.claim.37.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_344_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial111",
    "stage7.hamming_weight_claim_reduction.claim.37.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_345_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial112",
    "stage7.hamming_weight_claim_reduction.claim.38.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_345_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial112",
    "stage7.hamming_weight_claim_reduction.claim.38.hw.gamma_term",
];

pub const STAGE7_FIELD_EXPR_346_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial113",
    "stage7.hamming_weight_claim_reduction.claim.38.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_346_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial113",
    "stage7.hamming_weight_claim_reduction.claim.38.booleanity.gamma_term",
];

pub const STAGE7_FIELD_EXPR_347_OPERAND_NAMES: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial114",
    "stage7.hamming_weight_claim_reduction.claim.38.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPR_347_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.claim_expr.partial114",
    "stage7.hamming_weight_claim_reduction.claim.38.virtualization.gamma_term",
];

pub const STAGE7_FIELD_EXPRS: &[Stage7FieldExprPlan] = &[
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.0.booleanity.gamma_pow", kind: "op", formula: "field.pow:1", operand_names: STAGE7_FIELD_EXPR_0_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_0_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.0.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_1_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_1_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.0.virtualization.gamma_pow", kind: "op", formula: "field.pow:2", operand_names: STAGE7_FIELD_EXPR_2_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_2_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.0.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_3_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_3_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.1.hw.gamma_pow", kind: "op", formula: "field.pow:3", operand_names: STAGE7_FIELD_EXPR_4_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_4_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.1.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_5_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_5_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.1.booleanity.gamma_pow", kind: "op", formula: "field.pow:4", operand_names: STAGE7_FIELD_EXPR_6_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_6_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.1.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_7_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_7_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.1.virtualization.gamma_pow", kind: "op", formula: "field.pow:5", operand_names: STAGE7_FIELD_EXPR_8_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_8_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.1.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_9_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_9_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.2.hw.gamma_pow", kind: "op", formula: "field.pow:6", operand_names: STAGE7_FIELD_EXPR_10_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_10_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.2.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_11_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_11_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.2.booleanity.gamma_pow", kind: "op", formula: "field.pow:7", operand_names: STAGE7_FIELD_EXPR_12_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_12_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.2.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_13_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_13_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.2.virtualization.gamma_pow", kind: "op", formula: "field.pow:8", operand_names: STAGE7_FIELD_EXPR_14_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_14_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.2.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_15_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_15_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.3.hw.gamma_pow", kind: "op", formula: "field.pow:9", operand_names: STAGE7_FIELD_EXPR_16_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_16_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.3.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_17_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_17_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.3.booleanity.gamma_pow", kind: "op", formula: "field.pow:10", operand_names: STAGE7_FIELD_EXPR_18_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_18_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.3.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_19_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_19_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.3.virtualization.gamma_pow", kind: "op", formula: "field.pow:11", operand_names: STAGE7_FIELD_EXPR_20_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_20_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.3.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_21_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_21_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.4.hw.gamma_pow", kind: "op", formula: "field.pow:12", operand_names: STAGE7_FIELD_EXPR_22_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_22_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.4.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_23_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_23_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.4.booleanity.gamma_pow", kind: "op", formula: "field.pow:13", operand_names: STAGE7_FIELD_EXPR_24_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_24_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.4.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_25_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_25_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.4.virtualization.gamma_pow", kind: "op", formula: "field.pow:14", operand_names: STAGE7_FIELD_EXPR_26_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_26_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.4.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_27_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_27_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.5.hw.gamma_pow", kind: "op", formula: "field.pow:15", operand_names: STAGE7_FIELD_EXPR_28_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_28_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.5.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_29_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_29_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.5.booleanity.gamma_pow", kind: "op", formula: "field.pow:16", operand_names: STAGE7_FIELD_EXPR_30_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_30_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.5.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_31_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_31_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.5.virtualization.gamma_pow", kind: "op", formula: "field.pow:17", operand_names: STAGE7_FIELD_EXPR_32_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_32_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.5.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_33_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_33_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.6.hw.gamma_pow", kind: "op", formula: "field.pow:18", operand_names: STAGE7_FIELD_EXPR_34_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_34_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.6.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_35_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_35_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.6.booleanity.gamma_pow", kind: "op", formula: "field.pow:19", operand_names: STAGE7_FIELD_EXPR_36_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_36_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.6.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_37_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_37_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.6.virtualization.gamma_pow", kind: "op", formula: "field.pow:20", operand_names: STAGE7_FIELD_EXPR_38_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_38_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.6.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_39_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_39_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.7.hw.gamma_pow", kind: "op", formula: "field.pow:21", operand_names: STAGE7_FIELD_EXPR_40_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_40_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.7.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_41_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_41_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.7.booleanity.gamma_pow", kind: "op", formula: "field.pow:22", operand_names: STAGE7_FIELD_EXPR_42_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_42_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.7.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_43_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_43_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.7.virtualization.gamma_pow", kind: "op", formula: "field.pow:23", operand_names: STAGE7_FIELD_EXPR_44_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_44_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.7.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_45_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_45_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.8.hw.gamma_pow", kind: "op", formula: "field.pow:24", operand_names: STAGE7_FIELD_EXPR_46_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_46_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.8.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_47_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_47_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.8.booleanity.gamma_pow", kind: "op", formula: "field.pow:25", operand_names: STAGE7_FIELD_EXPR_48_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_48_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.8.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_49_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_49_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.8.virtualization.gamma_pow", kind: "op", formula: "field.pow:26", operand_names: STAGE7_FIELD_EXPR_50_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_50_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.8.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_51_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_51_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.9.hw.gamma_pow", kind: "op", formula: "field.pow:27", operand_names: STAGE7_FIELD_EXPR_52_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_52_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.9.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_53_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_53_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.9.booleanity.gamma_pow", kind: "op", formula: "field.pow:28", operand_names: STAGE7_FIELD_EXPR_54_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_54_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.9.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_55_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_55_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.9.virtualization.gamma_pow", kind: "op", formula: "field.pow:29", operand_names: STAGE7_FIELD_EXPR_56_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_56_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.9.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_57_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_57_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.10.hw.gamma_pow", kind: "op", formula: "field.pow:30", operand_names: STAGE7_FIELD_EXPR_58_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_58_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.10.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_59_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_59_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.10.booleanity.gamma_pow", kind: "op", formula: "field.pow:31", operand_names: STAGE7_FIELD_EXPR_60_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_60_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.10.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_61_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_61_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.10.virtualization.gamma_pow", kind: "op", formula: "field.pow:32", operand_names: STAGE7_FIELD_EXPR_62_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_62_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.10.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_63_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_63_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.11.hw.gamma_pow", kind: "op", formula: "field.pow:33", operand_names: STAGE7_FIELD_EXPR_64_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_64_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.11.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_65_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_65_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.11.booleanity.gamma_pow", kind: "op", formula: "field.pow:34", operand_names: STAGE7_FIELD_EXPR_66_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_66_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.11.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_67_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_67_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.11.virtualization.gamma_pow", kind: "op", formula: "field.pow:35", operand_names: STAGE7_FIELD_EXPR_68_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_68_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.11.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_69_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_69_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.12.hw.gamma_pow", kind: "op", formula: "field.pow:36", operand_names: STAGE7_FIELD_EXPR_70_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_70_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.12.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_71_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_71_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.12.booleanity.gamma_pow", kind: "op", formula: "field.pow:37", operand_names: STAGE7_FIELD_EXPR_72_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_72_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.12.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_73_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_73_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.12.virtualization.gamma_pow", kind: "op", formula: "field.pow:38", operand_names: STAGE7_FIELD_EXPR_74_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_74_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.12.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_75_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_75_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.13.hw.gamma_pow", kind: "op", formula: "field.pow:39", operand_names: STAGE7_FIELD_EXPR_76_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_76_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.13.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_77_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_77_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.13.booleanity.gamma_pow", kind: "op", formula: "field.pow:40", operand_names: STAGE7_FIELD_EXPR_78_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_78_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.13.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_79_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_79_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.13.virtualization.gamma_pow", kind: "op", formula: "field.pow:41", operand_names: STAGE7_FIELD_EXPR_80_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_80_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.13.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_81_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_81_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.14.hw.gamma_pow", kind: "op", formula: "field.pow:42", operand_names: STAGE7_FIELD_EXPR_82_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_82_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.14.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_83_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_83_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.14.booleanity.gamma_pow", kind: "op", formula: "field.pow:43", operand_names: STAGE7_FIELD_EXPR_84_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_84_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.14.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_85_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_85_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.14.virtualization.gamma_pow", kind: "op", formula: "field.pow:44", operand_names: STAGE7_FIELD_EXPR_86_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_86_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.14.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_87_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_87_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.15.hw.gamma_pow", kind: "op", formula: "field.pow:45", operand_names: STAGE7_FIELD_EXPR_88_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_88_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.15.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_89_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_89_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.15.booleanity.gamma_pow", kind: "op", formula: "field.pow:46", operand_names: STAGE7_FIELD_EXPR_90_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_90_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.15.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_91_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_91_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.15.virtualization.gamma_pow", kind: "op", formula: "field.pow:47", operand_names: STAGE7_FIELD_EXPR_92_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_92_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.15.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_93_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_93_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.16.hw.gamma_pow", kind: "op", formula: "field.pow:48", operand_names: STAGE7_FIELD_EXPR_94_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_94_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.16.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_95_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_95_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.16.booleanity.gamma_pow", kind: "op", formula: "field.pow:49", operand_names: STAGE7_FIELD_EXPR_96_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_96_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.16.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_97_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_97_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.16.virtualization.gamma_pow", kind: "op", formula: "field.pow:50", operand_names: STAGE7_FIELD_EXPR_98_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_98_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.16.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_99_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_99_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.17.hw.gamma_pow", kind: "op", formula: "field.pow:51", operand_names: STAGE7_FIELD_EXPR_100_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_100_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.17.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_101_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_101_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.17.booleanity.gamma_pow", kind: "op", formula: "field.pow:52", operand_names: STAGE7_FIELD_EXPR_102_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_102_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.17.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_103_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_103_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.17.virtualization.gamma_pow", kind: "op", formula: "field.pow:53", operand_names: STAGE7_FIELD_EXPR_104_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_104_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.17.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_105_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_105_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.18.hw.gamma_pow", kind: "op", formula: "field.pow:54", operand_names: STAGE7_FIELD_EXPR_106_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_106_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.18.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_107_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_107_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.18.booleanity.gamma_pow", kind: "op", formula: "field.pow:55", operand_names: STAGE7_FIELD_EXPR_108_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_108_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.18.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_109_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_109_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.18.virtualization.gamma_pow", kind: "op", formula: "field.pow:56", operand_names: STAGE7_FIELD_EXPR_110_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_110_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.18.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_111_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_111_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.19.hw.gamma_pow", kind: "op", formula: "field.pow:57", operand_names: STAGE7_FIELD_EXPR_112_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_112_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.19.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_113_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_113_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.19.booleanity.gamma_pow", kind: "op", formula: "field.pow:58", operand_names: STAGE7_FIELD_EXPR_114_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_114_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.19.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_115_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_115_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.19.virtualization.gamma_pow", kind: "op", formula: "field.pow:59", operand_names: STAGE7_FIELD_EXPR_116_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_116_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.19.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_117_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_117_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.20.hw.gamma_pow", kind: "op", formula: "field.pow:60", operand_names: STAGE7_FIELD_EXPR_118_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_118_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.20.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_119_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_119_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.20.booleanity.gamma_pow", kind: "op", formula: "field.pow:61", operand_names: STAGE7_FIELD_EXPR_120_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_120_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.20.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_121_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_121_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.20.virtualization.gamma_pow", kind: "op", formula: "field.pow:62", operand_names: STAGE7_FIELD_EXPR_122_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_122_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.20.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_123_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_123_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.21.hw.gamma_pow", kind: "op", formula: "field.pow:63", operand_names: STAGE7_FIELD_EXPR_124_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_124_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.21.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_125_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_125_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.21.booleanity.gamma_pow", kind: "op", formula: "field.pow:64", operand_names: STAGE7_FIELD_EXPR_126_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_126_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.21.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_127_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_127_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.21.virtualization.gamma_pow", kind: "op", formula: "field.pow:65", operand_names: STAGE7_FIELD_EXPR_128_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_128_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.21.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_129_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_129_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.22.hw.gamma_pow", kind: "op", formula: "field.pow:66", operand_names: STAGE7_FIELD_EXPR_130_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_130_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.22.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_131_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_131_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.22.booleanity.gamma_pow", kind: "op", formula: "field.pow:67", operand_names: STAGE7_FIELD_EXPR_132_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_132_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.22.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_133_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_133_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.22.virtualization.gamma_pow", kind: "op", formula: "field.pow:68", operand_names: STAGE7_FIELD_EXPR_134_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_134_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.22.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_135_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_135_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.23.hw.gamma_pow", kind: "op", formula: "field.pow:69", operand_names: STAGE7_FIELD_EXPR_136_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_136_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.23.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_137_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_137_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.23.booleanity.gamma_pow", kind: "op", formula: "field.pow:70", operand_names: STAGE7_FIELD_EXPR_138_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_138_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.23.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_139_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_139_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.23.virtualization.gamma_pow", kind: "op", formula: "field.pow:71", operand_names: STAGE7_FIELD_EXPR_140_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_140_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.23.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_141_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_141_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.24.hw.gamma_pow", kind: "op", formula: "field.pow:72", operand_names: STAGE7_FIELD_EXPR_142_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_142_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.24.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_143_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_143_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.24.booleanity.gamma_pow", kind: "op", formula: "field.pow:73", operand_names: STAGE7_FIELD_EXPR_144_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_144_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.24.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_145_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_145_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.24.virtualization.gamma_pow", kind: "op", formula: "field.pow:74", operand_names: STAGE7_FIELD_EXPR_146_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_146_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.24.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_147_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_147_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.25.hw.gamma_pow", kind: "op", formula: "field.pow:75", operand_names: STAGE7_FIELD_EXPR_148_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_148_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.25.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_149_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_149_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.25.booleanity.gamma_pow", kind: "op", formula: "field.pow:76", operand_names: STAGE7_FIELD_EXPR_150_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_150_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.25.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_151_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_151_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.25.virtualization.gamma_pow", kind: "op", formula: "field.pow:77", operand_names: STAGE7_FIELD_EXPR_152_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_152_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.25.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_153_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_153_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.26.hw.gamma_pow", kind: "op", formula: "field.pow:78", operand_names: STAGE7_FIELD_EXPR_154_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_154_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.26.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_155_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_155_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.26.booleanity.gamma_pow", kind: "op", formula: "field.pow:79", operand_names: STAGE7_FIELD_EXPR_156_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_156_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.26.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_157_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_157_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.26.virtualization.gamma_pow", kind: "op", formula: "field.pow:80", operand_names: STAGE7_FIELD_EXPR_158_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_158_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.26.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_159_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_159_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.27.hw.gamma_pow", kind: "op", formula: "field.pow:81", operand_names: STAGE7_FIELD_EXPR_160_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_160_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.27.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_161_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_161_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.27.booleanity.gamma_pow", kind: "op", formula: "field.pow:82", operand_names: STAGE7_FIELD_EXPR_162_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_162_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.27.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_163_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_163_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.27.virtualization.gamma_pow", kind: "op", formula: "field.pow:83", operand_names: STAGE7_FIELD_EXPR_164_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_164_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.27.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_165_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_165_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.28.hw.gamma_pow", kind: "op", formula: "field.pow:84", operand_names: STAGE7_FIELD_EXPR_166_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_166_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.28.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_167_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_167_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.28.booleanity.gamma_pow", kind: "op", formula: "field.pow:85", operand_names: STAGE7_FIELD_EXPR_168_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_168_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.28.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_169_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_169_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.28.virtualization.gamma_pow", kind: "op", formula: "field.pow:86", operand_names: STAGE7_FIELD_EXPR_170_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_170_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.28.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_171_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_171_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.29.hw.gamma_pow", kind: "op", formula: "field.pow:87", operand_names: STAGE7_FIELD_EXPR_172_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_172_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.29.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_173_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_173_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.29.booleanity.gamma_pow", kind: "op", formula: "field.pow:88", operand_names: STAGE7_FIELD_EXPR_174_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_174_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.29.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_175_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_175_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.29.virtualization.gamma_pow", kind: "op", formula: "field.pow:89", operand_names: STAGE7_FIELD_EXPR_176_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_176_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.29.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_177_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_177_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.30.hw.gamma_pow", kind: "op", formula: "field.pow:90", operand_names: STAGE7_FIELD_EXPR_178_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_178_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.30.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_179_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_179_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.30.booleanity.gamma_pow", kind: "op", formula: "field.pow:91", operand_names: STAGE7_FIELD_EXPR_180_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_180_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.30.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_181_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_181_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.30.virtualization.gamma_pow", kind: "op", formula: "field.pow:92", operand_names: STAGE7_FIELD_EXPR_182_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_182_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.30.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_183_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_183_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.31.hw.gamma_pow", kind: "op", formula: "field.pow:93", operand_names: STAGE7_FIELD_EXPR_184_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_184_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.31.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_185_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_185_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.31.booleanity.gamma_pow", kind: "op", formula: "field.pow:94", operand_names: STAGE7_FIELD_EXPR_186_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_186_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.31.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_187_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_187_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.31.virtualization.gamma_pow", kind: "op", formula: "field.pow:95", operand_names: STAGE7_FIELD_EXPR_188_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_188_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.31.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_189_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_189_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.32.hw.gamma_pow", kind: "op", formula: "field.pow:96", operand_names: STAGE7_FIELD_EXPR_190_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_190_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.32.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_191_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_191_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.32.booleanity.gamma_pow", kind: "op", formula: "field.pow:97", operand_names: STAGE7_FIELD_EXPR_192_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_192_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.32.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_193_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_193_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.32.virtualization.gamma_pow", kind: "op", formula: "field.pow:98", operand_names: STAGE7_FIELD_EXPR_194_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_194_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.32.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_195_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_195_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.33.hw.gamma_pow", kind: "op", formula: "field.pow:99", operand_names: STAGE7_FIELD_EXPR_196_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_196_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.33.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_197_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_197_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.33.booleanity.gamma_pow", kind: "op", formula: "field.pow:100", operand_names: STAGE7_FIELD_EXPR_198_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_198_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.33.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_199_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_199_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.33.virtualization.gamma_pow", kind: "op", formula: "field.pow:101", operand_names: STAGE7_FIELD_EXPR_200_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_200_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.33.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_201_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_201_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.34.hw.gamma_pow", kind: "op", formula: "field.pow:102", operand_names: STAGE7_FIELD_EXPR_202_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_202_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.34.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_203_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_203_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.34.booleanity.gamma_pow", kind: "op", formula: "field.pow:103", operand_names: STAGE7_FIELD_EXPR_204_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_204_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.34.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_205_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_205_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.34.virtualization.gamma_pow", kind: "op", formula: "field.pow:104", operand_names: STAGE7_FIELD_EXPR_206_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_206_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.34.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_207_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_207_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.35.hw.gamma_pow", kind: "op", formula: "field.pow:105", operand_names: STAGE7_FIELD_EXPR_208_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_208_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.35.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_209_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_209_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.35.booleanity.gamma_pow", kind: "op", formula: "field.pow:106", operand_names: STAGE7_FIELD_EXPR_210_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_210_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.35.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_211_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_211_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.35.virtualization.gamma_pow", kind: "op", formula: "field.pow:107", operand_names: STAGE7_FIELD_EXPR_212_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_212_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.35.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_213_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_213_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.36.hw.gamma_pow", kind: "op", formula: "field.pow:108", operand_names: STAGE7_FIELD_EXPR_214_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_214_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.36.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_215_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_215_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.36.booleanity.gamma_pow", kind: "op", formula: "field.pow:109", operand_names: STAGE7_FIELD_EXPR_216_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_216_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.36.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_217_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_217_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.36.virtualization.gamma_pow", kind: "op", formula: "field.pow:110", operand_names: STAGE7_FIELD_EXPR_218_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_218_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.36.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_219_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_219_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.37.hw.gamma_pow", kind: "op", formula: "field.pow:111", operand_names: STAGE7_FIELD_EXPR_220_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_220_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.37.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_221_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_221_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.37.booleanity.gamma_pow", kind: "op", formula: "field.pow:112", operand_names: STAGE7_FIELD_EXPR_222_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_222_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.37.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_223_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_223_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.37.virtualization.gamma_pow", kind: "op", formula: "field.pow:113", operand_names: STAGE7_FIELD_EXPR_224_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_224_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.37.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_225_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_225_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.38.hw.gamma_pow", kind: "op", formula: "field.pow:114", operand_names: STAGE7_FIELD_EXPR_226_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_226_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.38.hw.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_227_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_227_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.38.booleanity.gamma_pow", kind: "op", formula: "field.pow:115", operand_names: STAGE7_FIELD_EXPR_228_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_228_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.38.booleanity.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_229_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_229_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.38.virtualization.gamma_pow", kind: "op", formula: "field.pow:116", operand_names: STAGE7_FIELD_EXPR_230_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_230_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim.38.virtualization.gamma_term", kind: "op", formula: "field.mul", operand_names: STAGE7_FIELD_EXPR_231_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_231_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial0", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_232_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_232_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial1", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_233_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_233_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial2", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_234_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_234_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial3", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_235_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_235_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial4", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_236_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_236_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial5", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_237_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_237_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial6", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_238_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_238_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial7", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_239_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_239_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial8", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_240_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_240_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial9", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_241_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_241_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial10", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_242_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_242_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial11", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_243_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_243_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial12", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_244_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_244_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial13", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_245_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_245_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial14", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_246_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_246_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial15", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_247_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_247_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial16", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_248_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_248_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial17", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_249_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_249_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial18", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_250_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_250_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial19", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_251_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_251_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial20", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_252_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_252_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial21", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_253_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_253_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial22", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_254_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_254_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial23", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_255_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_255_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial24", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_256_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_256_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial25", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_257_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_257_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial26", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_258_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_258_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial27", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_259_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_259_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial28", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_260_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_260_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial29", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_261_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_261_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial30", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_262_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_262_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial31", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_263_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_263_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial32", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_264_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_264_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial33", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_265_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_265_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial34", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_266_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_266_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial35", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_267_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_267_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial36", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_268_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_268_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial37", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_269_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_269_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial38", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_270_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_270_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial39", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_271_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_271_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial40", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_272_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_272_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial41", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_273_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_273_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial42", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_274_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_274_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial43", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_275_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_275_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial44", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_276_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_276_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial45", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_277_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_277_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial46", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_278_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_278_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial47", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_279_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_279_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial48", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_280_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_280_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial49", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_281_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_281_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial50", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_282_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_282_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial51", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_283_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_283_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial52", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_284_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_284_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial53", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_285_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_285_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial54", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_286_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_286_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial55", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_287_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_287_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial56", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_288_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_288_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial57", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_289_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_289_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial58", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_290_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_290_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial59", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_291_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_291_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial60", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_292_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_292_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial61", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_293_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_293_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial62", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_294_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_294_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial63", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_295_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_295_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial64", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_296_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_296_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial65", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_297_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_297_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial66", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_298_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_298_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial67", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_299_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_299_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial68", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_300_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_300_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial69", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_301_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_301_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial70", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_302_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_302_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial71", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_303_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_303_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial72", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_304_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_304_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial73", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_305_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_305_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial74", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_306_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_306_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial75", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_307_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_307_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial76", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_308_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_308_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial77", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_309_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_309_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial78", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_310_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_310_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial79", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_311_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_311_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial80", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_312_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_312_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial81", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_313_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_313_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial82", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_314_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_314_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial83", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_315_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_315_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial84", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_316_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_316_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial85", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_317_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_317_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial86", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_318_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_318_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial87", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_319_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_319_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial88", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_320_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_320_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial89", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_321_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_321_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial90", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_322_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_322_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial91", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_323_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_323_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial92", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_324_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_324_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial93", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_325_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_325_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial94", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_326_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_326_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial95", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_327_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_327_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial96", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_328_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_328_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial97", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_329_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_329_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial98", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_330_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_330_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial99", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_331_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_331_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial100", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_332_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_332_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial101", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_333_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_333_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial102", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_334_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_334_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial103", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_335_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_335_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial104", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_336_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_336_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial105", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_337_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_337_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial106", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_338_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_338_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial107", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_339_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_339_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial108", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_340_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_340_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial109", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_341_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_341_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial110", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_342_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_342_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial111", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_343_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_343_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial112", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_344_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_344_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial113", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_345_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_345_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial114", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_346_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_346_OPERANDS },
    Stage7FieldExprPlan { symbol: "stage7.hamming_weight_claim_reduction.claim_expr.partial115", kind: "op", formula: "field.add", operand_names: STAGE7_FIELD_EXPR_347_OPERAND_NAMES, operands: STAGE7_FIELD_EXPR_347_OPERANDS },
];
pub const STAGE7_KERNELS: &[Stage7KernelPlan] = &[

];

pub const STAGE7_SUMCHECK_CLAIM_0_INPUT_OPENINGS: &[&str] = &[
    "stage7.input.stage6.hamming_booleanity.HammingWeight",
    "stage7.input.stage6.booleanity.InstructionRa_0",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_0",
    "stage7.input.stage6.booleanity.InstructionRa_1",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_1",
    "stage7.input.stage6.booleanity.InstructionRa_2",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_2",
    "stage7.input.stage6.booleanity.InstructionRa_3",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_3",
    "stage7.input.stage6.booleanity.InstructionRa_4",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_4",
    "stage7.input.stage6.booleanity.InstructionRa_5",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_5",
    "stage7.input.stage6.booleanity.InstructionRa_6",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_6",
    "stage7.input.stage6.booleanity.InstructionRa_7",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_7",
    "stage7.input.stage6.booleanity.InstructionRa_8",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_8",
    "stage7.input.stage6.booleanity.InstructionRa_9",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_9",
    "stage7.input.stage6.booleanity.InstructionRa_10",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_10",
    "stage7.input.stage6.booleanity.InstructionRa_11",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_11",
    "stage7.input.stage6.booleanity.InstructionRa_12",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_12",
    "stage7.input.stage6.booleanity.InstructionRa_13",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_13",
    "stage7.input.stage6.booleanity.InstructionRa_14",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_14",
    "stage7.input.stage6.booleanity.InstructionRa_15",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_15",
    "stage7.input.stage6.booleanity.InstructionRa_16",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_16",
    "stage7.input.stage6.booleanity.InstructionRa_17",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_17",
    "stage7.input.stage6.booleanity.InstructionRa_18",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_18",
    "stage7.input.stage6.booleanity.InstructionRa_19",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_19",
    "stage7.input.stage6.booleanity.InstructionRa_20",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_20",
    "stage7.input.stage6.booleanity.InstructionRa_21",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_21",
    "stage7.input.stage6.booleanity.InstructionRa_22",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_22",
    "stage7.input.stage6.booleanity.InstructionRa_23",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_23",
    "stage7.input.stage6.booleanity.InstructionRa_24",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_24",
    "stage7.input.stage6.booleanity.InstructionRa_25",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_25",
    "stage7.input.stage6.booleanity.InstructionRa_26",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_26",
    "stage7.input.stage6.booleanity.InstructionRa_27",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_27",
    "stage7.input.stage6.booleanity.InstructionRa_28",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_28",
    "stage7.input.stage6.booleanity.InstructionRa_29",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_29",
    "stage7.input.stage6.booleanity.InstructionRa_30",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_30",
    "stage7.input.stage6.booleanity.InstructionRa_31",
    "stage7.input.stage6.instruction_ra_virtual.InstructionRa_31",
    "stage7.input.stage6.booleanity.BytecodeRa_0",
    "stage7.input.stage6.bytecode_read_raf.BytecodeRa_0",
    "stage7.input.stage6.booleanity.BytecodeRa_1",
    "stage7.input.stage6.bytecode_read_raf.BytecodeRa_1",
    "stage7.input.stage6.booleanity.BytecodeRa_2",
    "stage7.input.stage6.bytecode_read_raf.BytecodeRa_2",
    "stage7.input.stage6.booleanity.RamRa_0",
    "stage7.input.stage6.ram_ra_virtual.RamRa_0",
    "stage7.input.stage6.booleanity.RamRa_1",
    "stage7.input.stage6.ram_ra_virtual.RamRa_1",
    "stage7.input.stage6.booleanity.RamRa_2",
    "stage7.input.stage6.ram_ra_virtual.RamRa_2",
    "stage7.input.stage6.booleanity.RamRa_3",
    "stage7.input.stage6.ram_ra_virtual.RamRa_3",
];

pub const STAGE7_SUMCHECK_CLAIMS: &[Stage7SumcheckClaimPlan] = &[
    Stage7SumcheckClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.input", stage: "stage7", domain: "jolt.stage7_hamming_weight_claim_reduction_domain", num_rounds: 4, degree: 2, claim: "stage7.hamming_weight_claim_reduction.weighted_stage6_claims", kernel: None, relation: Some("jolt.stage7.hamming_weight_claim_reduction"), claim_value: "stage7.hamming_weight_claim_reduction.claim_expr.partial115", input_openings: STAGE7_SUMCHECK_CLAIM_0_INPUT_OPENINGS },
];
pub const STAGE7_SUMCHECK_BATCH_0_ORDERED_CLAIMS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.input",
];

pub const STAGE7_SUMCHECK_BATCH_0_CLAIM_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.input",
];

pub const STAGE7_SUMCHECK_BATCH_0_ROUND_SCHEDULE: &[usize] = &[
    4,
];

pub const STAGE7_SUMCHECK_BATCHES: &[Stage7SumcheckBatchPlan] = &[
    Stage7SumcheckBatchPlan { symbol: "stage7.batch", stage: "stage7", proof_slot: "stage7.sumcheck", policy: "jolt_core_stage7_aligned", count: 1, ordered_claims: STAGE7_SUMCHECK_BATCH_0_ORDERED_CLAIMS, claim_operands: STAGE7_SUMCHECK_BATCH_0_CLAIM_OPERANDS, claim_label: "sumcheck_claim", round_label: "sumcheck_poly", round_schedule: STAGE7_SUMCHECK_BATCH_0_ROUND_SCHEDULE },
];
pub const STAGE7_SUMCHECK_DRIVER_0_ROUND_SCHEDULE: &[usize] = &[
    4,
];

pub const STAGE7_SUMCHECK_DRIVERS: &[Stage7SumcheckDriverPlan] = &[
    Stage7SumcheckDriverPlan { symbol: "stage7.sumcheck", stage: "stage7", proof_slot: "stage7.sumcheck", kernel: None, relation: Some("jolt.stage7.batched"), batch: "stage7.batch", policy: "jolt_core_stage7_aligned", round_schedule: STAGE7_SUMCHECK_DRIVER_0_ROUND_SCHEDULE, claim_label: "sumcheck_claim", round_label: "sumcheck_poly", num_rounds: 4, degree: 2 },
];
pub const STAGE7_SUMCHECK_INSTANCE_RESULTS: &[Stage7SumcheckInstanceResultPlan] = &[
    Stage7SumcheckInstanceResultPlan { symbol: "stage7.hamming_weight_claim_reduction.instance", source: "stage7.sumcheck", claim: "stage7.hamming_weight_claim_reduction.input", relation: "jolt.stage7.hamming_weight_claim_reduction", index: 0, point_arity: 4, num_rounds: 4, round_offset: 0, point_order: "reverse", degree: 2 },
];

pub const STAGE7_SUMCHECK_EVALS: &[Stage7SumcheckEvalPlan] = &[
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_0", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_0", index: 0, oracle: "InstructionRa_0" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_1", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_1", index: 1, oracle: "InstructionRa_1" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_2", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_2", index: 2, oracle: "InstructionRa_2" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_3", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_3", index: 3, oracle: "InstructionRa_3" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_4", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_4", index: 4, oracle: "InstructionRa_4" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_5", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_5", index: 5, oracle: "InstructionRa_5" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_6", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_6", index: 6, oracle: "InstructionRa_6" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_7", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_7", index: 7, oracle: "InstructionRa_7" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_8", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_8", index: 8, oracle: "InstructionRa_8" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_9", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_9", index: 9, oracle: "InstructionRa_9" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_10", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_10", index: 10, oracle: "InstructionRa_10" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_11", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_11", index: 11, oracle: "InstructionRa_11" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_12", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_12", index: 12, oracle: "InstructionRa_12" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_13", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_13", index: 13, oracle: "InstructionRa_13" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_14", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_14", index: 14, oracle: "InstructionRa_14" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_15", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_15", index: 15, oracle: "InstructionRa_15" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_16", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_16", index: 16, oracle: "InstructionRa_16" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_17", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_17", index: 17, oracle: "InstructionRa_17" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_18", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_18", index: 18, oracle: "InstructionRa_18" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_19", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_19", index: 19, oracle: "InstructionRa_19" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_20", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_20", index: 20, oracle: "InstructionRa_20" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_21", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_21", index: 21, oracle: "InstructionRa_21" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_22", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_22", index: 22, oracle: "InstructionRa_22" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_23", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_23", index: 23, oracle: "InstructionRa_23" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_24", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_24", index: 24, oracle: "InstructionRa_24" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_25", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_25", index: 25, oracle: "InstructionRa_25" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_26", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_26", index: 26, oracle: "InstructionRa_26" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_27", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_27", index: 27, oracle: "InstructionRa_27" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_28", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_28", index: 28, oracle: "InstructionRa_28" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_29", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_29", index: 29, oracle: "InstructionRa_29" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_30", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_30", index: 30, oracle: "InstructionRa_30" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_31", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_31", index: 31, oracle: "InstructionRa_31" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.BytecodeRa_0", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.BytecodeRa_0", index: 32, oracle: "BytecodeRa_0" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.BytecodeRa_1", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.BytecodeRa_1", index: 33, oracle: "BytecodeRa_1" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.BytecodeRa_2", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.BytecodeRa_2", index: 34, oracle: "BytecodeRa_2" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.RamRa_0", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.RamRa_0", index: 35, oracle: "RamRa_0" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.RamRa_1", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.RamRa_1", index: 36, oracle: "RamRa_1" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.RamRa_2", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.RamRa_2", index: 37, oracle: "RamRa_2" },
    Stage7SumcheckEvalPlan { symbol: "stage7.hamming_weight_claim_reduction.eval.RamRa_3", source: "stage7.sumcheck", name: "stage7.hamming_weight_claim_reduction.eval.RamRa_3", index: 38, oracle: "RamRa_3" },
];

pub const STAGE7_POINT_ZEROS: &[Stage7PointZeroPlan] = &[

];

pub const STAGE7_POINT_SLICES: &[Stage7PointSlicePlan] = &[
    Stage7PointSlicePlan { symbol: "stage7.hamming_weight_claim_reduction.point.cycle", source: "stage7.input.stage6.booleanity.InstructionRa_0", offset: 4, length: 16, input: "stage7.input.stage6.booleanity.InstructionRa_0" },
];

pub const STAGE7_POINT_CONCAT_0_INPUTS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.instance",
    "stage7.hamming_weight_claim_reduction.point.cycle",
];

pub const STAGE7_POINT_CONCATS: &[Stage7PointConcatPlan] = &[
    Stage7PointConcatPlan { symbol: "stage7.hamming_weight_claim_reduction.point", layout: "address_chunk_then_cycle", arity: 20, inputs: STAGE7_POINT_CONCAT_0_INPUTS },
];
pub const STAGE7_OPENING_CLAIMS: &[Stage7OpeningClaimPlan] = &[
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_0", oracle: "InstructionRa_0", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_0" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_1", oracle: "InstructionRa_1", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_1" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_2", oracle: "InstructionRa_2", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_2" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_3", oracle: "InstructionRa_3", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_3" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_4", oracle: "InstructionRa_4", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_4" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_5", oracle: "InstructionRa_5", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_5" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_6", oracle: "InstructionRa_6", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_6" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_7", oracle: "InstructionRa_7", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_7" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_8", oracle: "InstructionRa_8", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_8" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_9", oracle: "InstructionRa_9", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_9" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_10", oracle: "InstructionRa_10", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_10" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_11", oracle: "InstructionRa_11", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_11" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_12", oracle: "InstructionRa_12", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_12" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_13", oracle: "InstructionRa_13", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_13" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_14", oracle: "InstructionRa_14", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_14" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_15", oracle: "InstructionRa_15", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_15" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_16", oracle: "InstructionRa_16", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_16" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_17", oracle: "InstructionRa_17", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_17" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_18", oracle: "InstructionRa_18", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_18" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_19", oracle: "InstructionRa_19", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_19" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_20", oracle: "InstructionRa_20", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_20" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_21", oracle: "InstructionRa_21", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_21" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_22", oracle: "InstructionRa_22", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_22" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_23", oracle: "InstructionRa_23", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_23" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_24", oracle: "InstructionRa_24", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_24" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_25", oracle: "InstructionRa_25", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_25" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_26", oracle: "InstructionRa_26", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_26" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_27", oracle: "InstructionRa_27", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_27" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_28", oracle: "InstructionRa_28", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_28" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_29", oracle: "InstructionRa_29", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_29" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_30", oracle: "InstructionRa_30", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_30" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_31", oracle: "InstructionRa_31", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_31" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.BytecodeRa_0", oracle: "BytecodeRa_0", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.BytecodeRa_0" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.BytecodeRa_1", oracle: "BytecodeRa_1", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.BytecodeRa_1" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.BytecodeRa_2", oracle: "BytecodeRa_2", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.BytecodeRa_2" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.RamRa_0", oracle: "RamRa_0", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.RamRa_0" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.RamRa_1", oracle: "RamRa_1", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.RamRa_1" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.RamRa_2", oracle: "RamRa_2", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.RamRa_2" },
    Stage7OpeningClaimPlan { symbol: "stage7.hamming_weight_claim_reduction.opening.RamRa_3", oracle: "RamRa_3", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed", point_source: "stage7.hamming_weight_claim_reduction.point", eval_source: "stage7.hamming_weight_claim_reduction.eval.RamRa_3" },
];

pub const STAGE7_OPENING_EQUALITIES: &[Stage7OpeningClaimEqualityPlan] = &[

];

pub const STAGE7_OPENING_BATCH_0_ORDERED_CLAIMS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_0",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_1",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_2",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_3",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_4",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_5",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_6",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_7",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_8",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_9",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_10",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_11",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_12",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_13",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_14",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_15",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_16",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_17",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_18",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_19",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_20",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_21",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_22",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_23",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_24",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_25",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_26",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_27",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_28",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_29",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_30",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_31",
    "stage7.hamming_weight_claim_reduction.opening.BytecodeRa_0",
    "stage7.hamming_weight_claim_reduction.opening.BytecodeRa_1",
    "stage7.hamming_weight_claim_reduction.opening.BytecodeRa_2",
    "stage7.hamming_weight_claim_reduction.opening.RamRa_0",
    "stage7.hamming_weight_claim_reduction.opening.RamRa_1",
    "stage7.hamming_weight_claim_reduction.opening.RamRa_2",
    "stage7.hamming_weight_claim_reduction.opening.RamRa_3",
];

pub const STAGE7_OPENING_BATCH_0_CLAIM_OPERANDS: &[&str] = &[
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_0",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_1",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_2",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_3",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_4",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_5",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_6",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_7",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_8",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_9",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_10",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_11",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_12",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_13",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_14",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_15",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_16",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_17",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_18",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_19",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_20",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_21",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_22",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_23",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_24",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_25",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_26",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_27",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_28",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_29",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_30",
    "stage7.hamming_weight_claim_reduction.opening.InstructionRa_31",
    "stage7.hamming_weight_claim_reduction.opening.BytecodeRa_0",
    "stage7.hamming_weight_claim_reduction.opening.BytecodeRa_1",
    "stage7.hamming_weight_claim_reduction.opening.BytecodeRa_2",
    "stage7.hamming_weight_claim_reduction.opening.RamRa_0",
    "stage7.hamming_weight_claim_reduction.opening.RamRa_1",
    "stage7.hamming_weight_claim_reduction.opening.RamRa_2",
    "stage7.hamming_weight_claim_reduction.opening.RamRa_3",
];

pub const STAGE7_OPENING_BATCHES: &[Stage7OpeningBatchPlan] = &[
    Stage7OpeningBatchPlan { symbol: "stage7.openings", stage: "stage7", proof_slot: "stage7.openings", policy: "jolt_stage7_output_order", count: 39, ordered_claims: STAGE7_OPENING_BATCH_0_ORDERED_CLAIMS, claim_operands: STAGE7_OPENING_BATCH_0_CLAIM_OPERANDS },
];
pub const STAGE7_PROGRAM: Stage7VerifierProgramPlan = Stage7CpuProgramPlan {
    role: "verifier",
    params: STAGE7_PARAMS,
    steps: STAGE7_PROGRAM_STEPS,
    transcript_squeezes: STAGE7_TRANSCRIPT_SQUEEZES,
    transcript_absorb_bytes: STAGE7_TRANSCRIPT_ABSORB_BYTES,
    opening_inputs: STAGE7_OPENING_INPUTS,
    field_constants: STAGE7_FIELD_CONSTANTS,
    field_exprs: STAGE7_FIELD_EXPRS,
    kernels: STAGE7_KERNELS,
    claims: STAGE7_SUMCHECK_CLAIMS,
    batches: STAGE7_SUMCHECK_BATCHES,
    drivers: STAGE7_SUMCHECK_DRIVERS,
    instance_results: STAGE7_SUMCHECK_INSTANCE_RESULTS,
    evals: STAGE7_SUMCHECK_EVALS,
    point_zeros: STAGE7_POINT_ZEROS,
    point_slices: STAGE7_POINT_SLICES,
    point_concats: STAGE7_POINT_CONCATS,
    opening_claims: STAGE7_OPENING_CLAIMS,
    opening_equalities: STAGE7_OPENING_EQUALITIES,
    opening_batches: STAGE7_OPENING_BATCHES,
};

pub fn verify_stage7<T>(
    proof: &Stage7Proof<Fr>,
    opening_inputs: &[Stage7OpeningInputValue<Fr>],
    transcript: &mut T,
) -> Result<Stage7ExecutionArtifacts<Fr>, VerifyStage7Error>
where
    T: Transcript<Challenge = Fr>,
{
    verify_stage7_with_program(&STAGE7_PROGRAM, proof, opening_inputs, transcript)
}

pub fn verify_stage7_with_program<T>(
    program: &'static Stage7VerifierProgramPlan,
    proof: &Stage7Proof<Fr>,
    opening_inputs: &[Stage7OpeningInputValue<Fr>],
    transcript: &mut T,
) -> Result<Stage7ExecutionArtifacts<Fr>, VerifyStage7Error>
where
    T: Transcript<Challenge = Fr>,
{
    if proof.sumchecks.len() != program.drivers.len() {
        return Err(VerifyStage7Error::UnexpectedProofCount {
            expected: program.drivers.len(),
            got: proof.sumchecks.len(),
        });
    }
    let mut store = Stage7ValueStore::with_opening_inputs(opening_inputs);
    store.seed_constants(program);
    let mut artifacts = Stage7ExecutionArtifacts::default();
    for step in program.steps {
        match step.kind {
            "transcript_squeeze" => {
                let squeeze =
                    find_squeeze(program, step.symbol).ok_or(VerifyStage7Error::MissingValue {
                        symbol: step.symbol,
                    })?;
                verify_stage7_squeeze(program, squeeze, &mut store, transcript, &mut artifacts)?;
            }
            "transcript_absorb_bytes" => {
                let absorb = find_absorb_bytes(program, step.symbol).ok_or(
                    VerifyStage7Error::MissingValue {
                        symbol: step.symbol,
                    },
                )?;
                absorb_stage7_bytes(absorb, transcript);
            }
            "sumcheck_driver" => {
                let driver =
                    find_driver(program, step.symbol).ok_or(VerifyStage7Error::MissingProof {
                        driver: step.symbol,
                    })?;
                verify_stage7_driver(
                    program,
                    driver,
                    proof,
                    &mut store,
                    transcript,
                    &mut artifacts,
                )?;
            }
            _ => {
                return Err(VerifyStage7Error::InvalidProof {
                    driver: step.symbol,
                    reason: "unsupported stage7 program step",
                });
            }
        }
    }
    artifacts
        .opening_batches
        .extend(program.opening_batches.iter());
    Ok(artifacts)
}

pub fn stage7_verifier_program() -> &'static Stage7VerifierProgramPlan {
    &STAGE7_PROGRAM
}

fn verify_stage7_squeeze<T>(
    program: &'static Stage7VerifierProgramPlan,
    squeeze: &'static Stage7TranscriptSqueezePlan,
    store: &mut Stage7ValueStore<Fr>,
    transcript: &mut T,
    artifacts: &mut Stage7ExecutionArtifacts<Fr>,
) -> Result<(), VerifyStage7Error>
where
    T: Transcript<Challenge = Fr>,
{
    let values = transcript.challenge_vector(squeeze.count);
    store.observe_challenge_vector(program, squeeze, &values)?;
    artifacts.challenge_vectors.push(Stage7ChallengeVector {
        symbol: squeeze.symbol,
        values,
    });
    Ok(())
}

fn absorb_stage7_bytes<T>(absorb: &'static Stage7TranscriptAbsorbBytesPlan, transcript: &mut T)
where
    T: Transcript<Challenge = Fr>,
{
    transcript.append(&LabelWithCount(
        absorb.label.as_bytes(),
        absorb.payload.len() as u64,
    ));
    transcript.append_bytes(absorb.payload.as_bytes());
}

fn verify_stage7_driver<T>(
    program: &'static Stage7VerifierProgramPlan,
    driver: &'static Stage7SumcheckDriverPlan,
    proof: &Stage7Proof<Fr>,
    store: &mut Stage7ValueStore<Fr>,
    transcript: &mut T,
    artifacts: &mut Stage7ExecutionArtifacts<Fr>,
) -> Result<(), VerifyStage7Error>
where
    T: Transcript<Challenge = Fr>,
{
    let proof = proof
        .sumchecks
        .get(artifacts.sumchecks.len())
        .ok_or(VerifyStage7Error::MissingProof {
            driver: driver.symbol,
        })?;
    let output = match driver.relation {
        Some("jolt.stage7.batched") => {
            verify_batched_stage7(program, driver, proof, store, transcript)?
        }
        Some(relation) => return Err(VerifyStage7Error::UnsupportedRelation { relation }),
        None => return Err(VerifyStage7Error::UnsupportedRelation { relation: "<missing>" }),
    };
    artifacts.sumchecks.push(output);
    Ok(())
}

fn verify_batched_stage7<T>(
    program: &'static Stage7VerifierProgramPlan,
    driver: &'static Stage7SumcheckDriverPlan,
    proof: &Stage7SumcheckOutput<Fr>,
    store: &mut Stage7ValueStore<Fr>,
    transcript: &mut T,
) -> Result<Stage7SumcheckOutput<Fr>, VerifyStage7Error>
where
    T: Transcript<Challenge = Fr>,
{
    if proof.driver != driver.symbol {
        return Err(VerifyStage7Error::InvalidProof {
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
        .map_err(|error| VerifyStage7Error::Sumcheck {
            driver: driver.symbol,
            error,
        })?;
    if !proof.point.is_empty() && proof.point != output.point {
        return Err(VerifyStage7Error::InvalidProof {
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
        return Err(VerifyStage7Error::InvalidProof {
            driver: driver.symbol,
            reason: "batched output claim mismatch",
        });
    }
    let verified = Stage7SumcheckOutput {
        driver: driver.symbol,
        point: output.point,
        evals: proof.evals.clone(),
        proof: proof.proof.clone(),
    };
    store.observe_sumcheck_output(program, &verified)?;
    append_opening_claims(program, store, transcript, &verified.evals)?;
    Ok(verified)
}

impl<F: Field> Stage7ValueStore<F> {
    fn with_opening_inputs(inputs: &[Stage7OpeningInputValue<F>]) -> Self {
        let mut store = Self::default();
        for input in inputs {
            store.insert_scalar(input.symbol, input.eval);
            store.insert_point(input.symbol, input.point.clone());
        }
        store
    }

    fn seed_constants(&mut self, program: &'static Stage7VerifierProgramPlan) {
        for constant in program.field_constants {
            self.insert_scalar(constant.symbol, F::from_u64(constant.value as u64));
        }
        for zero in program.point_zeros {
            self.insert_point(zero.symbol, vec![F::from_u64(0); zero.arity]);
        }
    }

    fn observe_challenge_vector(
        &mut self,
        program: &'static Stage7VerifierProgramPlan,
        plan: &'static Stage7TranscriptSqueezePlan,
        values: &[F],
    ) -> Result<(), VerifyStage7Error> {
        self.insert_point(plan.symbol, values.to_vec());
        if matches!(plan.kind, "challenge_scalar" | "scalar") {
            if values.len() != 1 {
                return Err(VerifyStage7Error::InvalidInputLength {
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
        program: &'static Stage7VerifierProgramPlan,
        output: &Stage7SumcheckOutput<F>,
    ) -> Result<(), VerifyStage7Error> {
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
                .ok_or(VerifyStage7Error::InvalidInputLength {
                    input: instance.symbol,
                    expected: end,
                    actual: output.point.len(),
                })?
                .to_vec();
            match instance.point_order {
                "as_is" => {}
                "reverse" => point.reverse(),
                "bytecode_read_raf" => point = normalize_bytecode_read_raf_point(program, &point)?,
                "stage7_booleanity" => {}
                "instruction_read_raf" => point = normalize_instruction_read_raf_point(&point)?,
                _ => {
                    return Err(VerifyStage7Error::InvalidProof {
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
                .ok_or(VerifyStage7Error::MissingValue {
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
        program: &'static Stage7VerifierProgramPlan,
        claim: &Stage7SumcheckClaimPlan,
    ) -> Result<F, VerifyStage7Error> {
        self.evaluate_available_field_exprs(program)?;
        self.scalar(claim.claim_value)
    }

    fn batch_claim_values(
        &mut self,
        program: &'static Stage7VerifierProgramPlan,
        batch: &Stage7SumcheckBatchPlan,
    ) -> Result<Vec<F>, VerifyStage7Error> {
        batch
            .claim_operands
            .iter()
            .map(|symbol| {
                let claim = find_claim(program, symbol).ok_or(VerifyStage7Error::MissingClaim {
                    batch: batch.symbol,
                    claim: symbol,
                })?;
                self.claim_value(program, claim)
            })
            .collect()
    }

    fn evaluate_available_points(
        &mut self,
        program: &'static Stage7VerifierProgramPlan,
    ) -> Result<(), VerifyStage7Error> {
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
                    .ok_or(VerifyStage7Error::InvalidInputLength {
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
                    return Err(VerifyStage7Error::InvalidInputLength {
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
        program: &'static Stage7VerifierProgramPlan,
    ) -> Result<(), VerifyStage7Error> {
        loop {
            let mut progress = 0usize;
            for expr in program.field_exprs {
                if self.try_scalar(expr.symbol).is_some() {
                    continue;
                }
                let Some(operands) = self.try_expr_operands(expr) else { continue };
                self.insert_scalar(expr.symbol, evaluate_stage7_field_expr(expr, &operands)?);
                progress += 1;
            }
            if progress == 0 {
                return Ok(());
            }
        }
    }

    fn verify_opening_equalities(
        &self,
        program: &'static Stage7VerifierProgramPlan,
    ) -> Result<(), VerifyStage7Error> {
        for equality in program.opening_equalities {
            match equality.mode {
                "point_and_eval" => {
                    if self.point(equality.lhs)? != self.point(equality.rhs)?
                        || self.scalar(equality.lhs)? != self.scalar(equality.rhs)?
                    {
                        return Err(VerifyStage7Error::InvalidProof {
                            driver: equality.symbol,
                            reason: "opening claim equality failed",
                        });
                    }
                }
                _ => {
                    return Err(VerifyStage7Error::InvalidProof {
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

    fn scalar(&self, symbol: &'static str) -> Result<F, VerifyStage7Error> {
        self.try_scalar(symbol)
            .ok_or(VerifyStage7Error::MissingValue { symbol })
    }

    fn try_scalar(&self, symbol: &str) -> Option<F> {
        self.scalars
            .iter()
            .find(|(name, _)| *name == symbol)
            .map(|(_, value)| *value)
    }

    fn point(&self, symbol: &'static str) -> Result<&[F], VerifyStage7Error> {
        self.try_point(symbol)
            .ok_or(VerifyStage7Error::MissingValue { symbol })
    }

    fn try_point(&self, symbol: &str) -> Option<&[F]> {
        self.points
            .iter()
            .find(|(name, _)| *name == symbol)
            .map(|(_, point)| point.as_slice())
    }

    fn try_expr_operands(&self, expr: &Stage7FieldExprPlan) -> Option<Vec<F>> {
        expr.operands
            .iter()
            .map(|operand| self.try_scalar(operand))
            .collect()
    }

    fn try_concat_point(&self, concat: &Stage7PointConcatPlan) -> Option<Vec<F>> {
        let mut point = Vec::with_capacity(concat.arity);
        for input in concat.inputs {
            point.extend_from_slice(self.try_point(input)?);
        }
        Some(point)
    }
}

fn evaluate_stage7_field_expr<F: Field>(
    expr: &Stage7FieldExprPlan,
    operands: &[F],
) -> Result<F, VerifyStage7Error> {
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
                    VerifyStage7Error::UnsupportedFieldExpr {
                        symbol: expr.symbol,
                        formula,
                    }
                })?;
                return Ok(pow_field(operands[0], exponent));
            }
            Err(VerifyStage7Error::UnsupportedFieldExpr {
                symbol: expr.symbol,
                formula,
            })
        }
    }
}

fn expected_batched_output_claim(
    program: &'static Stage7VerifierProgramPlan,
    driver: &'static Stage7SumcheckDriverPlan,
    store: &Stage7ValueStore<Fr>,
    evals: &[Stage7NamedEval<Fr>],
    point: &[Fr],
    batching_coeffs: &[Fr],
) -> Result<Fr, VerifyStage7Error> {
    let batch = find_batch(program, driver.batch)?;
    let claims = batch_claims(program, batch)?;
    let mut expected = Fr::from_u64(0);
    for (claim, coefficient) in claims.iter().zip(batching_coeffs) {
        let instance = program
            .instance_results
            .iter()
            .find(|instance| instance.claim == claim.symbol && instance.source == driver.symbol)
            .ok_or(VerifyStage7Error::MissingClaim {
                batch: batch.symbol,
                claim: claim.symbol,
            })?;
        let local_point = point
            .get(instance.round_offset..instance.round_offset + instance.num_rounds)
            .ok_or(VerifyStage7Error::InvalidInputLength {
                input: instance.symbol,
                expected: instance.round_offset + instance.num_rounds,
                actual: point.len(),
            })?;
        let value = match claim.relation {
            Some("jolt.stage7.hamming_weight_claim_reduction") => {
                expected_hamming_weight_claim_reduction(program, driver, store, evals, local_point)?
            }
            Some(relation) => return Err(VerifyStage7Error::UnsupportedRelation { relation }),
            None => return Err(VerifyStage7Error::UnsupportedRelation { relation: "<missing>" }),
        };
        expected += *coefficient * value;
    }
    Ok(expected)
}

fn expected_hamming_weight_claim_reduction(
    program: &'static Stage7VerifierProgramPlan,
    driver: &'static Stage7SumcheckDriverPlan,
    store: &Stage7ValueStore<Fr>,
    evals: &[Stage7NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage7Error> {
    let rho_rev = reverse_slice(local_point);
    let booleanity_point = store.point("stage7.input.stage6.booleanity.InstructionRa_0")?;
    let r_addr_bool =
        booleanity_point
            .get(..local_point.len())
            .ok_or(VerifyStage7Error::InvalidInputLength {
                input: "stage7.input.stage6.booleanity.InstructionRa_0",
                expected: local_point.len(),
                actual: booleanity_point.len(),
            })?;
    let eq_bool = EqPolynomial::<Fr>::mle(&rho_rev, r_addr_bool);
    let gamma = store.scalar("stage7.hamming_weight_claim_reduction.gamma")?;
    let mut gamma_power = Fr::from_u64(1);
    let mut expected = Fr::from_u64(0);
    let mut eval_plans = program
        .evals
        .iter()
        .filter(|eval| eval.source == driver.symbol)
        .collect::<Vec<_>>();
    eval_plans.sort_by_key(|eval| eval.index);
    for eval_plan in eval_plans {
        let g_i = eval_by_name(evals, eval_plan.name)?;
        let virt_point =
            stage7_virtualization_point(store, eval_plan.oracle, local_point.len())?;
        let eq_virt = EqPolynomial::<Fr>::mle(&rho_rev, virt_point);
        expected += g_i * (gamma_power + gamma_power * gamma * eq_bool
            + gamma_power * gamma.square() * eq_virt);
        gamma_power *= gamma;
        gamma_power *= gamma;
        gamma_power *= gamma;
    }
    Ok(expected)
}

fn stage7_virtualization_point<'a>(
    store: &'a Stage7ValueStore<Fr>,
    oracle: &str,
    log_k_chunk: usize,
) -> Result<&'a [Fr], VerifyStage7Error> {
    let symbol = if oracle.starts_with("InstructionRa_") {
        format!("stage7.input.stage6.instruction_ra_virtual.{oracle}")
    } else if oracle.starts_with("BytecodeRa_") {
        format!("stage7.input.stage6.bytecode_read_raf.{oracle}")
    } else if oracle.starts_with("RamRa_") {
        format!("stage7.input.stage6.ram_ra_virtual.{oracle}")
    } else {
        return Err(VerifyStage7Error::MissingValue {
            symbol: "stage7.hamming_weight_claim_reduction.oracle",
        });
    };
    let point = store.try_point(&symbol).ok_or(VerifyStage7Error::MissingValue {
        symbol: "stage7.hamming_weight_claim_reduction.virtualization_point",
    })?;
    point
        .get(..log_k_chunk)
        .ok_or(VerifyStage7Error::InvalidInputLength {
            input: "stage7.hamming_weight_claim_reduction.virtualization_point",
            expected: log_k_chunk,
            actual: point.len(),
        })
}

fn expected_bytecode_read_raf(
    program: &'static Stage7VerifierProgramPlan,
    data: &Stage7BytecodeReadRafData,
    store: &Stage7ValueStore<Fr>,
    evals: &[Stage7NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage7Error> {
    let log_t = stage7_trace_rounds(program)?;
    let opening_point = normalize_bytecode_read_raf_point(program, local_point)?;
    let log_k = opening_point.len() - log_t;
    let (r_address_prime, r_cycle_prime) = opening_point.split_at(log_k);

    let gamma = store.scalar("stage7.bytecode_read_raf.gamma")?;
    let gamma_powers = bytecode_gamma_powers(gamma);
    let int_eval = identity_polynomial_eval(r_address_prime);
    let stage_value_evals =
        bytecode_stage_value_evals(data, store, r_address_prime, r_cycle_prime.len())?;
    let stage_cycle_points = bytecode_stage_cycle_points(store, r_cycle_prime.len())?;
    let int_contrib = [
        gamma_powers[5] * int_eval,
        Fr::from_u64(0),
        gamma_powers[4] * int_eval,
        Fr::from_u64(0),
        Fr::from_u64(0),
    ];

    let mut val = Fr::from_u64(0);
    for index in 0..stage_value_evals.len() {
        val += (stage_value_evals[index] + int_contrib[index])
            * EqPolynomial::<Fr>::mle(&stage_cycle_points[index], r_cycle_prime)
            * gamma_powers[index];
    }

    let entry_bits = (0..log_k)
        .map(|index| {
            Fr::from_u64(((data.entry_bytecode_index >> (log_k - 1 - index)) & 1) as u64)
        })
        .collect::<Vec<_>>();
    let zero_cycle = vec![Fr::from_u64(0); r_cycle_prime.len()];
    let entry_contrib = gamma_powers[7]
        * EqPolynomial::<Fr>::mle(&entry_bits, r_address_prime)
        * EqPolynomial::<Fr>::mle(&zero_cycle, r_cycle_prime);
    let bytecode_ra =
        indexed_evals_by_prefix_any(evals, "stage7.bytecode_read_raf.eval.BytecodeRa_")?
            .into_iter()
            .product::<Fr>();
    Ok((val + entry_contrib) * bytecode_ra)
}

fn expected_booleanity(
    program: &'static Stage7VerifierProgramPlan,
    store: &Stage7ValueStore<Fr>,
    evals: &[Stage7NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage7Error> {
    let log_t = stage7_trace_rounds(program)?;
    let log_k_chunk =
        local_point
            .len()
            .checked_sub(log_t)
            .ok_or(VerifyStage7Error::InvalidInputLength {
                input: "stage7.booleanity.point",
                expected: log_t,
                actual: local_point.len(),
            })?;
    let stage5_point = store.point("stage7.input.stage5.instruction_read_raf.InstructionRa_0")?;
    let stage5_address_len =
        stage5_point
            .len()
            .checked_sub(log_t)
            .ok_or(VerifyStage7Error::InvalidInputLength {
                input: "stage7.input.stage5.instruction_read_raf.InstructionRa_0",
                expected: log_t,
                actual: stage5_point.len(),
            })?;
    if stage5_address_len < log_k_chunk {
        return Err(VerifyStage7Error::InvalidInputLength {
            input: "stage7.input.stage5.instruction_read_raf.InstructionRa_0",
            expected: log_k_chunk + log_t,
            actual: stage5_point.len(),
        });
    }

    let mut stage5_addr = stage5_point[..stage5_address_len].to_vec();
    stage5_addr.reverse();
    let mut combined_r = stage5_addr[stage5_address_len - log_k_chunk..].to_vec();
    combined_r.extend(stage5_point[stage5_address_len..].iter().rev().copied());
    if combined_r.len() != local_point.len() {
        return Err(VerifyStage7Error::InvalidInputLength {
            input: "stage7.booleanity.combined_point",
            expected: local_point.len(),
            actual: combined_r.len(),
        });
    }
    let eq_eval = EqPolynomial::<Fr>::mle(local_point, &combined_r);

    let gamma = store.scalar("stage7.booleanity.gamma")?;
    let gamma_sq = gamma.square();
    let mut gamma_power = Fr::from_u64(1);
    let mut booleanity = Fr::from_u64(0);
    for ra in booleanity_evals(evals)? {
        booleanity += gamma_power * (ra.square() - ra);
        gamma_power *= gamma_sq;
    }
    Ok(eq_eval * booleanity)
}

fn expected_hamming_booleanity(
    store: &Stage7ValueStore<Fr>,
    evals: &[Stage7NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage7Error> {
    let hamming = eval_by_name(evals, "stage7.hamming_booleanity.eval.HammingWeight")?;
    let lookup_output_point = reverse_slice(store.point("stage7.input.stage1.LookupOutput")?);
    if lookup_output_point.len() != local_point.len() {
        return Err(VerifyStage7Error::InvalidInputLength {
            input: "stage7.input.stage1.LookupOutput",
            expected: local_point.len(),
            actual: lookup_output_point.len(),
        });
    }
    let eq_eval = EqPolynomial::<Fr>::mle(local_point, &lookup_output_point);
    Ok((hamming.square() - hamming) * eq_eval)
}

fn expected_ram_ra_virtual(
    store: &Stage7ValueStore<Fr>,
    evals: &[Stage7NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage7Error> {
    let r_cycle_reduced = reverse_slice(local_point);
    let r_cycle = suffix_point(
        store.point("stage7.input.stage5.ram_ra_claim_reduction.RamRa")?,
        r_cycle_reduced.len(),
        "stage7.input.stage5.ram_ra_claim_reduction.RamRa",
    )?;
    let eq_eval = EqPolynomial::<Fr>::mle(r_cycle, &r_cycle_reduced);
    let ram_ra = indexed_evals_by_prefix_any(evals, "stage7.ram_ra_virtual.eval.RamRa_")?
        .into_iter()
        .product::<Fr>();
    Ok(eq_eval * ram_ra)
}

fn expected_instruction_ra_virtual(
    program: &'static Stage7VerifierProgramPlan,
    store: &Stage7ValueStore<Fr>,
    evals: &[Stage7NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage7Error> {
    let r_cycle_reduced = reverse_slice(local_point);
    let r_cycle = suffix_point(
        store.point("stage7.input.stage5.instruction_read_raf.InstructionRa_0")?,
        r_cycle_reduced.len(),
        "stage7.input.stage5.instruction_read_raf.InstructionRa_0",
    )?;
    let eq_eval = EqPolynomial::<Fr>::mle(r_cycle, &r_cycle_reduced);
    let committed_ra =
        indexed_evals_by_prefix_any(evals, "stage7.instruction_ra_virtual.eval.InstructionRa_")?;
    let virtual_count = program
        .opening_inputs
        .iter()
        .filter(|input| {
            input
                .symbol
                .starts_with("stage7.input.stage5.instruction_read_raf.InstructionRa_")
        })
        .count();
    if virtual_count == 0 || committed_ra.len() % virtual_count != 0 {
        return Err(VerifyStage7Error::InvalidInputLength {
            input: "stage7.instruction_ra_virtual.eval.InstructionRa_",
            expected: virtual_count,
            actual: committed_ra.len(),
        });
    }
    let committed_per_virtual = committed_ra.len() / virtual_count;
    let gamma = store.scalar("stage7.instruction_ra_virtual.gamma")?;
    let mut gamma_power = Fr::from_u64(1);
    let mut value = Fr::from_u64(0);
    for chunk in committed_ra.chunks(committed_per_virtual) {
        value += gamma_power * chunk.iter().copied().product::<Fr>();
        gamma_power *= gamma;
    }
    Ok(eq_eval * value)
}

fn expected_inc_claim_reduction(
    store: &Stage7ValueStore<Fr>,
    evals: &[Stage7NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage7Error> {
    let r_cycle_reduced = reverse_slice(local_point);
    let ram_inc_stage2 = suffix_point(
        store.point("stage7.input.stage2.ram_read_write.RamInc")?,
        r_cycle_reduced.len(),
        "stage7.input.stage2.ram_read_write.RamInc",
    )?;
    let ram_inc_stage4 = suffix_point(
        store.point("stage7.input.stage4.ram_val_check.RamInc")?,
        r_cycle_reduced.len(),
        "stage7.input.stage4.ram_val_check.RamInc",
    )?;
    let rd_inc_stage4 = suffix_point(
        store.point("stage7.input.stage4.registers_read_write.RdInc")?,
        r_cycle_reduced.len(),
        "stage7.input.stage4.registers_read_write.RdInc",
    )?;
    let rd_inc_stage5 = suffix_point(
        store.point("stage7.input.stage5.registers_val_evaluation.RdInc")?,
        r_cycle_reduced.len(),
        "stage7.input.stage5.registers_val_evaluation.RdInc",
    )?;
    let gamma = store.scalar("stage7.inc_claim_reduction.gamma")?;
    let eq_ram_combined = EqPolynomial::<Fr>::mle(ram_inc_stage2, &r_cycle_reduced)
        + gamma * EqPolynomial::<Fr>::mle(ram_inc_stage4, &r_cycle_reduced);
    let eq_rd_combined = EqPolynomial::<Fr>::mle(rd_inc_stage4, &r_cycle_reduced)
        + gamma * EqPolynomial::<Fr>::mle(rd_inc_stage5, &r_cycle_reduced);
    let ram_inc = eval_by_name(evals, "stage7.inc_claim_reduction.eval.RamInc")?;
    let rd_inc = eval_by_name(evals, "stage7.inc_claim_reduction.eval.RdInc")?;
    Ok(ram_inc * eq_ram_combined + gamma.square() * rd_inc * eq_rd_combined)
}

fn expected_instruction_read_raf(
    store: &Stage7ValueStore<Fr>,
    evals: &[Stage7NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage7Error> {
    const LOG_K: usize = 128;
    const XLEN: usize = 64;

    if local_point.len() < LOG_K {
        return Err(VerifyStage7Error::InvalidInputLength {
            input: "stage7.instruction_read_raf.point",
            expected: LOG_K,
            actual: local_point.len(),
        });
    }

    let (r_address_prime, r_cycle) = local_point.split_at(LOG_K);
    let r_cycle_prime = reverse_slice(r_cycle);
    let r_reduction = store.point("stage7.input.stage2.instruction.LookupOutput")?;
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
        "stage7.instruction_read_raf.eval.LookupTableFlag_",
        table_values.len(),
    )?;
    let val_claim = table_values
        .into_iter()
        .zip(table_flag_claims)
        .map(|(table_value, flag_claim)| table_value * flag_claim)
        .sum::<Fr>();

    let ra_claim = indexed_evals_by_prefix_any(
        evals,
        "stage7.instruction_read_raf.eval.InstructionRa_",
    )?
    .into_iter()
    .product::<Fr>();
    let raf_flag_claim = eval_by_name(
        evals,
        "stage7.instruction_read_raf.eval.InstructionRafFlag",
    )?;
    let gamma = store.scalar("stage7.instruction_read_raf.gamma")?;

    let raf_claim = (Fr::from_u64(1) - raf_flag_claim)
        * (left_operand_eval + gamma * right_operand_eval)
        + raf_flag_claim * gamma * identity_poly_eval;
    Ok(eq_eval_r_reduction * ra_claim * (val_claim + gamma * raf_claim))
}

fn expected_ram_ra_claim_reduction(
    store: &Stage7ValueStore<Fr>,
    evals: &[Stage7NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage7Error> {
    let r_cycle_reduced = reverse_slice(local_point);
    let r_cycle_raf = suffix_point(
        store.point("stage7.input.stage2.ram_raf.RamRa")?,
        r_cycle_reduced.len(),
        "stage7.input.stage2.ram_raf.RamRa",
    )?;
    let r_cycle_rw = suffix_point(
        store.point("stage7.input.stage2.ram_read_write.RamRa")?,
        r_cycle_reduced.len(),
        "stage7.input.stage2.ram_read_write.RamRa",
    )?;
    let r_cycle_val = suffix_point(
        store.point("stage7.input.stage4.ram_val_check.RamRa")?,
        r_cycle_reduced.len(),
        "stage7.input.stage4.ram_val_check.RamRa",
    )?;
    let gamma = store.scalar("stage7.ram_ra_claim_reduction.gamma")?;
    let eq_combined = EqPolynomial::<Fr>::mle(r_cycle_raf, &r_cycle_reduced)
        + gamma * EqPolynomial::<Fr>::mle(r_cycle_rw, &r_cycle_reduced)
        + gamma.square() * EqPolynomial::<Fr>::mle(r_cycle_val, &r_cycle_reduced);
    let ram_ra = eval_by_name(evals, "stage7.ram_ra_claim_reduction.eval.RamRa")?;
    Ok(eq_combined * ram_ra)
}

fn expected_registers_val_evaluation(
    store: &Stage7ValueStore<Fr>,
    evals: &[Stage7NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage7Error> {
    let registers_val_point = store.point("stage7.input.stage4.registers.RegistersVal")?;
    let r_cycle = suffix_point(
        registers_val_point,
        local_point.len(),
        "stage7.input.stage4.registers.RegistersVal",
    )?;
    let r_reduced = reverse_slice(local_point);
    let lt_eval = lt_polynomial_eval(&r_reduced, r_cycle);
    let rd_inc = eval_by_name(evals, "stage7.registers_val_evaluation.eval.RdInc")?;
    let rd_wa = eval_by_name(evals, "stage7.registers_val_evaluation.eval.RdWa")?;
    Ok(rd_inc * rd_wa * lt_eval)
}

fn append_opening_claims<T>(
    program: &'static Stage7VerifierProgramPlan,
    store: &mut Stage7ValueStore<Fr>,
    transcript: &mut T,
    evals: &[Stage7NamedEval<Fr>],
) -> Result<(), VerifyStage7Error>
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
            let claim = find_opening_claim(program, symbol).ok_or(VerifyStage7Error::MissingClaim {
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
    program: &'static Stage7VerifierProgramPlan,
    symbol: &str,
) -> Option<&'static Stage7TranscriptSqueezePlan> {
    program
        .transcript_squeezes
        .iter()
        .find(|squeeze| squeeze.symbol == symbol)
}

fn find_absorb_bytes(
    program: &'static Stage7VerifierProgramPlan,
    symbol: &str,
) -> Option<&'static Stage7TranscriptAbsorbBytesPlan> {
    program
        .transcript_absorb_bytes
        .iter()
        .find(|absorb| absorb.symbol == symbol)
}

fn find_driver(
    program: &'static Stage7VerifierProgramPlan,
    symbol: &str,
) -> Option<&'static Stage7SumcheckDriverPlan> {
    program
        .drivers
        .iter()
        .find(|driver| driver.symbol == symbol)
}

fn find_batch(
    program: &'static Stage7VerifierProgramPlan,
    symbol: &'static str,
) -> Result<&'static Stage7SumcheckBatchPlan, VerifyStage7Error> {
    program
        .batches
        .iter()
        .find(|batch| batch.symbol == symbol)
        .ok_or(VerifyStage7Error::MissingBatch {
            driver: symbol,
            batch: symbol,
        })
}

fn find_claim(
    program: &'static Stage7VerifierProgramPlan,
    symbol: &str,
) -> Option<&'static Stage7SumcheckClaimPlan> {
    program
        .claims
        .iter()
        .find(|claim| claim.symbol == symbol)
}

fn find_opening_claim(
    program: &'static Stage7VerifierProgramPlan,
    symbol: &str,
) -> Option<&'static Stage7OpeningClaimPlan> {
    program
        .opening_claims
        .iter()
        .find(|claim| claim.symbol == symbol)
}

fn batch_claims(
    program: &'static Stage7VerifierProgramPlan,
    batch: &Stage7SumcheckBatchPlan,
) -> Result<Vec<&'static Stage7SumcheckClaimPlan>, VerifyStage7Error> {
    batch
        .claim_operands
        .iter()
        .map(|symbol| {
            find_claim(program, symbol).ok_or(VerifyStage7Error::MissingClaim {
                batch: batch.symbol,
                claim: symbol,
            })
        })
        .collect()
}

fn stage7_trace_rounds(
    program: &'static Stage7VerifierProgramPlan,
) -> Result<usize, VerifyStage7Error> {
    program
        .instance_results
        .iter()
        .find(|instance| instance.relation == "jolt.stage7.hamming_booleanity")
        .map(|instance| instance.num_rounds)
        .ok_or(VerifyStage7Error::MissingValue {
            symbol: "stage7.hamming_booleanity.instance",
        })
}

fn bytecode_gamma_powers(gamma: Fr) -> [Fr; 8] {
    let mut powers = [Fr::from_u64(1); 8];
    for index in 1..powers.len() {
        powers[index] = powers[index - 1] * gamma;
    }
    powers
}

fn bytecode_stage_cycle_points(
    store: &Stage7ValueStore<Fr>,
    log_t: usize,
) -> Result<[Vec<Fr>; 5], VerifyStage7Error> {
    Ok([
        suffix_point(store.point("stage7.input.stage1.Imm")?, log_t, "stage7.input.stage1.Imm")?
            .to_vec(),
        suffix_point(
            store.point("stage7.input.stage2.OpFlagJump")?,
            log_t,
            "stage7.input.stage2.OpFlagJump",
        )?
        .to_vec(),
        suffix_point(
            store.point("stage7.input.stage3.spartan_shift.UnexpandedPC")?,
            log_t,
            "stage7.input.stage3.spartan_shift.UnexpandedPC",
        )?
        .to_vec(),
        suffix_point(
            store.point("stage7.input.stage4.Rs1Ra")?,
            log_t,
            "stage7.input.stage4.Rs1Ra",
        )?
        .to_vec(),
        suffix_point(
            store.point("stage7.input.stage5.registers_val_evaluation.RdWa")?,
            log_t,
            "stage7.input.stage5.registers_val_evaluation.RdWa",
        )?
        .to_vec(),
    ])
}

fn bytecode_stage_value_evals(
    data: &Stage7BytecodeReadRafData,
    store: &Stage7ValueStore<Fr>,
    r_address: &[Fr],
    log_t: usize,
) -> Result<[Fr; 5], VerifyStage7Error> {
    let expected_len =
        1usize
            .checked_shl(r_address.len() as u32)
            .ok_or(VerifyStage7Error::InvalidInputLength {
                input: "stage7.bytecode_read_raf.entries",
                expected: usize::BITS as usize,
                actual: r_address.len(),
            })?;
    if data.entries.len() != expected_len {
        return Err(VerifyStage7Error::InvalidInputLength {
            input: "stage7.bytecode_read_raf.entries",
            expected: expected_len,
            actual: data.entries.len(),
        });
    }
    if data.entry_bytecode_index >= expected_len {
        return Err(VerifyStage7Error::InvalidInputLength {
            input: "stage7.bytecode_read_raf.entry_bytecode_index",
            expected: expected_len,
            actual: data.entry_bytecode_index + 1,
        });
    }

    let stage1_gamma = store.scalar("stage7.bytecode_read_raf.stage1_gamma")?;
    let stage2_gamma = store.scalar("stage7.bytecode_read_raf.stage2_gamma")?;
    let stage3_gamma = store.scalar("stage7.bytecode_read_raf.stage3_gamma")?;
    let stage4_gamma = store.scalar("stage7.bytecode_read_raf.stage4_gamma")?;
    let stage5_gamma = store.scalar("stage7.bytecode_read_raf.stage5_gamma")?;
    let stage1_gamma_powers = field_powers(stage1_gamma, 16);
    let stage2_gamma_powers = field_powers(stage2_gamma, 4);
    let stage3_gamma_powers = field_powers(stage3_gamma, 9);
    let stage4_gamma_powers = field_powers(stage4_gamma, 3);
    let stage5_gamma_powers = field_powers(stage5_gamma, data.num_lookup_tables + 2);

    let stage4_register_point =
        register_prefix_point(store, "stage7.input.stage4.Rs1Ra", log_t)?;
    let stage5_register_point = register_prefix_point(
        store,
        "stage7.input.stage5.registers_val_evaluation.RdWa",
        log_t,
    )?;

    let mut evals = [Fr::from_u64(0); 5];
    for (index, entry) in data.entries.iter().enumerate() {
        let eq = indexed_boolean_eq(index, r_address);
        let values = bytecode_entry_stage_values(
            entry,
            data.num_lookup_tables,
            stage4_register_point,
            stage5_register_point,
            &stage1_gamma_powers,
            &stage2_gamma_powers,
            &stage3_gamma_powers,
            &stage4_gamma_powers,
            &stage5_gamma_powers,
        )?;
        for stage in 0..evals.len() {
            evals[stage] += eq * values[stage];
        }
    }
    Ok(evals)
}

fn bytecode_entry_stage_values(
    entry: &Stage7BytecodeEntry,
    num_lookup_tables: usize,
    stage4_register_point: &[Fr],
    stage5_register_point: &[Fr],
    stage1_gamma_powers: &[Fr],
    stage2_gamma_powers: &[Fr],
    stage3_gamma_powers: &[Fr],
    stage4_gamma_powers: &[Fr],
    stage5_gamma_powers: &[Fr],
) -> Result<[Fr; 5], VerifyStage7Error> {
    let mut stage1 = entry.address + entry.imm * stage1_gamma_powers[1];
    for (flag, gamma) in entry
        .circuit_flags
        .iter()
        .zip(stage1_gamma_powers.iter().skip(2))
    {
        if *flag {
            stage1 += *gamma;
        }
    }

    let mut stage2 = Fr::from_u64(0);
    if entry.circuit_flags[5] {
        stage2 += stage2_gamma_powers[0];
    }
    if entry.is_branch {
        stage2 += stage2_gamma_powers[1];
    }
    if entry.circuit_flags[6] {
        stage2 += stage2_gamma_powers[2];
    }
    if entry.circuit_flags[7] {
        stage2 += stage2_gamma_powers[3];
    }

    let mut stage3 = entry.imm + entry.address * stage3_gamma_powers[1];
    if entry.left_is_rs1 {
        stage3 += stage3_gamma_powers[2];
    }
    if entry.left_is_pc {
        stage3 += stage3_gamma_powers[3];
    }
    if entry.right_is_rs2 {
        stage3 += stage3_gamma_powers[4];
    }
    if entry.right_is_imm {
        stage3 += stage3_gamma_powers[5];
    }
    if entry.is_noop {
        stage3 += stage3_gamma_powers[6];
    }
    if entry.circuit_flags[7] {
        stage3 += stage3_gamma_powers[7];
    }
    if entry.circuit_flags[12] {
        stage3 += stage3_gamma_powers[8];
    }

    let stage4 = register_eq(entry.rd, stage4_register_point, "stage7.bytecode.entry.rd")?
        * stage4_gamma_powers[0]
        + register_eq(entry.rs1, stage4_register_point, "stage7.bytecode.entry.rs1")?
            * stage4_gamma_powers[1]
        + register_eq(entry.rs2, stage4_register_point, "stage7.bytecode.entry.rs2")?
            * stage4_gamma_powers[2];

    let mut stage5 =
        register_eq(entry.rd, stage5_register_point, "stage7.bytecode.entry.rd")?
            * stage5_gamma_powers[0];
    if !entry.is_interleaved {
        stage5 += stage5_gamma_powers[1];
    }
    if let Some(table) = entry.lookup_table {
        if table >= num_lookup_tables {
            return Err(VerifyStage7Error::InvalidInputLength {
                input: "stage7.bytecode.entry.lookup_table",
                expected: num_lookup_tables,
                actual: table + 1,
            });
        }
        stage5 += stage5_gamma_powers[2 + table];
    }

    Ok([stage1, stage2, stage3, stage4, stage5])
}

fn register_eq(
    index: Option<usize>,
    point: &[Fr],
    input: &'static str,
) -> Result<Fr, VerifyStage7Error> {
    let Some(index) = index else {
        return Ok(Fr::from_u64(0));
    };
    let register_count =
        1usize
            .checked_shl(point.len() as u32)
            .ok_or(VerifyStage7Error::InvalidInputLength {
                input,
                expected: usize::BITS as usize,
                actual: point.len(),
            })?;
    if index >= register_count {
        return Err(VerifyStage7Error::InvalidInputLength {
            input,
            expected: register_count,
            actual: index + 1,
        });
    }
    Ok(indexed_boolean_eq(index, point))
}

fn indexed_boolean_eq(index: usize, point: &[Fr]) -> Fr {
    let bits = (0..point.len())
        .map(|bit| Fr::from_u64(((index >> (point.len() - 1 - bit)) & 1) as u64))
        .collect::<Vec<_>>();
    EqPolynomial::<Fr>::mle(&bits, point)
}

fn field_powers(base: Fr, count: usize) -> Vec<Fr> {
    let mut powers = Vec::with_capacity(count);
    let mut power = Fr::from_u64(1);
    for _ in 0..count {
        powers.push(power);
        power *= base;
    }
    powers
}

fn normalize_bytecode_read_raf_point<F: Field>(
    program: &'static Stage7VerifierProgramPlan,
    point: &[F],
) -> Result<Vec<F>, VerifyStage7Error> {
    let log_t = stage7_trace_rounds(program)?;
    let log_k = point
        .len()
        .checked_sub(log_t)
        .ok_or(VerifyStage7Error::InvalidInputLength {
            input: "stage7.bytecode_read_raf.point",
            expected: log_t,
            actual: point.len(),
        })?;
    let mut normalized = point.to_vec();
    normalized[..log_k].reverse();
    normalized[log_k..].reverse();
    Ok(normalized)
}

fn prefix_point<'a>(
    point: &'a [Fr],
    length: usize,
    input: &'static str,
) -> Result<&'a [Fr], VerifyStage7Error> {
    point
        .get(..length)
        .filter(|prefix| prefix.len() == length)
        .ok_or(VerifyStage7Error::InvalidInputLength {
            input,
            expected: length,
            actual: point.len(),
        })
}

fn register_prefix_point<'a>(
    store: &'a Stage7ValueStore<Fr>,
    symbol: &'static str,
    log_t: usize,
) -> Result<&'a [Fr], VerifyStage7Error> {
    let point = store.point(symbol)?;
    let register_len = point
        .len()
        .checked_sub(log_t)
        .ok_or(VerifyStage7Error::InvalidInputLength {
            input: symbol,
            expected: log_t,
            actual: point.len(),
        })?;
    prefix_point(point, register_len, symbol)
}

fn booleanity_evals(evals: &[Stage7NamedEval<Fr>]) -> Result<Vec<Fr>, VerifyStage7Error> {
    let mut values = indexed_evals_by_prefix_any(
        evals,
        "stage7.booleanity.eval.InstructionRa_",
    )?;
    values.extend(indexed_evals_by_prefix_any(
        evals,
        "stage7.booleanity.eval.BytecodeRa_",
    )?);
    values.extend(indexed_evals_by_prefix_any(
        evals,
        "stage7.booleanity.eval.RamRa_",
    )?);
    Ok(values)
}

fn eval_by_name(evals: &[Stage7NamedEval<Fr>], name: &'static str) -> Result<Fr, VerifyStage7Error> {
    evals
        .iter()
        .find(|eval| eval.name == name)
        .map(|eval| eval.value)
        .ok_or(VerifyStage7Error::MissingValue { symbol: name })
}

fn indexed_evals_by_prefix(
    evals: &[Stage7NamedEval<Fr>],
    prefix: &'static str,
    count: usize,
) -> Result<Vec<Fr>, VerifyStage7Error> {
    let mut values = vec![None; count];
    for eval in evals {
        let Some(suffix) = eval.name.strip_prefix(prefix) else {
            continue;
        };
        let index = suffix.parse::<usize>().map_err(|_| {
            VerifyStage7Error::InvalidProof {
                driver: prefix,
                reason: "invalid indexed eval suffix",
            }
        })?;
        if index >= count || values[index].is_some() {
            return Err(VerifyStage7Error::InvalidProof {
                driver: prefix,
                reason: "invalid indexed eval",
            });
        }
        values[index] = Some(eval.value);
    }
    values
        .into_iter()
        .map(|value| value.ok_or(VerifyStage7Error::MissingValue { symbol: prefix }))
        .collect()
}

fn indexed_evals_by_prefix_any(
    evals: &[Stage7NamedEval<Fr>],
    prefix: &'static str,
) -> Result<Vec<Fr>, VerifyStage7Error> {
    let mut indexed_values = Vec::new();
    for eval in evals {
        let Some(suffix) = eval.name.strip_prefix(prefix) else {
            continue;
        };
        let index = suffix.parse::<usize>().map_err(|_| {
            VerifyStage7Error::InvalidProof {
                driver: prefix,
                reason: "invalid indexed eval suffix",
            }
        })?;
        if indexed_values
            .iter()
            .any(|(existing_index, _)| *existing_index == index)
        {
            return Err(VerifyStage7Error::InvalidProof {
                driver: prefix,
                reason: "duplicate indexed eval",
            });
        }
        indexed_values.push((index, eval.value));
    }
    if indexed_values.is_empty() {
        return Err(VerifyStage7Error::MissingValue { symbol: prefix });
    }
    indexed_values.sort_by_key(|(index, _)| *index);
    for (expected, (actual, _)) in indexed_values.iter().enumerate() {
        if *actual != expected {
            return Err(VerifyStage7Error::InvalidProof {
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
) -> Result<&'a [Fr], VerifyStage7Error> {
    point
        .get(point.len().saturating_sub(length)..)
        .filter(|suffix| suffix.len() == length)
        .ok_or(VerifyStage7Error::InvalidInputLength {
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

fn single_operand<F: Field>(symbol: &'static str, operands: &[F]) -> Result<F, VerifyStage7Error> {
    require_operand_count(symbol, 1, operands.len())?;
    Ok(operands[0])
}

fn require_operand_count(
    input: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), VerifyStage7Error> {
    if expected == actual {
        Ok(())
    } else {
        Err(VerifyStage7Error::InvalidInputLength {
            input,
            expected,
            actual,
        })
    }
}

fn reverse_slice(values: &[Fr]) -> Vec<Fr> {
    values.iter().rev().copied().collect()
}

fn normalize_instruction_read_raf_point<F: Field>(point: &[F]) -> Result<Vec<F>, VerifyStage7Error> {
    const LOG_K: usize = 128;
    if point.len() < LOG_K {
        return Err(VerifyStage7Error::InvalidInputLength {
            input: "stage7.instruction_read_raf.point",
            expected: LOG_K,
            actual: point.len(),
        });
    }
    let mut normalized = point.to_vec();
    normalized[LOG_K..].reverse();
    Ok(normalized)
}
