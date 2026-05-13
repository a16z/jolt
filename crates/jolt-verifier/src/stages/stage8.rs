#![allow(clippy::too_many_lines)]

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage8Params {
    pub field: &'static str,
    pub pcs: &'static str,
    pub transcript: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage8OpeningInputPlan {
    pub symbol: &'static str,
    pub source_stage: &'static str,
    pub source_claim: &'static str,
    pub oracle: &'static str,
    pub domain: &'static str,
    pub point_arity: usize,
    pub claim_kind: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage8OpeningClaimPlan {
    pub symbol: &'static str,
    pub oracle: &'static str,
    pub family: &'static str,
    pub domain: &'static str,
    pub point_arity: usize,
    pub point_source: &'static str,
    pub eval_source: &'static str,
    pub source_stage: &'static str,
    pub source_claim: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage8OpeningBatchPlan {
    pub symbol: &'static str,
    pub proof_slot: &'static str,
    pub policy: &'static str,
    pub count: usize,
    pub ordered_claims: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage8PcsProofPlan {
    pub symbol: &'static str,
    pub mode: &'static str,
    pub pcs: &'static str,
    pub proof_slot: &'static str,
    pub transcript_label: &'static str,
    pub batch: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage8EvaluationProgramPlan {
    pub role: &'static str,
    pub function: &'static str,
    pub params: Stage8Params,
    pub evaluation_point_source: Stage8OpeningInputPlan,
    pub opening_inputs: &'static [Stage8OpeningInputPlan],
    pub opening_claims: &'static [Stage8OpeningClaimPlan],
    pub opening_batch: Stage8OpeningBatchPlan,
    pub pcs_proof: Stage8PcsProofPlan,
}

pub const STAGE8_PARAMS: Stage8Params = Stage8Params { field: "bn254_fr", pcs: "dory", transcript: "blake2b_transcript" };

pub const STAGE8_EVALUATION_POINT_SOURCE: Stage8OpeningInputPlan = Stage8OpeningInputPlan { symbol: "stage8.evaluation.point_source", source_stage: "stage7", source_claim: "stage7.input.stage6.booleanity.InstructionRa_0", oracle: "InstructionRa_0", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" };

pub const STAGE8_OPENING_INPUTS: &[Stage8OpeningInputPlan] = &[
    Stage8OpeningInputPlan { symbol: "stage8.evaluation.point_source", source_stage: "stage7", source_claim: "stage7.input.stage6.booleanity.InstructionRa_0", oracle: "InstructionRa_0", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage8OpeningInputPlan { symbol: "stage8.input.stage6.RamInc", source_stage: "stage6", source_claim: "stage6.inc_claim_reduction.eval.RamInc", oracle: "RamInc", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage8OpeningInputPlan { symbol: "stage8.input.stage6.RdInc", source_stage: "stage6", source_claim: "stage6.inc_claim_reduction.eval.RdInc", oracle: "RdInc", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage8OpeningInputPlan { symbol: "stage8.input.stage7.InstructionRa_0", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_0", oracle: "InstructionRa_0", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage8OpeningInputPlan { symbol: "stage8.input.stage7.InstructionRa_1", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_1", oracle: "InstructionRa_1", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage8OpeningInputPlan { symbol: "stage8.input.stage7.InstructionRa_2", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_2", oracle: "InstructionRa_2", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage8OpeningInputPlan { symbol: "stage8.input.stage7.InstructionRa_3", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_3", oracle: "InstructionRa_3", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage8OpeningInputPlan { symbol: "stage8.input.stage7.InstructionRa_4", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_4", oracle: "InstructionRa_4", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage8OpeningInputPlan { symbol: "stage8.input.stage7.InstructionRa_5", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_5", oracle: "InstructionRa_5", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage8OpeningInputPlan { symbol: "stage8.input.stage7.InstructionRa_6", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_6", oracle: "InstructionRa_6", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage8OpeningInputPlan { symbol: "stage8.input.stage7.InstructionRa_7", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_7", oracle: "InstructionRa_7", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage8OpeningInputPlan { symbol: "stage8.input.stage7.InstructionRa_8", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_8", oracle: "InstructionRa_8", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage8OpeningInputPlan { symbol: "stage8.input.stage7.InstructionRa_9", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_9", oracle: "InstructionRa_9", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage8OpeningInputPlan { symbol: "stage8.input.stage7.InstructionRa_10", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_10", oracle: "InstructionRa_10", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage8OpeningInputPlan { symbol: "stage8.input.stage7.InstructionRa_11", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_11", oracle: "InstructionRa_11", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage8OpeningInputPlan { symbol: "stage8.input.stage7.InstructionRa_12", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_12", oracle: "InstructionRa_12", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage8OpeningInputPlan { symbol: "stage8.input.stage7.InstructionRa_13", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_13", oracle: "InstructionRa_13", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage8OpeningInputPlan { symbol: "stage8.input.stage7.InstructionRa_14", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_14", oracle: "InstructionRa_14", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage8OpeningInputPlan { symbol: "stage8.input.stage7.InstructionRa_15", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_15", oracle: "InstructionRa_15", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage8OpeningInputPlan { symbol: "stage8.input.stage7.InstructionRa_16", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_16", oracle: "InstructionRa_16", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage8OpeningInputPlan { symbol: "stage8.input.stage7.InstructionRa_17", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_17", oracle: "InstructionRa_17", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage8OpeningInputPlan { symbol: "stage8.input.stage7.InstructionRa_18", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_18", oracle: "InstructionRa_18", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage8OpeningInputPlan { symbol: "stage8.input.stage7.InstructionRa_19", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_19", oracle: "InstructionRa_19", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage8OpeningInputPlan { symbol: "stage8.input.stage7.InstructionRa_20", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_20", oracle: "InstructionRa_20", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage8OpeningInputPlan { symbol: "stage8.input.stage7.InstructionRa_21", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_21", oracle: "InstructionRa_21", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage8OpeningInputPlan { symbol: "stage8.input.stage7.InstructionRa_22", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_22", oracle: "InstructionRa_22", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage8OpeningInputPlan { symbol: "stage8.input.stage7.InstructionRa_23", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_23", oracle: "InstructionRa_23", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage8OpeningInputPlan { symbol: "stage8.input.stage7.InstructionRa_24", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_24", oracle: "InstructionRa_24", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage8OpeningInputPlan { symbol: "stage8.input.stage7.InstructionRa_25", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_25", oracle: "InstructionRa_25", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage8OpeningInputPlan { symbol: "stage8.input.stage7.InstructionRa_26", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_26", oracle: "InstructionRa_26", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage8OpeningInputPlan { symbol: "stage8.input.stage7.InstructionRa_27", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_27", oracle: "InstructionRa_27", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage8OpeningInputPlan { symbol: "stage8.input.stage7.InstructionRa_28", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_28", oracle: "InstructionRa_28", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage8OpeningInputPlan { symbol: "stage8.input.stage7.InstructionRa_29", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_29", oracle: "InstructionRa_29", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage8OpeningInputPlan { symbol: "stage8.input.stage7.InstructionRa_30", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_30", oracle: "InstructionRa_30", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage8OpeningInputPlan { symbol: "stage8.input.stage7.InstructionRa_31", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_31", oracle: "InstructionRa_31", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage8OpeningInputPlan { symbol: "stage8.input.stage7.BytecodeRa_0", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.BytecodeRa_0", oracle: "BytecodeRa_0", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage8OpeningInputPlan { symbol: "stage8.input.stage7.BytecodeRa_1", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.BytecodeRa_1", oracle: "BytecodeRa_1", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage8OpeningInputPlan { symbol: "stage8.input.stage7.BytecodeRa_2", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.BytecodeRa_2", oracle: "BytecodeRa_2", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage8OpeningInputPlan { symbol: "stage8.input.stage7.RamRa_0", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.RamRa_0", oracle: "RamRa_0", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage8OpeningInputPlan { symbol: "stage8.input.stage7.RamRa_1", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.RamRa_1", oracle: "RamRa_1", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage8OpeningInputPlan { symbol: "stage8.input.stage7.RamRa_2", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.RamRa_2", oracle: "RamRa_2", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
    Stage8OpeningInputPlan { symbol: "stage8.input.stage7.RamRa_3", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.RamRa_3", oracle: "RamRa_3", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: "committed" },
];

pub const STAGE8_OPENING_CLAIMS: &[Stage8OpeningClaimPlan] = &[
    Stage8OpeningClaimPlan { symbol: "stage8.evaluation.opening.RamInc", oracle: "RamInc", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: "stage8.input.stage6.RamInc", eval_source: "stage8.input.stage6.RamInc", source_stage: "stage6", source_claim: "stage6.inc_claim_reduction.eval.RamInc" },
    Stage8OpeningClaimPlan { symbol: "stage8.evaluation.opening.RdInc", oracle: "RdInc", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: "stage8.input.stage6.RdInc", eval_source: "stage8.input.stage6.RdInc", source_stage: "stage6", source_claim: "stage6.inc_claim_reduction.eval.RdInc" },
    Stage8OpeningClaimPlan { symbol: "stage8.evaluation.opening.InstructionRa_0", oracle: "InstructionRa_0", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: "stage8.input.stage7.InstructionRa_0", eval_source: "stage8.input.stage7.InstructionRa_0", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_0" },
    Stage8OpeningClaimPlan { symbol: "stage8.evaluation.opening.InstructionRa_1", oracle: "InstructionRa_1", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: "stage8.input.stage7.InstructionRa_1", eval_source: "stage8.input.stage7.InstructionRa_1", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_1" },
    Stage8OpeningClaimPlan { symbol: "stage8.evaluation.opening.InstructionRa_2", oracle: "InstructionRa_2", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: "stage8.input.stage7.InstructionRa_2", eval_source: "stage8.input.stage7.InstructionRa_2", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_2" },
    Stage8OpeningClaimPlan { symbol: "stage8.evaluation.opening.InstructionRa_3", oracle: "InstructionRa_3", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: "stage8.input.stage7.InstructionRa_3", eval_source: "stage8.input.stage7.InstructionRa_3", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_3" },
    Stage8OpeningClaimPlan { symbol: "stage8.evaluation.opening.InstructionRa_4", oracle: "InstructionRa_4", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: "stage8.input.stage7.InstructionRa_4", eval_source: "stage8.input.stage7.InstructionRa_4", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_4" },
    Stage8OpeningClaimPlan { symbol: "stage8.evaluation.opening.InstructionRa_5", oracle: "InstructionRa_5", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: "stage8.input.stage7.InstructionRa_5", eval_source: "stage8.input.stage7.InstructionRa_5", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_5" },
    Stage8OpeningClaimPlan { symbol: "stage8.evaluation.opening.InstructionRa_6", oracle: "InstructionRa_6", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: "stage8.input.stage7.InstructionRa_6", eval_source: "stage8.input.stage7.InstructionRa_6", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_6" },
    Stage8OpeningClaimPlan { symbol: "stage8.evaluation.opening.InstructionRa_7", oracle: "InstructionRa_7", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: "stage8.input.stage7.InstructionRa_7", eval_source: "stage8.input.stage7.InstructionRa_7", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_7" },
    Stage8OpeningClaimPlan { symbol: "stage8.evaluation.opening.InstructionRa_8", oracle: "InstructionRa_8", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: "stage8.input.stage7.InstructionRa_8", eval_source: "stage8.input.stage7.InstructionRa_8", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_8" },
    Stage8OpeningClaimPlan { symbol: "stage8.evaluation.opening.InstructionRa_9", oracle: "InstructionRa_9", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: "stage8.input.stage7.InstructionRa_9", eval_source: "stage8.input.stage7.InstructionRa_9", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_9" },
    Stage8OpeningClaimPlan { symbol: "stage8.evaluation.opening.InstructionRa_10", oracle: "InstructionRa_10", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: "stage8.input.stage7.InstructionRa_10", eval_source: "stage8.input.stage7.InstructionRa_10", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_10" },
    Stage8OpeningClaimPlan { symbol: "stage8.evaluation.opening.InstructionRa_11", oracle: "InstructionRa_11", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: "stage8.input.stage7.InstructionRa_11", eval_source: "stage8.input.stage7.InstructionRa_11", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_11" },
    Stage8OpeningClaimPlan { symbol: "stage8.evaluation.opening.InstructionRa_12", oracle: "InstructionRa_12", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: "stage8.input.stage7.InstructionRa_12", eval_source: "stage8.input.stage7.InstructionRa_12", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_12" },
    Stage8OpeningClaimPlan { symbol: "stage8.evaluation.opening.InstructionRa_13", oracle: "InstructionRa_13", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: "stage8.input.stage7.InstructionRa_13", eval_source: "stage8.input.stage7.InstructionRa_13", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_13" },
    Stage8OpeningClaimPlan { symbol: "stage8.evaluation.opening.InstructionRa_14", oracle: "InstructionRa_14", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: "stage8.input.stage7.InstructionRa_14", eval_source: "stage8.input.stage7.InstructionRa_14", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_14" },
    Stage8OpeningClaimPlan { symbol: "stage8.evaluation.opening.InstructionRa_15", oracle: "InstructionRa_15", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: "stage8.input.stage7.InstructionRa_15", eval_source: "stage8.input.stage7.InstructionRa_15", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_15" },
    Stage8OpeningClaimPlan { symbol: "stage8.evaluation.opening.InstructionRa_16", oracle: "InstructionRa_16", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: "stage8.input.stage7.InstructionRa_16", eval_source: "stage8.input.stage7.InstructionRa_16", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_16" },
    Stage8OpeningClaimPlan { symbol: "stage8.evaluation.opening.InstructionRa_17", oracle: "InstructionRa_17", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: "stage8.input.stage7.InstructionRa_17", eval_source: "stage8.input.stage7.InstructionRa_17", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_17" },
    Stage8OpeningClaimPlan { symbol: "stage8.evaluation.opening.InstructionRa_18", oracle: "InstructionRa_18", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: "stage8.input.stage7.InstructionRa_18", eval_source: "stage8.input.stage7.InstructionRa_18", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_18" },
    Stage8OpeningClaimPlan { symbol: "stage8.evaluation.opening.InstructionRa_19", oracle: "InstructionRa_19", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: "stage8.input.stage7.InstructionRa_19", eval_source: "stage8.input.stage7.InstructionRa_19", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_19" },
    Stage8OpeningClaimPlan { symbol: "stage8.evaluation.opening.InstructionRa_20", oracle: "InstructionRa_20", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: "stage8.input.stage7.InstructionRa_20", eval_source: "stage8.input.stage7.InstructionRa_20", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_20" },
    Stage8OpeningClaimPlan { symbol: "stage8.evaluation.opening.InstructionRa_21", oracle: "InstructionRa_21", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: "stage8.input.stage7.InstructionRa_21", eval_source: "stage8.input.stage7.InstructionRa_21", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_21" },
    Stage8OpeningClaimPlan { symbol: "stage8.evaluation.opening.InstructionRa_22", oracle: "InstructionRa_22", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: "stage8.input.stage7.InstructionRa_22", eval_source: "stage8.input.stage7.InstructionRa_22", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_22" },
    Stage8OpeningClaimPlan { symbol: "stage8.evaluation.opening.InstructionRa_23", oracle: "InstructionRa_23", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: "stage8.input.stage7.InstructionRa_23", eval_source: "stage8.input.stage7.InstructionRa_23", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_23" },
    Stage8OpeningClaimPlan { symbol: "stage8.evaluation.opening.InstructionRa_24", oracle: "InstructionRa_24", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: "stage8.input.stage7.InstructionRa_24", eval_source: "stage8.input.stage7.InstructionRa_24", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_24" },
    Stage8OpeningClaimPlan { symbol: "stage8.evaluation.opening.InstructionRa_25", oracle: "InstructionRa_25", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: "stage8.input.stage7.InstructionRa_25", eval_source: "stage8.input.stage7.InstructionRa_25", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_25" },
    Stage8OpeningClaimPlan { symbol: "stage8.evaluation.opening.InstructionRa_26", oracle: "InstructionRa_26", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: "stage8.input.stage7.InstructionRa_26", eval_source: "stage8.input.stage7.InstructionRa_26", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_26" },
    Stage8OpeningClaimPlan { symbol: "stage8.evaluation.opening.InstructionRa_27", oracle: "InstructionRa_27", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: "stage8.input.stage7.InstructionRa_27", eval_source: "stage8.input.stage7.InstructionRa_27", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_27" },
    Stage8OpeningClaimPlan { symbol: "stage8.evaluation.opening.InstructionRa_28", oracle: "InstructionRa_28", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: "stage8.input.stage7.InstructionRa_28", eval_source: "stage8.input.stage7.InstructionRa_28", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_28" },
    Stage8OpeningClaimPlan { symbol: "stage8.evaluation.opening.InstructionRa_29", oracle: "InstructionRa_29", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: "stage8.input.stage7.InstructionRa_29", eval_source: "stage8.input.stage7.InstructionRa_29", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_29" },
    Stage8OpeningClaimPlan { symbol: "stage8.evaluation.opening.InstructionRa_30", oracle: "InstructionRa_30", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: "stage8.input.stage7.InstructionRa_30", eval_source: "stage8.input.stage7.InstructionRa_30", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_30" },
    Stage8OpeningClaimPlan { symbol: "stage8.evaluation.opening.InstructionRa_31", oracle: "InstructionRa_31", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: "stage8.input.stage7.InstructionRa_31", eval_source: "stage8.input.stage7.InstructionRa_31", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_31" },
    Stage8OpeningClaimPlan { symbol: "stage8.evaluation.opening.BytecodeRa_0", oracle: "BytecodeRa_0", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: "stage8.input.stage7.BytecodeRa_0", eval_source: "stage8.input.stage7.BytecodeRa_0", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.BytecodeRa_0" },
    Stage8OpeningClaimPlan { symbol: "stage8.evaluation.opening.BytecodeRa_1", oracle: "BytecodeRa_1", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: "stage8.input.stage7.BytecodeRa_1", eval_source: "stage8.input.stage7.BytecodeRa_1", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.BytecodeRa_1" },
    Stage8OpeningClaimPlan { symbol: "stage8.evaluation.opening.BytecodeRa_2", oracle: "BytecodeRa_2", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: "stage8.input.stage7.BytecodeRa_2", eval_source: "stage8.input.stage7.BytecodeRa_2", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.BytecodeRa_2" },
    Stage8OpeningClaimPlan { symbol: "stage8.evaluation.opening.RamRa_0", oracle: "RamRa_0", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: "stage8.input.stage7.RamRa_0", eval_source: "stage8.input.stage7.RamRa_0", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.RamRa_0" },
    Stage8OpeningClaimPlan { symbol: "stage8.evaluation.opening.RamRa_1", oracle: "RamRa_1", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: "stage8.input.stage7.RamRa_1", eval_source: "stage8.input.stage7.RamRa_1", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.RamRa_1" },
    Stage8OpeningClaimPlan { symbol: "stage8.evaluation.opening.RamRa_2", oracle: "RamRa_2", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: "stage8.input.stage7.RamRa_2", eval_source: "stage8.input.stage7.RamRa_2", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.RamRa_2" },
    Stage8OpeningClaimPlan { symbol: "stage8.evaluation.opening.RamRa_3", oracle: "RamRa_3", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: "stage8.input.stage7.RamRa_3", eval_source: "stage8.input.stage7.RamRa_3", source_stage: "stage7", source_claim: "stage7.hamming_weight_claim_reduction.eval.RamRa_3" },
];

pub const STAGE8_OPENING_BATCH_ORDERED_CLAIMS: &[&str] = &["stage8.evaluation.opening.RamInc", "stage8.evaluation.opening.RdInc", "stage8.evaluation.opening.InstructionRa_0", "stage8.evaluation.opening.InstructionRa_1", "stage8.evaluation.opening.InstructionRa_2", "stage8.evaluation.opening.InstructionRa_3", "stage8.evaluation.opening.InstructionRa_4", "stage8.evaluation.opening.InstructionRa_5", "stage8.evaluation.opening.InstructionRa_6", "stage8.evaluation.opening.InstructionRa_7", "stage8.evaluation.opening.InstructionRa_8", "stage8.evaluation.opening.InstructionRa_9", "stage8.evaluation.opening.InstructionRa_10", "stage8.evaluation.opening.InstructionRa_11", "stage8.evaluation.opening.InstructionRa_12", "stage8.evaluation.opening.InstructionRa_13", "stage8.evaluation.opening.InstructionRa_14", "stage8.evaluation.opening.InstructionRa_15", "stage8.evaluation.opening.InstructionRa_16", "stage8.evaluation.opening.InstructionRa_17", "stage8.evaluation.opening.InstructionRa_18", "stage8.evaluation.opening.InstructionRa_19", "stage8.evaluation.opening.InstructionRa_20", "stage8.evaluation.opening.InstructionRa_21", "stage8.evaluation.opening.InstructionRa_22", "stage8.evaluation.opening.InstructionRa_23", "stage8.evaluation.opening.InstructionRa_24", "stage8.evaluation.opening.InstructionRa_25", "stage8.evaluation.opening.InstructionRa_26", "stage8.evaluation.opening.InstructionRa_27", "stage8.evaluation.opening.InstructionRa_28", "stage8.evaluation.opening.InstructionRa_29", "stage8.evaluation.opening.InstructionRa_30", "stage8.evaluation.opening.InstructionRa_31", "stage8.evaluation.opening.BytecodeRa_0", "stage8.evaluation.opening.BytecodeRa_1", "stage8.evaluation.opening.BytecodeRa_2", "stage8.evaluation.opening.RamRa_0", "stage8.evaluation.opening.RamRa_1", "stage8.evaluation.opening.RamRa_2", "stage8.evaluation.opening.RamRa_3"];

pub const STAGE8_OPENING_BATCH: Stage8OpeningBatchPlan = Stage8OpeningBatchPlan { symbol: "stage8.evaluation.openings", proof_slot: "stage8.evaluation", policy: "jolt_stage8_joint_rlc", count: 41, ordered_claims: STAGE8_OPENING_BATCH_ORDERED_CLAIMS };

pub const STAGE8_PCS_PROOF: Stage8PcsProofPlan = Stage8PcsProofPlan { symbol: "stage8.evaluation.proof", mode: "verify", pcs: "dory", proof_slot: "stage8.evaluation", transcript_label: "rlc_claims", batch: "stage8.evaluation.openings" };

pub const STAGE8_PROGRAM: Stage8EvaluationProgramPlan = Stage8EvaluationProgramPlan {
    role: "verifier",
    function: "jolt.stage8",
    params: STAGE8_PARAMS,
    evaluation_point_source: STAGE8_EVALUATION_POINT_SOURCE,
    opening_inputs: STAGE8_OPENING_INPUTS,
    opening_claims: STAGE8_OPENING_CLAIMS,
    opening_batch: STAGE8_OPENING_BATCH,
    pcs_proof: STAGE8_PCS_PROOF,
};
