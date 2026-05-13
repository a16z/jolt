#![allow(clippy::too_many_lines)]

pub use super::common::{ClaimKind as Stage8ClaimKind, PcsProofMode as Stage8PcsProofMode, SourceStage as Stage8SourceStage, StageParams as Stage8Params, TypedPlanSymbol};

pub enum Stage8OpeningInputTag {}
pub type Stage8OpeningInputSymbol = TypedPlanSymbol<Stage8OpeningInputTag>;
pub enum Stage8OpeningClaimTag {}
pub type Stage8OpeningClaimSymbol = TypedPlanSymbol<Stage8OpeningClaimTag>;
pub enum Stage8OpeningBatchTag {}
pub type Stage8OpeningBatchSymbol = TypedPlanSymbol<Stage8OpeningBatchTag>;
pub enum Stage8SourceClaimTag {}
pub type Stage8SourceClaim = TypedPlanSymbol<Stage8SourceClaimTag>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage8OpeningInputPlan {
    pub symbol: Stage8OpeningInputSymbol,
    pub source_stage: Stage8SourceStage,
    pub source_claim: Stage8SourceClaim,
    pub oracle: &'static str,
    pub domain: &'static str,
    pub point_arity: usize,
    pub claim_kind: Stage8ClaimKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage8OpeningClaimPlan {
    pub symbol: Stage8OpeningClaimSymbol,
    pub oracle: &'static str,
    pub family: &'static str,
    pub domain: &'static str,
    pub point_arity: usize,
    pub point_source: Stage8OpeningInputSymbol,
    pub eval_source: Stage8OpeningInputSymbol,
    pub source_stage: Stage8SourceStage,
    pub source_claim: Stage8SourceClaim,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage8OpeningBatchPlan {
    pub symbol: Stage8OpeningBatchSymbol,
    pub proof_slot: &'static str,
    pub policy: &'static str,
    pub count: usize,
    pub ordered_claims: &'static [Stage8OpeningClaimPlan],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage8PcsProofPlan {
    pub symbol: &'static str,
    pub mode: Stage8PcsProofMode,
    pub pcs: &'static str,
    pub proof_slot: &'static str,
    pub transcript_label: &'static str,
    pub batch: Stage8OpeningBatchSymbol,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage8EvaluationClaim<F> {
    pub oracle: &'static str,
    pub source_stage: Stage8SourceStage,
    pub value: F,
}

pub fn reverse_point<F: Copy>(point: &[F]) -> Vec<F> {
    point.iter().rev().copied().collect()
}

pub fn append_rlc_claims<F, T>(transcript: &mut T, claims: &[Stage8EvaluationClaim<F>])
where
    F: jolt_field::Field + jolt_transcript::AppendToTranscript,
    T: jolt_transcript::Transcript<Challenge = F>,
{
    transcript.append(&jolt_transcript::LabelWithCount(
        b"rlc_claims",
        claims.len() as u64,
    ));
    for claim in claims {
        jolt_transcript::AppendToTranscript::append_to_transcript(&claim.value, transcript);
    }
}

pub fn gamma_powers<F, T>(transcript: &mut T, count: usize) -> Vec<F>
where
    F: jolt_field::Field,
    T: jolt_transcript::Transcript<Challenge = F>,
{
    let gamma = transcript.challenge();
    let mut powers = Vec::with_capacity(count);
    let mut power = F::from_u64(1);
    for _ in 0..count {
        powers.push(power);
        power *= gamma;
    }
    powers
}

pub trait Stage8NamedEvalView<F> {
    fn name(&self) -> &'static str;
    fn value(&self) -> F;
}

pub trait Stage8SumcheckOutputView<F> {
    type Eval: Stage8NamedEvalView<F>;

    fn point(&self) -> &[F];
    fn evals(&self) -> &[Self::Eval];
}

pub fn stage7_claim_values<F, O>(
    program: &'static Stage8EvaluationProgramPlan,
    outputs: &[O],
) -> Option<(Vec<F>, std::collections::BTreeMap<Stage8SourceClaim, F>)>
where
    F: Copy,
    O: Stage8SumcheckOutputView<F>,
{
    let stage7_plans = program
        .opening_claims
        .iter()
        .filter(|plan| plan.source_stage == Stage8SourceStage::Stage7)
        .collect::<Vec<_>>();
    for output in outputs {
        let mut values = std::collections::BTreeMap::new();
        for plan in &stage7_plans {
            if let Some(eval) = output
                .evals()
                .iter()
                .find(|eval| eval.name() == plan.source_claim.as_str())
            {
                let _ = values.insert(plan.source_claim, eval.value());
            }
        }
        if values.len() == stage7_plans.len() {
            return Some((output.point().to_vec(), values));
        }
    }
    None
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage8EvaluationClaimError {
    pub stage: &'static str,
    pub eval: &'static str,
}

pub fn evaluation_claims<F, O>(
    program: &'static Stage8EvaluationProgramPlan,
    stage6_outputs: &[O],
    stage7_values: &std::collections::BTreeMap<Stage8SourceClaim, F>,
    lagrange_factor: F,
) -> Result<Vec<Stage8EvaluationClaim<F>>, Stage8EvaluationClaimError>
where
    F: Copy + std::ops::Mul<Output = F>,
    O: Stage8SumcheckOutputView<F>,
{
    let mut claims = Vec::with_capacity(program.opening_claims.len());
    for plan in program.opening_claims {
        let value = match plan.source_stage {
            Stage8SourceStage::Stage6 => {
                stage_eval(stage6_outputs, plan.source_stage.as_str(), plan.source_claim)?
                    * lagrange_factor
            }
            Stage8SourceStage::Stage7 => *stage7_values.get(&plan.source_claim).ok_or(
                Stage8EvaluationClaimError {
                    stage: plan.source_stage.as_str(),
                    eval: plan.source_claim.as_str(),
                },
            )?,
        };
        claims.push(Stage8EvaluationClaim {
            oracle: plan.oracle,
            source_stage: plan.source_stage,
            value,
        });
    }
    Ok(claims)
}

fn stage_eval<F, O>(
    outputs: &[O],
    stage: &'static str,
    eval_name: Stage8SourceClaim,
) -> Result<F, Stage8EvaluationClaimError>
where
    F: Copy,
    O: Stage8SumcheckOutputView<F>,
{
    for output in outputs {
        let eval = output
            .evals()
            .iter()
            .find(|eval| eval.name() == eval_name.as_str());
        if let Some(eval) = eval { return Ok(eval.value()); }
    }
    Err(Stage8EvaluationClaimError {
        stage,
        eval: eval_name.as_str(),
    })
}

pub trait Stage8OpeningInputView<F> {
    fn symbol(&self) -> &'static str;
    fn point(&self) -> &[F];
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage8EvaluationOpeningPointError {
    MissingStage7EvaluationPoint,
    InvalidPointLength {
        artifact: &'static str,
        expected: usize,
        actual: usize,
    },
}

pub fn stage7_evaluation_opening_point<F, I>(
    program: &'static Stage8EvaluationProgramPlan,
    address_point: &[F],
    stage7_openings: &[I],
) -> Result<(Vec<F>, usize), Stage8EvaluationOpeningPointError>
where
    F: Copy,
    I: Stage8OpeningInputView<F>,
{
    let cycle_source_symbol = program.evaluation_point_source.source_claim;
    let cycle_source = stage7_openings
        .iter()
        .find(|input| input.symbol() == cycle_source_symbol.as_str())
        .ok_or(Stage8EvaluationOpeningPointError::MissingStage7EvaluationPoint)?;
    let cycle_source_point = cycle_source.point();
    if cycle_source_point.len() < address_point.len() {
        return Err(Stage8EvaluationOpeningPointError::InvalidPointLength {
            artifact: cycle_source_symbol.as_str(),
            expected: address_point.len(),
            actual: cycle_source_point.len(),
        });
    }
    let cycle_len = cycle_source_point.len() - address_point.len();
    let mut point = Vec::with_capacity(cycle_source_point.len());
    point.extend_from_slice(address_point);
    point.extend_from_slice(&cycle_source_point[address_point.len()..]);
    Ok((point, cycle_len))
}

pub const STAGE8_PARAMS: Stage8Params = Stage8Params { field: "bn254_fr", pcs: "dory", transcript: "blake2b_transcript" };

pub const STAGE8_EVALUATION_POINT_SOURCE: Stage8OpeningInputPlan = Stage8OpeningInputPlan { symbol: Stage8OpeningInputSymbol::new("stage8.evaluation.point_source"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.input.stage6.booleanity.InstructionRa_0"), oracle: "InstructionRa_0", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: Stage8ClaimKind::Committed };

#[rustfmt::skip]
pub const STAGE8_OPENING_INPUTS: &[Stage8OpeningInputPlan] = &[
    Stage8OpeningInputPlan { symbol: Stage8OpeningInputSymbol::new("stage8.evaluation.point_source"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.input.stage6.booleanity.InstructionRa_0"), oracle: "InstructionRa_0", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: Stage8ClaimKind::Committed },
    Stage8OpeningInputPlan { symbol: Stage8OpeningInputSymbol::new("stage8.input.stage6.RamInc"), source_stage: Stage8SourceStage::Stage6, source_claim: Stage8SourceClaim::new("stage6.inc_claim_reduction.eval.RamInc"), oracle: "RamInc", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: Stage8ClaimKind::Committed },
    Stage8OpeningInputPlan { symbol: Stage8OpeningInputSymbol::new("stage8.input.stage6.RdInc"), source_stage: Stage8SourceStage::Stage6, source_claim: Stage8SourceClaim::new("stage6.inc_claim_reduction.eval.RdInc"), oracle: "RdInc", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: Stage8ClaimKind::Committed },
    Stage8OpeningInputPlan { symbol: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_0"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_0"), oracle: "InstructionRa_0", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: Stage8ClaimKind::Committed },
    Stage8OpeningInputPlan { symbol: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_1"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_1"), oracle: "InstructionRa_1", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: Stage8ClaimKind::Committed },
    Stage8OpeningInputPlan { symbol: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_2"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_2"), oracle: "InstructionRa_2", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: Stage8ClaimKind::Committed },
    Stage8OpeningInputPlan { symbol: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_3"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_3"), oracle: "InstructionRa_3", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: Stage8ClaimKind::Committed },
    Stage8OpeningInputPlan { symbol: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_4"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_4"), oracle: "InstructionRa_4", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: Stage8ClaimKind::Committed },
    Stage8OpeningInputPlan { symbol: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_5"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_5"), oracle: "InstructionRa_5", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: Stage8ClaimKind::Committed },
    Stage8OpeningInputPlan { symbol: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_6"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_6"), oracle: "InstructionRa_6", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: Stage8ClaimKind::Committed },
    Stage8OpeningInputPlan { symbol: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_7"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_7"), oracle: "InstructionRa_7", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: Stage8ClaimKind::Committed },
    Stage8OpeningInputPlan { symbol: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_8"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_8"), oracle: "InstructionRa_8", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: Stage8ClaimKind::Committed },
    Stage8OpeningInputPlan { symbol: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_9"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_9"), oracle: "InstructionRa_9", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: Stage8ClaimKind::Committed },
    Stage8OpeningInputPlan { symbol: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_10"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_10"), oracle: "InstructionRa_10", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: Stage8ClaimKind::Committed },
    Stage8OpeningInputPlan { symbol: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_11"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_11"), oracle: "InstructionRa_11", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: Stage8ClaimKind::Committed },
    Stage8OpeningInputPlan { symbol: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_12"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_12"), oracle: "InstructionRa_12", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: Stage8ClaimKind::Committed },
    Stage8OpeningInputPlan { symbol: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_13"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_13"), oracle: "InstructionRa_13", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: Stage8ClaimKind::Committed },
    Stage8OpeningInputPlan { symbol: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_14"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_14"), oracle: "InstructionRa_14", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: Stage8ClaimKind::Committed },
    Stage8OpeningInputPlan { symbol: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_15"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_15"), oracle: "InstructionRa_15", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: Stage8ClaimKind::Committed },
    Stage8OpeningInputPlan { symbol: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_16"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_16"), oracle: "InstructionRa_16", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: Stage8ClaimKind::Committed },
    Stage8OpeningInputPlan { symbol: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_17"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_17"), oracle: "InstructionRa_17", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: Stage8ClaimKind::Committed },
    Stage8OpeningInputPlan { symbol: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_18"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_18"), oracle: "InstructionRa_18", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: Stage8ClaimKind::Committed },
    Stage8OpeningInputPlan { symbol: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_19"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_19"), oracle: "InstructionRa_19", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: Stage8ClaimKind::Committed },
    Stage8OpeningInputPlan { symbol: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_20"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_20"), oracle: "InstructionRa_20", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: Stage8ClaimKind::Committed },
    Stage8OpeningInputPlan { symbol: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_21"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_21"), oracle: "InstructionRa_21", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: Stage8ClaimKind::Committed },
    Stage8OpeningInputPlan { symbol: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_22"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_22"), oracle: "InstructionRa_22", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: Stage8ClaimKind::Committed },
    Stage8OpeningInputPlan { symbol: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_23"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_23"), oracle: "InstructionRa_23", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: Stage8ClaimKind::Committed },
    Stage8OpeningInputPlan { symbol: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_24"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_24"), oracle: "InstructionRa_24", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: Stage8ClaimKind::Committed },
    Stage8OpeningInputPlan { symbol: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_25"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_25"), oracle: "InstructionRa_25", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: Stage8ClaimKind::Committed },
    Stage8OpeningInputPlan { symbol: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_26"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_26"), oracle: "InstructionRa_26", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: Stage8ClaimKind::Committed },
    Stage8OpeningInputPlan { symbol: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_27"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_27"), oracle: "InstructionRa_27", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: Stage8ClaimKind::Committed },
    Stage8OpeningInputPlan { symbol: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_28"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_28"), oracle: "InstructionRa_28", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: Stage8ClaimKind::Committed },
    Stage8OpeningInputPlan { symbol: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_29"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_29"), oracle: "InstructionRa_29", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: Stage8ClaimKind::Committed },
    Stage8OpeningInputPlan { symbol: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_30"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_30"), oracle: "InstructionRa_30", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: Stage8ClaimKind::Committed },
    Stage8OpeningInputPlan { symbol: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_31"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_31"), oracle: "InstructionRa_31", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: Stage8ClaimKind::Committed },
    Stage8OpeningInputPlan { symbol: Stage8OpeningInputSymbol::new("stage8.input.stage7.BytecodeRa_0"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.BytecodeRa_0"), oracle: "BytecodeRa_0", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: Stage8ClaimKind::Committed },
    Stage8OpeningInputPlan { symbol: Stage8OpeningInputSymbol::new("stage8.input.stage7.BytecodeRa_1"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.BytecodeRa_1"), oracle: "BytecodeRa_1", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: Stage8ClaimKind::Committed },
    Stage8OpeningInputPlan { symbol: Stage8OpeningInputSymbol::new("stage8.input.stage7.BytecodeRa_2"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.BytecodeRa_2"), oracle: "BytecodeRa_2", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: Stage8ClaimKind::Committed },
    Stage8OpeningInputPlan { symbol: Stage8OpeningInputSymbol::new("stage8.input.stage7.RamRa_0"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.RamRa_0"), oracle: "RamRa_0", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: Stage8ClaimKind::Committed },
    Stage8OpeningInputPlan { symbol: Stage8OpeningInputSymbol::new("stage8.input.stage7.RamRa_1"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.RamRa_1"), oracle: "RamRa_1", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: Stage8ClaimKind::Committed },
    Stage8OpeningInputPlan { symbol: Stage8OpeningInputSymbol::new("stage8.input.stage7.RamRa_2"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.RamRa_2"), oracle: "RamRa_2", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: Stage8ClaimKind::Committed },
    Stage8OpeningInputPlan { symbol: Stage8OpeningInputSymbol::new("stage8.input.stage7.RamRa_3"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.RamRa_3"), oracle: "RamRa_3", domain: "jolt.main_witness_commit_domain", point_arity: 20, claim_kind: Stage8ClaimKind::Committed },
];

#[rustfmt::skip]
pub const STAGE8_OPENING_CLAIMS: &[Stage8OpeningClaimPlan] = &[
    Stage8OpeningClaimPlan { symbol: Stage8OpeningClaimSymbol::new("stage8.evaluation.opening.RamInc"), oracle: "RamInc", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: Stage8OpeningInputSymbol::new("stage8.input.stage6.RamInc"), eval_source: Stage8OpeningInputSymbol::new("stage8.input.stage6.RamInc"), source_stage: Stage8SourceStage::Stage6, source_claim: Stage8SourceClaim::new("stage6.inc_claim_reduction.eval.RamInc") },
    Stage8OpeningClaimPlan { symbol: Stage8OpeningClaimSymbol::new("stage8.evaluation.opening.RdInc"), oracle: "RdInc", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: Stage8OpeningInputSymbol::new("stage8.input.stage6.RdInc"), eval_source: Stage8OpeningInputSymbol::new("stage8.input.stage6.RdInc"), source_stage: Stage8SourceStage::Stage6, source_claim: Stage8SourceClaim::new("stage6.inc_claim_reduction.eval.RdInc") },
    Stage8OpeningClaimPlan { symbol: Stage8OpeningClaimSymbol::new("stage8.evaluation.opening.InstructionRa_0"), oracle: "InstructionRa_0", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_0"), eval_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_0"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_0") },
    Stage8OpeningClaimPlan { symbol: Stage8OpeningClaimSymbol::new("stage8.evaluation.opening.InstructionRa_1"), oracle: "InstructionRa_1", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_1"), eval_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_1"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_1") },
    Stage8OpeningClaimPlan { symbol: Stage8OpeningClaimSymbol::new("stage8.evaluation.opening.InstructionRa_2"), oracle: "InstructionRa_2", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_2"), eval_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_2"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_2") },
    Stage8OpeningClaimPlan { symbol: Stage8OpeningClaimSymbol::new("stage8.evaluation.opening.InstructionRa_3"), oracle: "InstructionRa_3", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_3"), eval_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_3"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_3") },
    Stage8OpeningClaimPlan { symbol: Stage8OpeningClaimSymbol::new("stage8.evaluation.opening.InstructionRa_4"), oracle: "InstructionRa_4", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_4"), eval_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_4"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_4") },
    Stage8OpeningClaimPlan { symbol: Stage8OpeningClaimSymbol::new("stage8.evaluation.opening.InstructionRa_5"), oracle: "InstructionRa_5", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_5"), eval_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_5"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_5") },
    Stage8OpeningClaimPlan { symbol: Stage8OpeningClaimSymbol::new("stage8.evaluation.opening.InstructionRa_6"), oracle: "InstructionRa_6", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_6"), eval_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_6"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_6") },
    Stage8OpeningClaimPlan { symbol: Stage8OpeningClaimSymbol::new("stage8.evaluation.opening.InstructionRa_7"), oracle: "InstructionRa_7", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_7"), eval_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_7"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_7") },
    Stage8OpeningClaimPlan { symbol: Stage8OpeningClaimSymbol::new("stage8.evaluation.opening.InstructionRa_8"), oracle: "InstructionRa_8", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_8"), eval_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_8"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_8") },
    Stage8OpeningClaimPlan { symbol: Stage8OpeningClaimSymbol::new("stage8.evaluation.opening.InstructionRa_9"), oracle: "InstructionRa_9", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_9"), eval_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_9"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_9") },
    Stage8OpeningClaimPlan { symbol: Stage8OpeningClaimSymbol::new("stage8.evaluation.opening.InstructionRa_10"), oracle: "InstructionRa_10", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_10"), eval_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_10"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_10") },
    Stage8OpeningClaimPlan { symbol: Stage8OpeningClaimSymbol::new("stage8.evaluation.opening.InstructionRa_11"), oracle: "InstructionRa_11", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_11"), eval_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_11"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_11") },
    Stage8OpeningClaimPlan { symbol: Stage8OpeningClaimSymbol::new("stage8.evaluation.opening.InstructionRa_12"), oracle: "InstructionRa_12", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_12"), eval_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_12"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_12") },
    Stage8OpeningClaimPlan { symbol: Stage8OpeningClaimSymbol::new("stage8.evaluation.opening.InstructionRa_13"), oracle: "InstructionRa_13", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_13"), eval_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_13"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_13") },
    Stage8OpeningClaimPlan { symbol: Stage8OpeningClaimSymbol::new("stage8.evaluation.opening.InstructionRa_14"), oracle: "InstructionRa_14", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_14"), eval_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_14"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_14") },
    Stage8OpeningClaimPlan { symbol: Stage8OpeningClaimSymbol::new("stage8.evaluation.opening.InstructionRa_15"), oracle: "InstructionRa_15", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_15"), eval_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_15"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_15") },
    Stage8OpeningClaimPlan { symbol: Stage8OpeningClaimSymbol::new("stage8.evaluation.opening.InstructionRa_16"), oracle: "InstructionRa_16", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_16"), eval_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_16"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_16") },
    Stage8OpeningClaimPlan { symbol: Stage8OpeningClaimSymbol::new("stage8.evaluation.opening.InstructionRa_17"), oracle: "InstructionRa_17", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_17"), eval_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_17"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_17") },
    Stage8OpeningClaimPlan { symbol: Stage8OpeningClaimSymbol::new("stage8.evaluation.opening.InstructionRa_18"), oracle: "InstructionRa_18", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_18"), eval_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_18"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_18") },
    Stage8OpeningClaimPlan { symbol: Stage8OpeningClaimSymbol::new("stage8.evaluation.opening.InstructionRa_19"), oracle: "InstructionRa_19", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_19"), eval_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_19"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_19") },
    Stage8OpeningClaimPlan { symbol: Stage8OpeningClaimSymbol::new("stage8.evaluation.opening.InstructionRa_20"), oracle: "InstructionRa_20", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_20"), eval_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_20"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_20") },
    Stage8OpeningClaimPlan { symbol: Stage8OpeningClaimSymbol::new("stage8.evaluation.opening.InstructionRa_21"), oracle: "InstructionRa_21", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_21"), eval_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_21"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_21") },
    Stage8OpeningClaimPlan { symbol: Stage8OpeningClaimSymbol::new("stage8.evaluation.opening.InstructionRa_22"), oracle: "InstructionRa_22", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_22"), eval_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_22"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_22") },
    Stage8OpeningClaimPlan { symbol: Stage8OpeningClaimSymbol::new("stage8.evaluation.opening.InstructionRa_23"), oracle: "InstructionRa_23", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_23"), eval_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_23"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_23") },
    Stage8OpeningClaimPlan { symbol: Stage8OpeningClaimSymbol::new("stage8.evaluation.opening.InstructionRa_24"), oracle: "InstructionRa_24", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_24"), eval_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_24"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_24") },
    Stage8OpeningClaimPlan { symbol: Stage8OpeningClaimSymbol::new("stage8.evaluation.opening.InstructionRa_25"), oracle: "InstructionRa_25", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_25"), eval_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_25"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_25") },
    Stage8OpeningClaimPlan { symbol: Stage8OpeningClaimSymbol::new("stage8.evaluation.opening.InstructionRa_26"), oracle: "InstructionRa_26", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_26"), eval_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_26"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_26") },
    Stage8OpeningClaimPlan { symbol: Stage8OpeningClaimSymbol::new("stage8.evaluation.opening.InstructionRa_27"), oracle: "InstructionRa_27", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_27"), eval_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_27"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_27") },
    Stage8OpeningClaimPlan { symbol: Stage8OpeningClaimSymbol::new("stage8.evaluation.opening.InstructionRa_28"), oracle: "InstructionRa_28", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_28"), eval_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_28"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_28") },
    Stage8OpeningClaimPlan { symbol: Stage8OpeningClaimSymbol::new("stage8.evaluation.opening.InstructionRa_29"), oracle: "InstructionRa_29", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_29"), eval_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_29"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_29") },
    Stage8OpeningClaimPlan { symbol: Stage8OpeningClaimSymbol::new("stage8.evaluation.opening.InstructionRa_30"), oracle: "InstructionRa_30", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_30"), eval_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_30"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_30") },
    Stage8OpeningClaimPlan { symbol: Stage8OpeningClaimSymbol::new("stage8.evaluation.opening.InstructionRa_31"), oracle: "InstructionRa_31", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_31"), eval_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.InstructionRa_31"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.InstructionRa_31") },
    Stage8OpeningClaimPlan { symbol: Stage8OpeningClaimSymbol::new("stage8.evaluation.opening.BytecodeRa_0"), oracle: "BytecodeRa_0", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.BytecodeRa_0"), eval_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.BytecodeRa_0"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.BytecodeRa_0") },
    Stage8OpeningClaimPlan { symbol: Stage8OpeningClaimSymbol::new("stage8.evaluation.opening.BytecodeRa_1"), oracle: "BytecodeRa_1", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.BytecodeRa_1"), eval_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.BytecodeRa_1"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.BytecodeRa_1") },
    Stage8OpeningClaimPlan { symbol: Stage8OpeningClaimSymbol::new("stage8.evaluation.opening.BytecodeRa_2"), oracle: "BytecodeRa_2", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.BytecodeRa_2"), eval_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.BytecodeRa_2"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.BytecodeRa_2") },
    Stage8OpeningClaimPlan { symbol: Stage8OpeningClaimSymbol::new("stage8.evaluation.opening.RamRa_0"), oracle: "RamRa_0", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.RamRa_0"), eval_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.RamRa_0"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.RamRa_0") },
    Stage8OpeningClaimPlan { symbol: Stage8OpeningClaimSymbol::new("stage8.evaluation.opening.RamRa_1"), oracle: "RamRa_1", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.RamRa_1"), eval_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.RamRa_1"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.RamRa_1") },
    Stage8OpeningClaimPlan { symbol: Stage8OpeningClaimSymbol::new("stage8.evaluation.opening.RamRa_2"), oracle: "RamRa_2", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.RamRa_2"), eval_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.RamRa_2"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.RamRa_2") },
    Stage8OpeningClaimPlan { symbol: Stage8OpeningClaimSymbol::new("stage8.evaluation.opening.RamRa_3"), oracle: "RamRa_3", family: "jolt.main_witness_polys", domain: "jolt.main_witness_commit_domain", point_arity: 20, point_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.RamRa_3"), eval_source: Stage8OpeningInputSymbol::new("stage8.input.stage7.RamRa_3"), source_stage: Stage8SourceStage::Stage7, source_claim: Stage8SourceClaim::new("stage7.hamming_weight_claim_reduction.eval.RamRa_3") },
];

#[rustfmt::skip]
pub const STAGE8_OPENING_BATCH: Stage8OpeningBatchPlan = Stage8OpeningBatchPlan { symbol: Stage8OpeningBatchSymbol::new("stage8.evaluation.openings"), proof_slot: "stage8.evaluation", policy: "jolt_stage8_joint_rlc", count: 41, ordered_claims: STAGE8_OPENING_CLAIMS };

#[rustfmt::skip]
pub const STAGE8_PCS_PROOF: Stage8PcsProofPlan = Stage8PcsProofPlan { symbol: "stage8.evaluation.proof", mode: Stage8PcsProofMode::Verify, pcs: "dory", proof_slot: "stage8.evaluation", transcript_label: "rlc_claims", batch: Stage8OpeningBatchSymbol::new("stage8.evaluation.openings") };

#[rustfmt::skip]
pub const STAGE8_PROGRAM: Stage8EvaluationProgramPlan = Stage8EvaluationProgramPlan { role: "verifier", function: "jolt.stage8", params: STAGE8_PARAMS, evaluation_point_source: STAGE8_EVALUATION_POINT_SOURCE, opening_inputs: STAGE8_OPENING_INPUTS, opening_claims: STAGE8_OPENING_CLAIMS, opening_batch: STAGE8_OPENING_BATCH, pcs_proof: STAGE8_PCS_PROOF };
