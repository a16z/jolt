#![allow(clippy::too_many_lines)]

pub use bolt_verifier_runtime::{ClaimKind as Stage8ClaimKind, PcsProofMode as Stage8PcsProofMode, StageParams as Stage8Params, TypedPlanSymbol};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage8SourceStage {
    Stage6,
    Stage7,
}

impl Stage8SourceStage {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stage6 => "stage6",
            Self::Stage7 => "stage7",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Stage8OpeningInputTag {}
pub type Stage8OpeningInputSymbol = TypedPlanSymbol<Stage8OpeningInputTag>;
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Stage8OpeningClaimTag {}
pub type Stage8OpeningClaimSymbol = TypedPlanSymbol<Stage8OpeningClaimTag>;
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Stage8OpeningBatchTag {}
pub type Stage8OpeningBatchSymbol = TypedPlanSymbol<Stage8OpeningBatchTag>;
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
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

const fn stage8_opening_input(
    symbol: &'static str,
    source_stage: Stage8SourceStage,
    source_claim: &'static str,
    oracle: &'static str,
    domain: &'static str,
    point_arity: usize,
    claim_kind: Stage8ClaimKind,
) -> Stage8OpeningInputPlan {
    Stage8OpeningInputPlan {
        symbol: Stage8OpeningInputSymbol::new(symbol),
        source_stage,
        source_claim: Stage8SourceClaim::new(source_claim),
        oracle,
        domain,
        point_arity,
        claim_kind,
    }
}

const fn stage8_opening_claim(
    symbol: &'static str,
    oracle: &'static str,
    family: &'static str,
    domain: &'static str,
    point_arity: usize,
    input_symbol: &'static str,
    source: (Stage8SourceStage, &'static str),
) -> Stage8OpeningClaimPlan {
    Stage8OpeningClaimPlan {
        symbol: Stage8OpeningClaimSymbol::new(symbol),
        oracle,
        family,
        domain,
        point_arity,
        point_source: Stage8OpeningInputSymbol::new(input_symbol),
        eval_source: Stage8OpeningInputSymbol::new(input_symbol),
        source_stage: source.0,
        source_claim: Stage8SourceClaim::new(source.1),
    }
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

pub const STAGE8_EVALUATION_POINT_SOURCE: Stage8OpeningInputPlan = stage8_opening_input("stage8.evaluation.point_source", Stage8SourceStage::Stage7, "stage7.input.stage6.booleanity.InstructionRa_0", "InstructionRa_0", "jolt.main_witness_commit_domain", 20, Stage8ClaimKind::Committed);

#[rustfmt::skip]
pub const STAGE8_OPENING_INPUTS: &[Stage8OpeningInputPlan] = &[
    stage8_opening_input("stage8.evaluation.point_source", Stage8SourceStage::Stage7, "stage7.input.stage6.booleanity.InstructionRa_0", "InstructionRa_0", "jolt.main_witness_commit_domain", 20, Stage8ClaimKind::Committed),
    stage8_opening_input("stage8.input.stage6.RamInc", Stage8SourceStage::Stage6, "stage6.inc_claim_reduction.eval.RamInc", "RamInc", "jolt.main_witness_commit_domain", 20, Stage8ClaimKind::Committed),
    stage8_opening_input("stage8.input.stage6.RdInc", Stage8SourceStage::Stage6, "stage6.inc_claim_reduction.eval.RdInc", "RdInc", "jolt.main_witness_commit_domain", 20, Stage8ClaimKind::Committed),
    stage8_opening_input("stage8.input.stage7.InstructionRa_0", Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_0", "InstructionRa_0", "jolt.main_witness_commit_domain", 20, Stage8ClaimKind::Committed),
    stage8_opening_input("stage8.input.stage7.InstructionRa_1", Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_1", "InstructionRa_1", "jolt.main_witness_commit_domain", 20, Stage8ClaimKind::Committed),
    stage8_opening_input("stage8.input.stage7.InstructionRa_2", Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_2", "InstructionRa_2", "jolt.main_witness_commit_domain", 20, Stage8ClaimKind::Committed),
    stage8_opening_input("stage8.input.stage7.InstructionRa_3", Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_3", "InstructionRa_3", "jolt.main_witness_commit_domain", 20, Stage8ClaimKind::Committed),
    stage8_opening_input("stage8.input.stage7.InstructionRa_4", Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_4", "InstructionRa_4", "jolt.main_witness_commit_domain", 20, Stage8ClaimKind::Committed),
    stage8_opening_input("stage8.input.stage7.InstructionRa_5", Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_5", "InstructionRa_5", "jolt.main_witness_commit_domain", 20, Stage8ClaimKind::Committed),
    stage8_opening_input("stage8.input.stage7.InstructionRa_6", Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_6", "InstructionRa_6", "jolt.main_witness_commit_domain", 20, Stage8ClaimKind::Committed),
    stage8_opening_input("stage8.input.stage7.InstructionRa_7", Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_7", "InstructionRa_7", "jolt.main_witness_commit_domain", 20, Stage8ClaimKind::Committed),
    stage8_opening_input("stage8.input.stage7.InstructionRa_8", Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_8", "InstructionRa_8", "jolt.main_witness_commit_domain", 20, Stage8ClaimKind::Committed),
    stage8_opening_input("stage8.input.stage7.InstructionRa_9", Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_9", "InstructionRa_9", "jolt.main_witness_commit_domain", 20, Stage8ClaimKind::Committed),
    stage8_opening_input("stage8.input.stage7.InstructionRa_10", Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_10", "InstructionRa_10", "jolt.main_witness_commit_domain", 20, Stage8ClaimKind::Committed),
    stage8_opening_input("stage8.input.stage7.InstructionRa_11", Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_11", "InstructionRa_11", "jolt.main_witness_commit_domain", 20, Stage8ClaimKind::Committed),
    stage8_opening_input("stage8.input.stage7.InstructionRa_12", Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_12", "InstructionRa_12", "jolt.main_witness_commit_domain", 20, Stage8ClaimKind::Committed),
    stage8_opening_input("stage8.input.stage7.InstructionRa_13", Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_13", "InstructionRa_13", "jolt.main_witness_commit_domain", 20, Stage8ClaimKind::Committed),
    stage8_opening_input("stage8.input.stage7.InstructionRa_14", Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_14", "InstructionRa_14", "jolt.main_witness_commit_domain", 20, Stage8ClaimKind::Committed),
    stage8_opening_input("stage8.input.stage7.InstructionRa_15", Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_15", "InstructionRa_15", "jolt.main_witness_commit_domain", 20, Stage8ClaimKind::Committed),
    stage8_opening_input("stage8.input.stage7.InstructionRa_16", Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_16", "InstructionRa_16", "jolt.main_witness_commit_domain", 20, Stage8ClaimKind::Committed),
    stage8_opening_input("stage8.input.stage7.InstructionRa_17", Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_17", "InstructionRa_17", "jolt.main_witness_commit_domain", 20, Stage8ClaimKind::Committed),
    stage8_opening_input("stage8.input.stage7.InstructionRa_18", Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_18", "InstructionRa_18", "jolt.main_witness_commit_domain", 20, Stage8ClaimKind::Committed),
    stage8_opening_input("stage8.input.stage7.InstructionRa_19", Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_19", "InstructionRa_19", "jolt.main_witness_commit_domain", 20, Stage8ClaimKind::Committed),
    stage8_opening_input("stage8.input.stage7.InstructionRa_20", Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_20", "InstructionRa_20", "jolt.main_witness_commit_domain", 20, Stage8ClaimKind::Committed),
    stage8_opening_input("stage8.input.stage7.InstructionRa_21", Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_21", "InstructionRa_21", "jolt.main_witness_commit_domain", 20, Stage8ClaimKind::Committed),
    stage8_opening_input("stage8.input.stage7.InstructionRa_22", Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_22", "InstructionRa_22", "jolt.main_witness_commit_domain", 20, Stage8ClaimKind::Committed),
    stage8_opening_input("stage8.input.stage7.InstructionRa_23", Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_23", "InstructionRa_23", "jolt.main_witness_commit_domain", 20, Stage8ClaimKind::Committed),
    stage8_opening_input("stage8.input.stage7.InstructionRa_24", Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_24", "InstructionRa_24", "jolt.main_witness_commit_domain", 20, Stage8ClaimKind::Committed),
    stage8_opening_input("stage8.input.stage7.InstructionRa_25", Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_25", "InstructionRa_25", "jolt.main_witness_commit_domain", 20, Stage8ClaimKind::Committed),
    stage8_opening_input("stage8.input.stage7.InstructionRa_26", Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_26", "InstructionRa_26", "jolt.main_witness_commit_domain", 20, Stage8ClaimKind::Committed),
    stage8_opening_input("stage8.input.stage7.InstructionRa_27", Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_27", "InstructionRa_27", "jolt.main_witness_commit_domain", 20, Stage8ClaimKind::Committed),
    stage8_opening_input("stage8.input.stage7.InstructionRa_28", Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_28", "InstructionRa_28", "jolt.main_witness_commit_domain", 20, Stage8ClaimKind::Committed),
    stage8_opening_input("stage8.input.stage7.InstructionRa_29", Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_29", "InstructionRa_29", "jolt.main_witness_commit_domain", 20, Stage8ClaimKind::Committed),
    stage8_opening_input("stage8.input.stage7.InstructionRa_30", Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_30", "InstructionRa_30", "jolt.main_witness_commit_domain", 20, Stage8ClaimKind::Committed),
    stage8_opening_input("stage8.input.stage7.InstructionRa_31", Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_31", "InstructionRa_31", "jolt.main_witness_commit_domain", 20, Stage8ClaimKind::Committed),
    stage8_opening_input("stage8.input.stage7.BytecodeRa_0", Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.BytecodeRa_0", "BytecodeRa_0", "jolt.main_witness_commit_domain", 20, Stage8ClaimKind::Committed),
    stage8_opening_input("stage8.input.stage7.BytecodeRa_1", Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.BytecodeRa_1", "BytecodeRa_1", "jolt.main_witness_commit_domain", 20, Stage8ClaimKind::Committed),
    stage8_opening_input("stage8.input.stage7.BytecodeRa_2", Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.BytecodeRa_2", "BytecodeRa_2", "jolt.main_witness_commit_domain", 20, Stage8ClaimKind::Committed),
    stage8_opening_input("stage8.input.stage7.RamRa_0", Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.RamRa_0", "RamRa_0", "jolt.main_witness_commit_domain", 20, Stage8ClaimKind::Committed),
    stage8_opening_input("stage8.input.stage7.RamRa_1", Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.RamRa_1", "RamRa_1", "jolt.main_witness_commit_domain", 20, Stage8ClaimKind::Committed),
    stage8_opening_input("stage8.input.stage7.RamRa_2", Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.RamRa_2", "RamRa_2", "jolt.main_witness_commit_domain", 20, Stage8ClaimKind::Committed),
    stage8_opening_input("stage8.input.stage7.RamRa_3", Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.RamRa_3", "RamRa_3", "jolt.main_witness_commit_domain", 20, Stage8ClaimKind::Committed),
];

#[rustfmt::skip]
pub const STAGE8_OPENING_CLAIMS: &[Stage8OpeningClaimPlan] = &[
    stage8_opening_claim("stage8.evaluation.opening.RamInc", "RamInc", "jolt.main_witness_polys", "jolt.main_witness_commit_domain", 20, "stage8.input.stage6.RamInc", (Stage8SourceStage::Stage6, "stage6.inc_claim_reduction.eval.RamInc")),
    stage8_opening_claim("stage8.evaluation.opening.RdInc", "RdInc", "jolt.main_witness_polys", "jolt.main_witness_commit_domain", 20, "stage8.input.stage6.RdInc", (Stage8SourceStage::Stage6, "stage6.inc_claim_reduction.eval.RdInc")),
    stage8_opening_claim("stage8.evaluation.opening.InstructionRa_0", "InstructionRa_0", "jolt.main_witness_polys", "jolt.main_witness_commit_domain", 20, "stage8.input.stage7.InstructionRa_0", (Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_0")),
    stage8_opening_claim("stage8.evaluation.opening.InstructionRa_1", "InstructionRa_1", "jolt.main_witness_polys", "jolt.main_witness_commit_domain", 20, "stage8.input.stage7.InstructionRa_1", (Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_1")),
    stage8_opening_claim("stage8.evaluation.opening.InstructionRa_2", "InstructionRa_2", "jolt.main_witness_polys", "jolt.main_witness_commit_domain", 20, "stage8.input.stage7.InstructionRa_2", (Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_2")),
    stage8_opening_claim("stage8.evaluation.opening.InstructionRa_3", "InstructionRa_3", "jolt.main_witness_polys", "jolt.main_witness_commit_domain", 20, "stage8.input.stage7.InstructionRa_3", (Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_3")),
    stage8_opening_claim("stage8.evaluation.opening.InstructionRa_4", "InstructionRa_4", "jolt.main_witness_polys", "jolt.main_witness_commit_domain", 20, "stage8.input.stage7.InstructionRa_4", (Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_4")),
    stage8_opening_claim("stage8.evaluation.opening.InstructionRa_5", "InstructionRa_5", "jolt.main_witness_polys", "jolt.main_witness_commit_domain", 20, "stage8.input.stage7.InstructionRa_5", (Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_5")),
    stage8_opening_claim("stage8.evaluation.opening.InstructionRa_6", "InstructionRa_6", "jolt.main_witness_polys", "jolt.main_witness_commit_domain", 20, "stage8.input.stage7.InstructionRa_6", (Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_6")),
    stage8_opening_claim("stage8.evaluation.opening.InstructionRa_7", "InstructionRa_7", "jolt.main_witness_polys", "jolt.main_witness_commit_domain", 20, "stage8.input.stage7.InstructionRa_7", (Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_7")),
    stage8_opening_claim("stage8.evaluation.opening.InstructionRa_8", "InstructionRa_8", "jolt.main_witness_polys", "jolt.main_witness_commit_domain", 20, "stage8.input.stage7.InstructionRa_8", (Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_8")),
    stage8_opening_claim("stage8.evaluation.opening.InstructionRa_9", "InstructionRa_9", "jolt.main_witness_polys", "jolt.main_witness_commit_domain", 20, "stage8.input.stage7.InstructionRa_9", (Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_9")),
    stage8_opening_claim("stage8.evaluation.opening.InstructionRa_10", "InstructionRa_10", "jolt.main_witness_polys", "jolt.main_witness_commit_domain", 20, "stage8.input.stage7.InstructionRa_10", (Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_10")),
    stage8_opening_claim("stage8.evaluation.opening.InstructionRa_11", "InstructionRa_11", "jolt.main_witness_polys", "jolt.main_witness_commit_domain", 20, "stage8.input.stage7.InstructionRa_11", (Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_11")),
    stage8_opening_claim("stage8.evaluation.opening.InstructionRa_12", "InstructionRa_12", "jolt.main_witness_polys", "jolt.main_witness_commit_domain", 20, "stage8.input.stage7.InstructionRa_12", (Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_12")),
    stage8_opening_claim("stage8.evaluation.opening.InstructionRa_13", "InstructionRa_13", "jolt.main_witness_polys", "jolt.main_witness_commit_domain", 20, "stage8.input.stage7.InstructionRa_13", (Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_13")),
    stage8_opening_claim("stage8.evaluation.opening.InstructionRa_14", "InstructionRa_14", "jolt.main_witness_polys", "jolt.main_witness_commit_domain", 20, "stage8.input.stage7.InstructionRa_14", (Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_14")),
    stage8_opening_claim("stage8.evaluation.opening.InstructionRa_15", "InstructionRa_15", "jolt.main_witness_polys", "jolt.main_witness_commit_domain", 20, "stage8.input.stage7.InstructionRa_15", (Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_15")),
    stage8_opening_claim("stage8.evaluation.opening.InstructionRa_16", "InstructionRa_16", "jolt.main_witness_polys", "jolt.main_witness_commit_domain", 20, "stage8.input.stage7.InstructionRa_16", (Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_16")),
    stage8_opening_claim("stage8.evaluation.opening.InstructionRa_17", "InstructionRa_17", "jolt.main_witness_polys", "jolt.main_witness_commit_domain", 20, "stage8.input.stage7.InstructionRa_17", (Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_17")),
    stage8_opening_claim("stage8.evaluation.opening.InstructionRa_18", "InstructionRa_18", "jolt.main_witness_polys", "jolt.main_witness_commit_domain", 20, "stage8.input.stage7.InstructionRa_18", (Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_18")),
    stage8_opening_claim("stage8.evaluation.opening.InstructionRa_19", "InstructionRa_19", "jolt.main_witness_polys", "jolt.main_witness_commit_domain", 20, "stage8.input.stage7.InstructionRa_19", (Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_19")),
    stage8_opening_claim("stage8.evaluation.opening.InstructionRa_20", "InstructionRa_20", "jolt.main_witness_polys", "jolt.main_witness_commit_domain", 20, "stage8.input.stage7.InstructionRa_20", (Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_20")),
    stage8_opening_claim("stage8.evaluation.opening.InstructionRa_21", "InstructionRa_21", "jolt.main_witness_polys", "jolt.main_witness_commit_domain", 20, "stage8.input.stage7.InstructionRa_21", (Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_21")),
    stage8_opening_claim("stage8.evaluation.opening.InstructionRa_22", "InstructionRa_22", "jolt.main_witness_polys", "jolt.main_witness_commit_domain", 20, "stage8.input.stage7.InstructionRa_22", (Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_22")),
    stage8_opening_claim("stage8.evaluation.opening.InstructionRa_23", "InstructionRa_23", "jolt.main_witness_polys", "jolt.main_witness_commit_domain", 20, "stage8.input.stage7.InstructionRa_23", (Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_23")),
    stage8_opening_claim("stage8.evaluation.opening.InstructionRa_24", "InstructionRa_24", "jolt.main_witness_polys", "jolt.main_witness_commit_domain", 20, "stage8.input.stage7.InstructionRa_24", (Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_24")),
    stage8_opening_claim("stage8.evaluation.opening.InstructionRa_25", "InstructionRa_25", "jolt.main_witness_polys", "jolt.main_witness_commit_domain", 20, "stage8.input.stage7.InstructionRa_25", (Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_25")),
    stage8_opening_claim("stage8.evaluation.opening.InstructionRa_26", "InstructionRa_26", "jolt.main_witness_polys", "jolt.main_witness_commit_domain", 20, "stage8.input.stage7.InstructionRa_26", (Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_26")),
    stage8_opening_claim("stage8.evaluation.opening.InstructionRa_27", "InstructionRa_27", "jolt.main_witness_polys", "jolt.main_witness_commit_domain", 20, "stage8.input.stage7.InstructionRa_27", (Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_27")),
    stage8_opening_claim("stage8.evaluation.opening.InstructionRa_28", "InstructionRa_28", "jolt.main_witness_polys", "jolt.main_witness_commit_domain", 20, "stage8.input.stage7.InstructionRa_28", (Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_28")),
    stage8_opening_claim("stage8.evaluation.opening.InstructionRa_29", "InstructionRa_29", "jolt.main_witness_polys", "jolt.main_witness_commit_domain", 20, "stage8.input.stage7.InstructionRa_29", (Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_29")),
    stage8_opening_claim("stage8.evaluation.opening.InstructionRa_30", "InstructionRa_30", "jolt.main_witness_polys", "jolt.main_witness_commit_domain", 20, "stage8.input.stage7.InstructionRa_30", (Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_30")),
    stage8_opening_claim("stage8.evaluation.opening.InstructionRa_31", "InstructionRa_31", "jolt.main_witness_polys", "jolt.main_witness_commit_domain", 20, "stage8.input.stage7.InstructionRa_31", (Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.InstructionRa_31")),
    stage8_opening_claim("stage8.evaluation.opening.BytecodeRa_0", "BytecodeRa_0", "jolt.main_witness_polys", "jolt.main_witness_commit_domain", 20, "stage8.input.stage7.BytecodeRa_0", (Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.BytecodeRa_0")),
    stage8_opening_claim("stage8.evaluation.opening.BytecodeRa_1", "BytecodeRa_1", "jolt.main_witness_polys", "jolt.main_witness_commit_domain", 20, "stage8.input.stage7.BytecodeRa_1", (Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.BytecodeRa_1")),
    stage8_opening_claim("stage8.evaluation.opening.BytecodeRa_2", "BytecodeRa_2", "jolt.main_witness_polys", "jolt.main_witness_commit_domain", 20, "stage8.input.stage7.BytecodeRa_2", (Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.BytecodeRa_2")),
    stage8_opening_claim("stage8.evaluation.opening.RamRa_0", "RamRa_0", "jolt.main_witness_polys", "jolt.main_witness_commit_domain", 20, "stage8.input.stage7.RamRa_0", (Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.RamRa_0")),
    stage8_opening_claim("stage8.evaluation.opening.RamRa_1", "RamRa_1", "jolt.main_witness_polys", "jolt.main_witness_commit_domain", 20, "stage8.input.stage7.RamRa_1", (Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.RamRa_1")),
    stage8_opening_claim("stage8.evaluation.opening.RamRa_2", "RamRa_2", "jolt.main_witness_polys", "jolt.main_witness_commit_domain", 20, "stage8.input.stage7.RamRa_2", (Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.RamRa_2")),
    stage8_opening_claim("stage8.evaluation.opening.RamRa_3", "RamRa_3", "jolt.main_witness_polys", "jolt.main_witness_commit_domain", 20, "stage8.input.stage7.RamRa_3", (Stage8SourceStage::Stage7, "stage7.hamming_weight_claim_reduction.eval.RamRa_3")),
];

#[rustfmt::skip]
pub const STAGE8_OPENING_BATCH: Stage8OpeningBatchPlan = Stage8OpeningBatchPlan { symbol: Stage8OpeningBatchSymbol::new("stage8.evaluation.openings"), proof_slot: "stage8.evaluation", policy: "jolt_stage8_joint_rlc", count: 41, ordered_claims: STAGE8_OPENING_CLAIMS };

#[rustfmt::skip]
pub const STAGE8_PCS_PROOF: Stage8PcsProofPlan = Stage8PcsProofPlan { symbol: "stage8.evaluation.proof", mode: Stage8PcsProofMode::Verify, pcs: "dory", proof_slot: "stage8.evaluation", transcript_label: "rlc_claims", batch: Stage8OpeningBatchSymbol::new("stage8.evaluation.openings") };

#[rustfmt::skip]
pub const STAGE8_PROGRAM: Stage8EvaluationProgramPlan = Stage8EvaluationProgramPlan { role: "verifier", function: "jolt.stage8", params: STAGE8_PARAMS, evaluation_point_source: STAGE8_EVALUATION_POINT_SOURCE, opening_inputs: STAGE8_OPENING_INPUTS, opening_claims: STAGE8_OPENING_CLAIMS, opening_batch: STAGE8_OPENING_BATCH, pcs_proof: STAGE8_PCS_PROOF };
