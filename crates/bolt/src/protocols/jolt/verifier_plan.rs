use std::fmt::Write as _;

use crate::emit::rust::EmitError;
use crate::protocols::jolt::rust_target_plan::{
    ClaimKind, FieldExprKind, JoltVerifierRelationKind, OpeningEqualityMode, ProgramStepKind,
    RustTargetPlanError, SumcheckPointOrder, TranscriptSqueezeKind,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct VerifierProgramStepPlan {
    pub(crate) kind: ProgramStepKind,
    pub(crate) symbol: String,
}

impl VerifierProgramStepPlan {
    pub(crate) fn from_cpu(kind: &str, symbol: &str) -> Result<Self, EmitError> {
        Ok(Self {
            kind: ProgramStepKind::from_cpu_attr(kind).map_err(plan_error)?,
            symbol: symbol.to_owned(),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct VerifierTranscriptSqueezePlan {
    pub(crate) symbol: String,
    pub(crate) label: String,
    pub(crate) kind: TranscriptSqueezeKind,
    pub(crate) count: usize,
}

impl VerifierTranscriptSqueezePlan {
    pub(crate) fn from_cpu(
        symbol: &str,
        label: &str,
        kind: &str,
        count: usize,
    ) -> Result<Self, EmitError> {
        Ok(Self {
            symbol: symbol.to_owned(),
            label: label.to_owned(),
            kind: TranscriptSqueezeKind::from_cpu_attr(kind).map_err(plan_error)?,
            count,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct VerifierOpeningInputPlan {
    pub(crate) symbol: String,
    pub(crate) source_stage: String,
    pub(crate) source_claim: String,
    pub(crate) oracle: String,
    pub(crate) domain: String,
    pub(crate) point_arity: usize,
    pub(crate) claim_kind: ClaimKind,
}

impl VerifierOpeningInputPlan {
    pub(crate) fn from_cpu(
        symbol: &str,
        source_stage: &str,
        source_claim: &str,
        oracle: &str,
        domain: &str,
        point_arity: usize,
        claim_kind: &str,
    ) -> Result<Self, EmitError> {
        Ok(Self {
            symbol: symbol.to_owned(),
            source_stage: source_stage.to_owned(),
            source_claim: source_claim.to_owned(),
            oracle: oracle.to_owned(),
            domain: domain.to_owned(),
            point_arity,
            claim_kind: ClaimKind::from_cpu_attr(claim_kind).map_err(plan_error)?,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct VerifierFieldExprPlan {
    pub(crate) symbol: String,
    pub(crate) kind: FieldExprKind,
    pub(crate) operands: Vec<String>,
}

impl VerifierFieldExprPlan {
    pub(crate) fn from_cpu(
        symbol: &str,
        formula: &str,
        operands: &[String],
    ) -> Result<Self, EmitError> {
        Ok(Self {
            symbol: symbol.to_owned(),
            kind: FieldExprKind::from_cpu_attr(formula).map_err(plan_error)?,
            operands: operands.to_vec(),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct VerifierSumcheckClaimPlan {
    pub(crate) symbol: String,
    pub(crate) stage: String,
    pub(crate) domain: String,
    pub(crate) num_rounds: usize,
    pub(crate) degree: usize,
    pub(crate) claim: String,
    pub(crate) relation: JoltVerifierRelationKind,
    pub(crate) claim_value: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct VerifierSumcheckDriverPlan {
    pub(crate) symbol: String,
    pub(crate) stage: String,
    pub(crate) proof_slot: String,
    pub(crate) relation: JoltVerifierRelationKind,
    pub(crate) batch: String,
    pub(crate) policy: String,
    pub(crate) round_schedule: Vec<usize>,
    pub(crate) claim_label: String,
    pub(crate) round_label: String,
    pub(crate) num_rounds: usize,
    pub(crate) degree: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct VerifierSumcheckInstanceResultPlan {
    pub(crate) symbol: String,
    pub(crate) source: String,
    pub(crate) claim: String,
    pub(crate) relation: JoltVerifierRelationKind,
    pub(crate) index: usize,
    pub(crate) point_arity: usize,
    pub(crate) num_rounds: usize,
    pub(crate) round_offset: usize,
    pub(crate) point_order: SumcheckPointOrder,
    pub(crate) degree: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct VerifierOpeningClaimPlan {
    pub(crate) symbol: String,
    pub(crate) oracle: String,
    pub(crate) domain: String,
    pub(crate) point_arity: usize,
    pub(crate) claim_kind: ClaimKind,
    pub(crate) point_source: String,
    pub(crate) eval_source: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct VerifierOpeningClaimEqualityPlan {
    pub(crate) symbol: String,
    pub(crate) mode: OpeningEqualityMode,
    pub(crate) lhs: String,
    pub(crate) rhs: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct VerifierStagePlan {
    pub(crate) steps: Vec<VerifierProgramStepPlan>,
    pub(crate) transcript_squeezes: Vec<VerifierTranscriptSqueezePlan>,
    pub(crate) opening_inputs: Vec<VerifierOpeningInputPlan>,
    pub(crate) field_exprs: Vec<VerifierFieldExprPlan>,
    pub(crate) claims: Vec<VerifierSumcheckClaimPlan>,
    pub(crate) drivers: Vec<VerifierSumcheckDriverPlan>,
    pub(crate) instance_results: Vec<VerifierSumcheckInstanceResultPlan>,
    pub(crate) opening_claims: Vec<VerifierOpeningClaimPlan>,
    pub(crate) opening_equalities: Vec<VerifierOpeningClaimEqualityPlan>,
}

pub(crate) fn relation_from_cpu(value: &str) -> Result<JoltVerifierRelationKind, EmitError> {
    JoltVerifierRelationKind::from_cpu_attr(value).map_err(plan_error)
}

pub(crate) fn required_relation_from_cpu(
    value: Option<&str>,
    kind: &str,
    symbol: &str,
) -> Result<JoltVerifierRelationKind, EmitError> {
    value
        .ok_or_else(|| EmitError::new(format!("missing verifier {kind} relation for `{symbol}`")))
        .and_then(relation_from_cpu)
}

pub(crate) fn claim_kind_from_cpu(value: &str) -> Result<ClaimKind, EmitError> {
    ClaimKind::from_cpu_attr(value).map_err(plan_error)
}

pub(crate) fn sumcheck_point_order_from_cpu(value: &str) -> Result<SumcheckPointOrder, EmitError> {
    SumcheckPointOrder::from_cpu_attr(value).map_err(plan_error)
}

pub(crate) fn opening_equality_mode_from_cpu(
    value: &str,
) -> Result<OpeningEqualityMode, EmitError> {
    OpeningEqualityMode::from_cpu_attr(value).map_err(plan_error)
}

pub(crate) fn relation_kind_expr(
    stage_type_prefix: &str,
    relation: JoltVerifierRelationKind,
) -> String {
    format!(
        "{stage_type_prefix}RelationKind::{}",
        relation.rust_variant()
    )
}

pub(crate) fn program_step_kind_expr(stage_type_prefix: &str, kind: ProgramStepKind) -> String {
    format!(
        "{stage_type_prefix}ProgramStepKind::{}",
        kind.rust_variant()
    )
}

pub(crate) fn transcript_squeeze_kind_expr(
    stage_type_prefix: &str,
    kind: TranscriptSqueezeKind,
) -> String {
    format!(
        "{stage_type_prefix}TranscriptSqueezeKind::{}",
        kind.rust_variant()
    )
}

pub(crate) fn claim_kind_expr(stage_type_prefix: &str, kind: ClaimKind) -> String {
    format!("{stage_type_prefix}ClaimKind::{}", kind.rust_variant())
}

pub(crate) fn field_expr_kind_expr(stage_type_prefix: &str, kind: FieldExprKind) -> String {
    format!(
        "{stage_type_prefix}FieldExprKind::{}",
        kind.rust_variant_expr()
    )
}

pub(crate) fn sumcheck_point_order_expr(point_order: SumcheckPointOrder) -> String {
    format!(
        "bolt_verifier_runtime::SumcheckPointOrder::{}",
        point_order.rust_variant()
    )
}

pub(crate) fn opening_equality_mode_expr(
    stage_type_prefix: &str,
    mode: OpeningEqualityMode,
) -> String {
    format!(
        "{stage_type_prefix}OpeningEqualityMode::{}",
        mode.rust_variant()
    )
}

pub(crate) fn emit_program_step_constants(
    stage_type_prefix: &str,
    const_prefix: &str,
    steps: &[VerifierProgramStepPlan],
) -> String {
    let steps = steps
        .iter()
        .map(|step| {
            format!(
                "    {stage_type_prefix}ProgramStepPlan {{ kind: {}, symbol: {} }},",
                program_step_kind_expr(stage_type_prefix, step.kind),
                rust_str(&step.symbol),
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "pub const {const_prefix}_PROGRAM_STEPS: &[{stage_type_prefix}ProgramStepPlan] = &[\n{steps}\n];\n\n"
    )
}

pub(crate) fn emit_transcript_squeeze_constants(
    stage_type_prefix: &str,
    const_prefix: &str,
    squeezes: &[VerifierTranscriptSqueezePlan],
) -> String {
    let squeezes = squeezes
        .iter()
        .map(|squeeze| {
            format!(
                "    {stage_type_prefix}TranscriptSqueezePlan {{ symbol: {}, label: {}, kind: {}, count: {} }},",
                rust_str(&squeeze.symbol),
                rust_str(&squeeze.label),
                transcript_squeeze_kind_expr(stage_type_prefix, squeeze.kind),
                squeeze.count,
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "pub const {const_prefix}_TRANSCRIPT_SQUEEZES: &[{stage_type_prefix}TranscriptSqueezePlan] = &[\n{squeezes}\n];\n\n"
    )
}

pub(crate) fn emit_opening_input_constants(
    stage_type_prefix: &str,
    const_prefix: &str,
    inputs: &[VerifierOpeningInputPlan],
) -> String {
    let inputs = inputs
        .iter()
        .map(|input| {
            format!(
                "    {stage_type_prefix}OpeningInputPlan {{ symbol: {}, source_stage: {}, source_claim: {}, oracle: {}, domain: {}, point_arity: {}, claim_kind: {} }},",
                rust_str(&input.symbol),
                rust_str(&input.source_stage),
                rust_str(&input.source_claim),
                rust_str(&input.oracle),
                rust_str(&input.domain),
                input.point_arity,
                claim_kind_expr(stage_type_prefix, input.claim_kind),
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "pub const {const_prefix}_OPENING_INPUTS: &[{stage_type_prefix}OpeningInputPlan] = &[\n{inputs}\n];\n\n"
    )
}

pub(crate) fn emit_field_expr_constants(
    stage_type_prefix: &str,
    const_prefix: &str,
    exprs: &[VerifierFieldExprPlan],
) -> String {
    let exprs = exprs
        .iter()
        .map(|expr| {
            format!(
                "    {stage_type_prefix}FieldExprPlan {{ symbol: {}, kind: {}, operands: {} }},",
                rust_str(&expr.symbol),
                field_expr_kind_expr(stage_type_prefix, expr.kind),
                rust_str_slice_expr(&expr.operands),
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "pub const {const_prefix}_FIELD_EXPRS: &[{stage_type_prefix}FieldExprPlan] = &[\n{exprs}\n];\n"
    )
}

pub(crate) fn emit_sumcheck_claim_constants(
    stage_type_prefix: &str,
    const_prefix: &str,
    claims: &[VerifierSumcheckClaimPlan],
) -> String {
    let claims = claims
        .iter()
        .map(|claim| {
            format!(
                "    {stage_type_prefix}SumcheckClaimPlan {{ symbol: {}, stage: {}, domain: {}, num_rounds: {}, degree: {}, claim: {}, kernel: None, relation: Some({}), claim_value: {} }},",
                rust_str(&claim.symbol),
                rust_str(&claim.stage),
                rust_str(&claim.domain),
                claim.num_rounds,
                claim.degree,
                rust_str(&claim.claim),
                relation_kind_expr(stage_type_prefix, claim.relation),
                rust_str(&claim.claim_value),
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "pub const {const_prefix}_SUMCHECK_CLAIMS: &[{stage_type_prefix}SumcheckClaimPlan] = &[\n{claims}\n];\n"
    )
}

pub(crate) fn emit_sumcheck_driver_constants(
    stage_type_prefix: &str,
    const_prefix: &str,
    drivers: &[VerifierSumcheckDriverPlan],
) -> String {
    let mut source = String::new();
    for (index, driver) in drivers.iter().enumerate() {
        source.push_str(&emit_usize_array(
            &format!("{const_prefix}_SUMCHECK_DRIVER_{index}_ROUND_SCHEDULE"),
            &driver.round_schedule,
        ));
    }
    let drivers = drivers
        .iter()
        .enumerate()
        .map(|(index, driver)| {
            format!(
                "    {stage_type_prefix}SumcheckDriverPlan {{ symbol: {}, stage: {}, proof_slot: {}, kernel: None, relation: Some({}), batch: {}, policy: {}, round_schedule: {const_prefix}_SUMCHECK_DRIVER_{index}_ROUND_SCHEDULE, claim_label: {}, round_label: {}, num_rounds: {}, degree: {} }},",
                rust_str(&driver.symbol),
                rust_str(&driver.stage),
                rust_str(&driver.proof_slot),
                relation_kind_expr(stage_type_prefix, driver.relation),
                rust_str(&driver.batch),
                rust_str(&driver.policy),
                rust_str(&driver.claim_label),
                rust_str(&driver.round_label),
                driver.num_rounds,
                driver.degree,
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let _ = write!(
        source,
        "pub const {const_prefix}_SUMCHECK_DRIVERS: &[{stage_type_prefix}SumcheckDriverPlan] = &[\n{drivers}\n];\n"
    );
    source
}

pub(crate) fn emit_sumcheck_instance_result_constants(
    stage_type_prefix: &str,
    const_prefix: &str,
    instances: &[VerifierSumcheckInstanceResultPlan],
) -> String {
    let instances = instances
        .iter()
        .map(|instance| {
            format!(
                "    {stage_type_prefix}SumcheckInstanceResultPlan {{ symbol: {}, source: {}, claim: {}, relation: {}, index: {}, point_arity: {}, num_rounds: {}, round_offset: {}, point_order: {}, degree: {} }},",
                rust_str(&instance.symbol),
                rust_str(&instance.source),
                rust_str(&instance.claim),
                relation_kind_expr(stage_type_prefix, instance.relation),
                instance.index,
                instance.point_arity,
                instance.num_rounds,
                instance.round_offset,
                sumcheck_point_order_expr(instance.point_order),
                instance.degree,
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "pub const {const_prefix}_SUMCHECK_INSTANCE_RESULTS: &[{stage_type_prefix}SumcheckInstanceResultPlan] = &[\n{instances}\n];\n\n"
    )
}

pub(crate) fn emit_opening_claim_constants(
    stage_type_prefix: &str,
    const_prefix: &str,
    claims: &[VerifierOpeningClaimPlan],
) -> String {
    let claims = claims
        .iter()
        .map(|claim| {
            format!(
                "    {stage_type_prefix}OpeningClaimPlan {{ symbol: {}, oracle: {}, domain: {}, point_arity: {}, claim_kind: {}, point_source: {}, eval_source: {} }},",
                rust_str(&claim.symbol),
                rust_str(&claim.oracle),
                rust_str(&claim.domain),
                claim.point_arity,
                claim_kind_expr(stage_type_prefix, claim.claim_kind),
                rust_str(&claim.point_source),
                rust_str(&claim.eval_source),
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "pub const {const_prefix}_OPENING_CLAIMS: &[{stage_type_prefix}OpeningClaimPlan] = &[\n{claims}\n];\n\n"
    )
}

pub(crate) fn emit_opening_claim_equality_constants(
    stage_type_prefix: &str,
    const_prefix: &str,
    equalities: &[VerifierOpeningClaimEqualityPlan],
) -> String {
    let equalities = equalities
        .iter()
        .map(|equality| {
            format!(
                "    {stage_type_prefix}OpeningClaimEqualityPlan {{ symbol: {}, mode: {}, lhs: {}, rhs: {} }},",
                rust_str(&equality.symbol),
                opening_equality_mode_expr(stage_type_prefix, equality.mode),
                rust_str(&equality.lhs),
                rust_str(&equality.rhs),
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "pub const {const_prefix}_OPENING_EQUALITIES: &[{stage_type_prefix}OpeningClaimEqualityPlan] = &[\n{equalities}\n];\n\n"
    )
}

fn emit_usize_array(name: &str, values: &[usize]) -> String {
    let entries = values
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    format!("pub const {name}: &[usize] = &[{entries}];\n\n")
}

fn rust_str_slice_expr(values: &[String]) -> String {
    if values.is_empty() {
        return "&[]".to_owned();
    }
    let values = values
        .iter()
        .map(|value| rust_str(value))
        .collect::<Vec<_>>()
        .join(", ");
    format!("&[{values}]")
}

fn rust_str(value: &str) -> String {
    format!("{value:?}")
}

fn plan_error(error: RustTargetPlanError) -> EmitError {
    EmitError::new(error.to_string())
}
