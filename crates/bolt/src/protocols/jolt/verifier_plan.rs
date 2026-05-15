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
    pub(crate) relation: Option<JoltVerifierRelationKind>,
}

impl VerifierSumcheckClaimPlan {
    pub(crate) fn from_cpu(relation: Option<&str>) -> Result<Self, EmitError> {
        Ok(Self {
            relation: relation
                .map(JoltVerifierRelationKind::from_cpu_attr)
                .transpose()
                .map_err(plan_error)?,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct VerifierSumcheckDriverPlan {
    pub(crate) relation: Option<JoltVerifierRelationKind>,
}

impl VerifierSumcheckDriverPlan {
    pub(crate) fn from_cpu(relation: Option<&str>) -> Result<Self, EmitError> {
        Ok(Self {
            relation: relation
                .map(JoltVerifierRelationKind::from_cpu_attr)
                .transpose()
                .map_err(plan_error)?,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct VerifierSumcheckInstanceResultPlan {
    pub(crate) relation: JoltVerifierRelationKind,
    pub(crate) point_order: SumcheckPointOrder,
}

impl VerifierSumcheckInstanceResultPlan {
    pub(crate) fn from_cpu(relation: &str, point_order: &str) -> Result<Self, EmitError> {
        Ok(Self {
            relation: JoltVerifierRelationKind::from_cpu_attr(relation).map_err(plan_error)?,
            point_order: SumcheckPointOrder::from_cpu_attr(point_order).map_err(plan_error)?,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct VerifierOpeningClaimPlan {
    pub(crate) claim_kind: ClaimKind,
}

impl VerifierOpeningClaimPlan {
    pub(crate) fn from_cpu(claim_kind: &str) -> Result<Self, EmitError> {
        Ok(Self {
            claim_kind: ClaimKind::from_cpu_attr(claim_kind).map_err(plan_error)?,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct VerifierOpeningClaimEqualityPlan {
    pub(crate) mode: OpeningEqualityMode,
}

impl VerifierOpeningClaimEqualityPlan {
    pub(crate) fn from_cpu(mode: &str) -> Result<Self, EmitError> {
        Ok(Self {
            mode: OpeningEqualityMode::from_cpu_attr(mode).map_err(plan_error)?,
        })
    }
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

impl VerifierStagePlan {
    pub(crate) fn claim_relation(
        &self,
        index: usize,
    ) -> Result<JoltVerifierRelationKind, EmitError> {
        self.claims
            .get(index)
            .and_then(|claim| claim.relation)
            .ok_or_else(|| missing_plan_row("claim relation", index))
    }

    pub(crate) fn driver_relation(
        &self,
        index: usize,
    ) -> Result<JoltVerifierRelationKind, EmitError> {
        self.drivers
            .get(index)
            .and_then(|driver| driver.relation)
            .ok_or_else(|| missing_plan_row("driver relation", index))
    }
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

fn missing_plan_row(kind: &'static str, index: usize) -> EmitError {
    EmitError::new(format!("missing verifier-plan {kind} at index {index}"))
}

fn plan_error(error: RustTargetPlanError) -> EmitError {
    EmitError::new(error.to_string())
}
