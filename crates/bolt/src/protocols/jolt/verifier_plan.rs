use std::collections::BTreeSet;
use std::fmt::Write as _;

use crate::emit::rust::EmitError;
use crate::protocols::jolt::rust_target_plan::{
    ClaimKind, FieldExprKind, JoltVerifierRelationKind, OpeningEqualityMode, ProgramStepKind,
    RustTargetPlanError, ScalarExprKind, SumcheckPointOrder, TranscriptSqueezeKind,
};
use crate::protocols::jolt::verifier_eval_families::IndexedEvalFamilyPlan;
use crate::protocols::jolt::verifier_opening_rows::{
    CpuOpeningBatchPlan, CpuOpeningClaimEqualityPlan, CpuOpeningClaimPlan,
};
use crate::protocols::jolt::verifier_relation_outputs::{
    RelationOutputEvalFamilyPlan, RelationOutputFunctionFamilyPlan, RelationOutputPlan,
    RelationOutputProductFamilyPlan, StructuredPolynomialEvalPlan,
};
use crate::protocols::jolt::verifier_values::{
    VerifierFieldVectorSourceKind, VerifierFieldVectorSourceSet, VerifierPointSourceKind,
    VerifierPointSourceSet, VerifierScalarSourceKind, VerifierScalarSourceSet,
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
pub(crate) struct VerifierTranscriptAbsorbBytesPlan {
    pub(crate) symbol: String,
    pub(crate) label: String,
    pub(crate) payload: String,
}

impl VerifierTranscriptAbsorbBytesPlan {
    pub(crate) fn from_cpu(symbol: &str, label: &str, payload: &str) -> Self {
        Self {
            symbol: symbol.to_owned(),
            label: label.to_owned(),
            payload: payload.to_owned(),
        }
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
pub(crate) struct VerifierFieldConstantPlan {
    pub(crate) symbol: String,
    pub(crate) field: String,
    pub(crate) value: usize,
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
pub(crate) struct VerifierScalarExprPlan {
    pub(crate) symbol: String,
    pub(crate) kind: ScalarExprKind,
    pub(crate) operands: Vec<String>,
}

impl VerifierScalarExprPlan {
    pub(crate) fn from_cpu(
        symbol: &str,
        formula: &str,
        operands: &[String],
    ) -> Result<Self, EmitError> {
        Ok(Self {
            symbol: symbol.to_owned(),
            kind: ScalarExprKind::from_cpu_attr(formula).map_err(plan_error)?,
            operands: operands.to_vec(),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct VerifierSumcheckEvalPlan {
    pub(crate) symbol: String,
    pub(crate) source: String,
    pub(crate) name: String,
    pub(crate) index: usize,
    pub(crate) oracle: String,
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
pub(crate) struct VerifierSumcheckBatchPlan {
    pub(crate) symbol: String,
    pub(crate) stage: String,
    pub(crate) proof_slot: String,
    pub(crate) policy: String,
    pub(crate) count: usize,
    pub(crate) claim_operands: Vec<String>,
    pub(crate) claim_label: String,
    pub(crate) round_label: String,
    pub(crate) round_schedule: Vec<usize>,
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
pub(crate) enum VerifierPointExprKind {
    Zero { field: String, arity: usize },
    Slice { offset: usize, length: usize },
    Concat { layout: String, arity: usize },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct VerifierPointExprPlan {
    pub(crate) symbol: String,
    pub(crate) kind: VerifierPointExprKind,
    pub(crate) operands: Vec<String>,
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
pub(crate) struct VerifierOpeningBatchPlan {
    pub(crate) symbol: String,
    pub(crate) stage: String,
    pub(crate) proof_slot: String,
    pub(crate) policy: String,
    pub(crate) count: usize,
    pub(crate) ordered_claims: Vec<String>,
    pub(crate) claim_operands: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct VerifierStagePlan {
    pub(crate) steps: Vec<VerifierProgramStepPlan>,
    pub(crate) transcript_squeezes: Vec<VerifierTranscriptSqueezePlan>,
    pub(crate) transcript_absorb_bytes: Vec<VerifierTranscriptAbsorbBytesPlan>,
    pub(crate) opening_inputs: Vec<VerifierOpeningInputPlan>,
    pub(crate) field_constants: Vec<VerifierFieldConstantPlan>,
    pub(crate) field_exprs: Vec<VerifierFieldExprPlan>,
    pub(crate) scalar_exprs: Vec<VerifierScalarExprPlan>,
    pub(crate) claims: Vec<VerifierSumcheckClaimPlan>,
    pub(crate) batches: Vec<VerifierSumcheckBatchPlan>,
    pub(crate) drivers: Vec<VerifierSumcheckDriverPlan>,
    pub(crate) instance_results: Vec<VerifierSumcheckInstanceResultPlan>,
    pub(crate) sumcheck_evals: Vec<VerifierSumcheckEvalPlan>,
    pub(crate) indexed_eval_families: Vec<IndexedEvalFamilyPlan>,
    pub(crate) relation_output_values: Vec<StructuredPolynomialEvalPlan>,
    pub(crate) relation_output_eval_families: Vec<RelationOutputEvalFamilyPlan>,
    pub(crate) relation_output_product_families: Vec<RelationOutputProductFamilyPlan>,
    pub(crate) relation_output_function_families: Vec<RelationOutputFunctionFamilyPlan>,
    pub(crate) relation_outputs: Vec<RelationOutputPlan>,
    pub(crate) point_exprs: Vec<VerifierPointExprPlan>,
    pub(crate) opening_claims: Vec<VerifierOpeningClaimPlan>,
    pub(crate) opening_equalities: Vec<VerifierOpeningClaimEqualityPlan>,
    pub(crate) opening_batches: Vec<VerifierOpeningBatchPlan>,
}

impl VerifierStagePlan {
    pub(crate) fn scalar_value_sources(&self) -> VerifierScalarSourceSet {
        let mut values = VerifierScalarSourceSet::default();
        values.extend(
            self.opening_inputs.iter().map(|input| &input.symbol),
            VerifierScalarSourceKind::OpeningInput,
        );
        values.extend(
            self.field_constants.iter().map(|constant| &constant.symbol),
            VerifierScalarSourceKind::FieldConstant,
        );
        values.extend(
            self.transcript_squeezes
                .iter()
                .filter(|squeeze| {
                    matches!(
                        squeeze.kind,
                        TranscriptSqueezeKind::ChallengeScalar | TranscriptSqueezeKind::Scalar
                    )
                })
                .map(|squeeze| &squeeze.symbol),
            VerifierScalarSourceKind::TranscriptScalar,
        );
        values.extend(
            self.relation_output_eval_families
                .iter()
                .map(|family| &family.symbol),
            VerifierScalarSourceKind::OutputEvalFamily,
        );
        values.extend(
            self.relation_output_product_families
                .iter()
                .map(|family| &family.symbol),
            VerifierScalarSourceKind::OutputProductFamily,
        );
        values.extend(
            self.relation_output_function_families
                .iter()
                .map(|family| &family.symbol),
            VerifierScalarSourceKind::OutputFunctionFamily,
        );
        values.extend(
            self.field_exprs.iter().map(|expr| &expr.symbol),
            VerifierScalarSourceKind::FieldExpr,
        );
        values.extend(
            self.scalar_exprs.iter().map(|expr| &expr.symbol),
            VerifierScalarSourceKind::ScalarExpr,
        );
        values.extend(
            self.relation_outputs
                .iter()
                .flat_map(|claim| claim.local_scalars.iter()),
            VerifierScalarSourceKind::PointDerived,
        );
        values.extend(
            self.sumcheck_evals.iter().map(|eval| &eval.symbol),
            VerifierScalarSourceKind::SumcheckEval,
        );
        values
    }

    pub(crate) fn field_vector_value_sources(&self) -> VerifierFieldVectorSourceSet {
        let mut values = VerifierFieldVectorSourceSet::default();
        values.extend(
            self.indexed_eval_families
                .iter()
                .map(|family| &family.symbol),
            VerifierFieldVectorSourceKind::IndexedEvalFamily,
        );
        values
    }

    pub(crate) fn point_value_sources(&self) -> VerifierPointSourceSet {
        let mut values = VerifierPointSourceSet::default();
        values.extend(
            self.instance_results
                .iter()
                .map(|instance| &instance.symbol),
            VerifierPointSourceKind::SumcheckInstance,
        );
        values.extend(
            self.opening_inputs.iter().map(|input| &input.symbol),
            VerifierPointSourceKind::OpeningInput,
        );
        values.extend(
            self.point_exprs.iter().map(|expr| &expr.symbol),
            VerifierPointSourceKind::PointExpr,
        );
        values
    }

    pub(crate) fn opening_point_sources(&self) -> BTreeSet<String> {
        let mut values = BTreeSet::new();
        values.extend(self.drivers.iter().map(|driver| driver.symbol.clone()));
        values.extend(
            self.instance_results
                .iter()
                .map(|instance| instance.symbol.clone()),
        );
        values.extend(self.opening_inputs.iter().map(|input| input.symbol.clone()));
        values.extend(self.point_exprs.iter().map(|expr| expr.symbol.clone()));
        values
    }
}

pub(crate) trait VerifierProgramStepSource {
    fn kind(&self) -> &str;
    fn symbol(&self) -> &str;
}

pub(crate) trait VerifierTranscriptSqueezeSource {
    fn symbol(&self) -> &str;
    fn label(&self) -> &str;
    fn kind(&self) -> &str;
    fn count(&self) -> usize;
}

pub(crate) trait VerifierTranscriptAbsorbBytesSource {
    fn symbol(&self) -> &str;
    fn label(&self) -> &str;
    fn payload(&self) -> &str;
}

pub(crate) trait VerifierOpeningInputSource {
    fn symbol(&self) -> &str;
    fn source_stage(&self) -> &str;
    fn source_claim(&self) -> &str;
    fn oracle(&self) -> &str;
    fn domain(&self) -> &str;
    fn point_arity(&self) -> usize;
    fn claim_kind(&self) -> &str;
}

pub(crate) trait VerifierFieldConstantSource {
    fn symbol(&self) -> &str;
    fn field(&self) -> &str;
    fn value(&self) -> usize;
}

pub(crate) trait VerifierFieldExprSource {
    fn symbol(&self) -> &str;
    fn formula(&self) -> &str;
    fn operands(&self) -> &[String];
}

pub(crate) trait VerifierScalarExprSource {
    fn symbol(&self) -> &str;
    fn formula(&self) -> &str;
    fn operands(&self) -> &[String];
}

pub(crate) trait VerifierSumcheckEvalSource {
    fn symbol(&self) -> &str;
    fn source(&self) -> &str;
    fn name(&self) -> &str;
    fn index(&self) -> usize;
    fn oracle(&self) -> &str;
}

pub(crate) trait VerifierSumcheckClaimSource {
    fn symbol(&self) -> &str;
    fn stage(&self) -> &str;
    fn domain(&self) -> &str;
    fn num_rounds(&self) -> usize;
    fn degree(&self) -> usize;
    fn claim(&self) -> &str;
    fn relation(&self) -> Option<&str>;
    fn claim_value(&self) -> &str;
}

pub(crate) trait VerifierSumcheckBatchSource {
    fn symbol(&self) -> &str;
    fn stage(&self) -> &str;
    fn proof_slot(&self) -> &str;
    fn policy(&self) -> &str;
    fn count(&self) -> usize;
    fn claim_operands(&self) -> &[String];
    fn claim_label(&self) -> &str;
    fn round_label(&self) -> &str;
    fn round_schedule(&self) -> &[usize];
}

pub(crate) trait VerifierSumcheckDriverSource {
    fn symbol(&self) -> &str;
    fn stage(&self) -> &str;
    fn proof_slot(&self) -> &str;
    fn relation(&self) -> Option<&str>;
    fn batch(&self) -> &str;
    fn policy(&self) -> &str;
    fn round_schedule(&self) -> &[usize];
    fn claim_label(&self) -> &str;
    fn round_label(&self) -> &str;
    fn num_rounds(&self) -> usize;
    fn degree(&self) -> usize;
}

pub(crate) trait VerifierSumcheckInstanceResultSource {
    fn symbol(&self) -> &str;
    fn source(&self) -> &str;
    fn claim(&self) -> &str;
    fn relation(&self) -> &str;
    fn index(&self) -> usize;
    fn point_arity(&self) -> usize;
    fn num_rounds(&self) -> usize;
    fn round_offset(&self) -> usize;
    fn point_order(&self) -> &str;
    fn degree(&self) -> usize;
}

pub(crate) trait VerifierPointZeroSource {
    fn symbol(&self) -> &str;
    fn field(&self) -> &str;
    fn arity(&self) -> usize;
}

pub(crate) trait VerifierPointSliceSource {
    fn symbol(&self) -> &str;
    fn offset(&self) -> usize;
    fn length(&self) -> usize;
    fn input(&self) -> &str;
}

pub(crate) trait VerifierPointConcatSource {
    fn symbol(&self) -> &str;
    fn layout(&self) -> &str;
    fn arity(&self) -> usize;
    fn inputs(&self) -> &[String];
}

pub(crate) trait VerifierOpeningClaimSource {
    fn symbol(&self) -> &str;
    fn oracle(&self) -> &str;
    fn domain(&self) -> &str;
    fn point_arity(&self) -> usize;
    fn claim_kind(&self) -> &str;
    fn point_source(&self) -> &str;
    fn eval_source(&self) -> &str;
}

pub(crate) trait VerifierOpeningClaimEqualitySource {
    fn symbol(&self) -> &str;
    fn mode(&self) -> &str;
    fn lhs(&self) -> &str;
    fn rhs(&self) -> &str;
}

pub(crate) trait VerifierOpeningBatchSource {
    fn symbol(&self) -> &str;
    fn stage(&self) -> &str;
    fn proof_slot(&self) -> &str;
    fn policy(&self) -> &str;
    fn count(&self) -> usize;
    fn ordered_claims(&self) -> &[String];
    fn claim_operands(&self) -> &[String];
}

pub(crate) trait VerifierStagePlanSource {
    type Step: VerifierProgramStepSource;
    type Squeeze: VerifierTranscriptSqueezeSource;
    type OpeningInput: VerifierOpeningInputSource;
    type FieldConstant: VerifierFieldConstantSource;
    type FieldExpr: VerifierFieldExprSource;
    type ScalarExpr: VerifierScalarExprSource;
    type Claim: VerifierSumcheckClaimSource;
    type Batch: VerifierSumcheckBatchSource;
    type Driver: VerifierSumcheckDriverSource;
    type Instance: VerifierSumcheckInstanceResultSource;
    type Eval: VerifierSumcheckEvalSource;
    type PointSlice: VerifierPointSliceSource;
    type PointConcat: VerifierPointConcatSource;

    fn steps(&self) -> &[Self::Step];
    fn transcript_squeezes(&self) -> &[Self::Squeeze];
    fn transcript_absorb_bytes(&self) -> Vec<VerifierTranscriptAbsorbBytesPlan>;
    fn opening_inputs(&self) -> &[Self::OpeningInput];
    fn field_constants(&self) -> &[Self::FieldConstant];
    fn field_exprs(&self) -> &[Self::FieldExpr];
    fn scalar_exprs(&self) -> &[Self::ScalarExpr];
    fn claims(&self) -> &[Self::Claim];
    fn batches(&self) -> &[Self::Batch];
    fn drivers(&self) -> &[Self::Driver];
    fn instance_results(&self) -> &[Self::Instance];
    fn sumcheck_evals(&self) -> &[Self::Eval];
    fn indexed_eval_families(&self) -> &[IndexedEvalFamilyPlan];
    fn relation_output_values(&self) -> &[StructuredPolynomialEvalPlan];
    fn relation_output_eval_families(&self) -> &[RelationOutputEvalFamilyPlan] {
        &[]
    }
    fn relation_output_product_families(&self) -> &[RelationOutputProductFamilyPlan] {
        &[]
    }
    fn relation_output_function_families(&self) -> &[RelationOutputFunctionFamilyPlan] {
        &[]
    }
    fn relation_outputs(&self) -> &[RelationOutputPlan];
    fn point_exprs(&self) -> Vec<VerifierPointExprPlan>;
    fn opening_claims(&self) -> &[CpuOpeningClaimPlan];
    fn opening_equalities(&self) -> &[CpuOpeningClaimEqualityPlan];
    fn opening_batches(&self) -> &[CpuOpeningBatchPlan];
}

impl VerifierOpeningClaimSource for CpuOpeningClaimPlan {
    fn symbol(&self) -> &str {
        &self.symbol
    }

    fn oracle(&self) -> &str {
        &self.oracle
    }

    fn domain(&self) -> &str {
        &self.domain
    }

    fn point_arity(&self) -> usize {
        self.point_arity
    }

    fn claim_kind(&self) -> &str {
        &self.claim_kind
    }

    fn point_source(&self) -> &str {
        &self.point_source
    }

    fn eval_source(&self) -> &str {
        &self.eval_source
    }
}

impl VerifierOpeningClaimEqualitySource for CpuOpeningClaimEqualityPlan {
    fn symbol(&self) -> &str {
        &self.symbol
    }

    fn mode(&self) -> &str {
        &self.mode
    }

    fn lhs(&self) -> &str {
        &self.lhs
    }

    fn rhs(&self) -> &str {
        &self.rhs
    }
}

impl VerifierOpeningBatchSource for CpuOpeningBatchPlan {
    fn symbol(&self) -> &str {
        &self.symbol
    }

    fn stage(&self) -> &str {
        &self.stage
    }

    fn proof_slot(&self) -> &str {
        &self.proof_slot
    }

    fn policy(&self) -> &str {
        &self.policy
    }

    fn count(&self) -> usize {
        self.count
    }

    fn ordered_claims(&self) -> &[String] {
        &self.ordered_claims
    }

    fn claim_operands(&self) -> &[String] {
        &self.claim_operands
    }
}

pub(crate) fn stage_plan_from_cpu_sources<Source>(
    source: &Source,
) -> Result<VerifierStagePlan, EmitError>
where
    Source: VerifierStagePlanSource + ?Sized,
{
    Ok(VerifierStagePlan {
        steps: source
            .steps()
            .iter()
            .map(|step| VerifierProgramStepPlan::from_cpu(step.kind(), step.symbol()))
            .collect::<Result<Vec<_>, EmitError>>()?,
        transcript_squeezes: source
            .transcript_squeezes()
            .iter()
            .map(|squeeze| {
                VerifierTranscriptSqueezePlan::from_cpu(
                    squeeze.symbol(),
                    squeeze.label(),
                    squeeze.kind(),
                    squeeze.count(),
                )
            })
            .collect::<Result<Vec<_>, EmitError>>()?,
        transcript_absorb_bytes: source.transcript_absorb_bytes(),
        opening_inputs: source
            .opening_inputs()
            .iter()
            .map(|opening_input| {
                VerifierOpeningInputPlan::from_cpu(
                    opening_input.symbol(),
                    opening_input.source_stage(),
                    opening_input.source_claim(),
                    opening_input.oracle(),
                    opening_input.domain(),
                    opening_input.point_arity(),
                    opening_input.claim_kind(),
                )
            })
            .collect::<Result<Vec<_>, EmitError>>()?,
        field_constants: source
            .field_constants()
            .iter()
            .map(|constant| VerifierFieldConstantPlan {
                symbol: constant.symbol().to_owned(),
                field: constant.field().to_owned(),
                value: constant.value(),
            })
            .collect(),
        field_exprs: source
            .field_exprs()
            .iter()
            .map(|expr| {
                VerifierFieldExprPlan::from_cpu(expr.symbol(), expr.formula(), expr.operands())
            })
            .collect::<Result<Vec<_>, EmitError>>()?,
        scalar_exprs: source
            .scalar_exprs()
            .iter()
            .map(|expr| {
                VerifierScalarExprPlan::from_cpu(expr.symbol(), expr.formula(), expr.operands())
            })
            .collect::<Result<Vec<_>, EmitError>>()?,
        claims: source
            .claims()
            .iter()
            .map(|claim| {
                Ok(VerifierSumcheckClaimPlan {
                    symbol: claim.symbol().to_owned(),
                    stage: claim.stage().to_owned(),
                    domain: claim.domain().to_owned(),
                    num_rounds: claim.num_rounds(),
                    degree: claim.degree(),
                    claim: claim.claim().to_owned(),
                    relation: required_relation_from_cpu(
                        claim.relation(),
                        "claim",
                        claim.symbol(),
                    )?,
                    claim_value: claim.claim_value().to_owned(),
                })
            })
            .collect::<Result<Vec<_>, EmitError>>()?,
        batches: source
            .batches()
            .iter()
            .map(|batch| VerifierSumcheckBatchPlan {
                symbol: batch.symbol().to_owned(),
                stage: batch.stage().to_owned(),
                proof_slot: batch.proof_slot().to_owned(),
                policy: batch.policy().to_owned(),
                count: batch.count(),
                claim_operands: batch.claim_operands().to_vec(),
                claim_label: batch.claim_label().to_owned(),
                round_label: batch.round_label().to_owned(),
                round_schedule: batch.round_schedule().to_vec(),
            })
            .collect(),
        drivers: source
            .drivers()
            .iter()
            .map(|driver| {
                Ok(VerifierSumcheckDriverPlan {
                    symbol: driver.symbol().to_owned(),
                    stage: driver.stage().to_owned(),
                    proof_slot: driver.proof_slot().to_owned(),
                    relation: required_relation_from_cpu(
                        driver.relation(),
                        "driver",
                        driver.symbol(),
                    )?,
                    batch: driver.batch().to_owned(),
                    policy: driver.policy().to_owned(),
                    round_schedule: driver.round_schedule().to_vec(),
                    claim_label: driver.claim_label().to_owned(),
                    round_label: driver.round_label().to_owned(),
                    num_rounds: driver.num_rounds(),
                    degree: driver.degree(),
                })
            })
            .collect::<Result<Vec<_>, EmitError>>()?,
        instance_results: source
            .instance_results()
            .iter()
            .map(|instance| {
                Ok(VerifierSumcheckInstanceResultPlan {
                    symbol: instance.symbol().to_owned(),
                    source: instance.source().to_owned(),
                    claim: instance.claim().to_owned(),
                    relation: relation_from_cpu(instance.relation())?,
                    index: instance.index(),
                    point_arity: instance.point_arity(),
                    num_rounds: instance.num_rounds(),
                    round_offset: instance.round_offset(),
                    point_order: sumcheck_point_order_from_cpu(instance.point_order())?,
                    degree: instance.degree(),
                })
            })
            .collect::<Result<Vec<_>, EmitError>>()?,
        sumcheck_evals: source
            .sumcheck_evals()
            .iter()
            .map(|eval| VerifierSumcheckEvalPlan {
                symbol: eval.symbol().to_owned(),
                source: eval.source().to_owned(),
                name: eval.name().to_owned(),
                index: eval.index(),
                oracle: eval.oracle().to_owned(),
            })
            .collect(),
        indexed_eval_families: source.indexed_eval_families().to_vec(),
        relation_output_values: source.relation_output_values().to_vec(),
        relation_output_eval_families: source.relation_output_eval_families().to_vec(),
        relation_output_product_families: source.relation_output_product_families().to_vec(),
        relation_output_function_families: source.relation_output_function_families().to_vec(),
        relation_outputs: source.relation_outputs().to_vec(),
        point_exprs: source.point_exprs(),
        opening_claims: source
            .opening_claims()
            .iter()
            .map(|claim| {
                Ok(VerifierOpeningClaimPlan {
                    symbol: claim.symbol().to_owned(),
                    oracle: claim.oracle().to_owned(),
                    domain: claim.domain().to_owned(),
                    point_arity: claim.point_arity(),
                    claim_kind: claim_kind_from_cpu(claim.claim_kind())?,
                    point_source: claim.point_source().to_owned(),
                    eval_source: claim.eval_source().to_owned(),
                })
            })
            .collect::<Result<Vec<_>, EmitError>>()?,
        opening_equalities: source
            .opening_equalities()
            .iter()
            .map(|equality| {
                Ok(VerifierOpeningClaimEqualityPlan {
                    symbol: equality.symbol().to_owned(),
                    mode: opening_equality_mode_from_cpu(equality.mode())?,
                    lhs: equality.lhs().to_owned(),
                    rhs: equality.rhs().to_owned(),
                })
            })
            .collect::<Result<Vec<_>, EmitError>>()?,
        opening_batches: source
            .opening_batches()
            .iter()
            .map(|batch| VerifierOpeningBatchPlan {
                symbol: batch.symbol().to_owned(),
                stage: batch.stage().to_owned(),
                proof_slot: batch.proof_slot().to_owned(),
                policy: batch.policy().to_owned(),
                count: batch.count(),
                ordered_claims: batch.ordered_claims().to_vec(),
                claim_operands: batch.claim_operands().to_vec(),
            })
            .collect(),
    })
}

pub(crate) fn transcript_absorb_bytes_from_cpu<T: VerifierTranscriptAbsorbBytesSource>(
    absorbs: &[T],
) -> Vec<VerifierTranscriptAbsorbBytesPlan> {
    absorbs
        .iter()
        .map(|absorb| {
            VerifierTranscriptAbsorbBytesPlan::from_cpu(
                absorb.symbol(),
                absorb.label(),
                absorb.payload(),
            )
        })
        .collect()
}

pub(crate) fn point_zero_exprs_from_cpu<T: VerifierPointZeroSource>(
    zeros: &[T],
) -> Vec<VerifierPointExprPlan> {
    zeros
        .iter()
        .map(|zero| VerifierPointExprPlan {
            symbol: zero.symbol().to_owned(),
            kind: VerifierPointExprKind::Zero {
                field: zero.field().to_owned(),
                arity: zero.arity(),
            },
            operands: Vec::new(),
        })
        .collect()
}

pub(crate) fn point_slice_exprs_from_cpu<T: VerifierPointSliceSource>(
    slices: &[T],
) -> Vec<VerifierPointExprPlan> {
    slices
        .iter()
        .map(|slice| VerifierPointExprPlan {
            symbol: slice.symbol().to_owned(),
            kind: VerifierPointExprKind::Slice {
                offset: slice.offset(),
                length: slice.length(),
            },
            operands: vec![slice.input().to_owned()],
        })
        .collect()
}

pub(crate) fn point_concat_exprs_from_cpu<T: VerifierPointConcatSource>(
    concats: &[T],
) -> Vec<VerifierPointExprPlan> {
    concats
        .iter()
        .map(|concat| VerifierPointExprPlan {
            symbol: concat.symbol().to_owned(),
            kind: VerifierPointExprKind::Concat {
                layout: concat.layout().to_owned(),
                arity: concat.arity(),
            },
            operands: concat.inputs().to_vec(),
        })
        .collect()
}

pub(crate) fn point_exprs_from_cpu<S, C>(slices: &[S], concats: &[C]) -> Vec<VerifierPointExprPlan>
where
    S: VerifierPointSliceSource,
    C: VerifierPointConcatSource,
{
    let mut exprs = point_slice_exprs_from_cpu(slices);
    exprs.extend(point_concat_exprs_from_cpu(concats));
    exprs
}

pub(crate) fn point_exprs_with_zeros_from_cpu<Z, S, C>(
    zeros: &[Z],
    slices: &[S],
    concats: &[C],
) -> Vec<VerifierPointExprPlan>
where
    Z: VerifierPointZeroSource,
    S: VerifierPointSliceSource,
    C: VerifierPointConcatSource,
{
    let mut exprs = point_zero_exprs_from_cpu(zeros);
    exprs.extend(point_exprs_from_cpu(slices, concats));
    exprs
}

macro_rules! impl_verifier_plan_source_traits {
    (
        program = $program:ty,
        step = $step:ty,
        squeeze = $squeeze:ty,
        opening_input = $opening_input:ty,
        field_constant = $field_constant:ty,
        field_expr = $field_expr:ty,
        scalar_expr = $scalar_expr:ty,
        claim = $claim:ty,
        batch = $batch:ty,
        driver = $driver:ty,
        instance = $instance:ty,
        eval = $eval:ty,
        point_slice = $point_slice:ty,
        point_concat = $point_concat:ty
        $(, absorb = $absorb:ty)?
        $(, point_zero = $point_zero:ty)?
        $(, indexed_eval_families = $indexed_eval_families:ident)?
        $(, relation_output_eval_families = $relation_output_eval_families:ident)?
        $(, relation_output_product_families = $relation_output_product_families:ident)?
        $(, relation_output_function_families = $relation_output_function_families:ident)?
        $(,)?
    ) => {
        impl $crate::protocols::jolt::verifier_plan::VerifierStagePlanSource for $program {
            type Step = $step;
            type Squeeze = $squeeze;
            type OpeningInput = $opening_input;
            type FieldConstant = $field_constant;
            type FieldExpr = $field_expr;
            type ScalarExpr = $scalar_expr;
            type Claim = $claim;
            type Batch = $batch;
            type Driver = $driver;
            type Instance = $instance;
            type Eval = $eval;
            type PointSlice = $point_slice;
            type PointConcat = $point_concat;

            fn steps(&self) -> &[Self::Step] { &self.steps }
            fn transcript_squeezes(&self) -> &[Self::Squeeze] { &self.transcript_squeezes }
            fn transcript_absorb_bytes(&self) -> Vec<$crate::protocols::jolt::verifier_plan::VerifierTranscriptAbsorbBytesPlan> {
                $crate::protocols::jolt::verifier_plan::impl_verifier_plan_source_traits!(
                    @transcript_absorb_bytes self $(, $absorb)?
                )
            }
            fn opening_inputs(&self) -> &[Self::OpeningInput] { &self.opening_inputs }
            fn field_constants(&self) -> &[Self::FieldConstant] { &self.field_constants }
            fn field_exprs(&self) -> &[Self::FieldExpr] { &self.field_exprs }
            fn scalar_exprs(&self) -> &[Self::ScalarExpr] { &self.scalar_exprs }
            fn claims(&self) -> &[Self::Claim] { &self.claims }
            fn batches(&self) -> &[Self::Batch] { &self.batches }
            fn drivers(&self) -> &[Self::Driver] { &self.drivers }
            fn instance_results(&self) -> &[Self::Instance] { &self.instance_results }
            fn sumcheck_evals(&self) -> &[Self::Eval] { &self.evals }
            fn indexed_eval_families(&self) -> &[$crate::protocols::jolt::verifier_eval_families::IndexedEvalFamilyPlan] {
                $crate::protocols::jolt::verifier_plan::impl_verifier_plan_source_traits!(
                    @indexed_eval_families self $(, $indexed_eval_families)?
                )
            }
            fn relation_output_values(&self) -> &[$crate::protocols::jolt::verifier_relation_outputs::StructuredPolynomialEvalPlan] {
                &self.relation_output_values
            }
            fn relation_output_eval_families(&self) -> &[$crate::protocols::jolt::verifier_relation_outputs::RelationOutputEvalFamilyPlan] {
                $crate::protocols::jolt::verifier_plan::impl_verifier_plan_source_traits!(
                    @relation_output_eval_families self $(, $relation_output_eval_families)?
                )
            }
            fn relation_output_product_families(&self) -> &[$crate::protocols::jolt::verifier_relation_outputs::RelationOutputProductFamilyPlan] {
                $crate::protocols::jolt::verifier_plan::impl_verifier_plan_source_traits!(
                    @relation_output_product_families self $(, $relation_output_product_families)?
                )
            }
            fn relation_output_function_families(&self) -> &[$crate::protocols::jolt::verifier_relation_outputs::RelationOutputFunctionFamilyPlan] {
                $crate::protocols::jolt::verifier_plan::impl_verifier_plan_source_traits!(
                    @relation_output_function_families self $(, $relation_output_function_families)?
                )
            }
            fn relation_outputs(&self) -> &[$crate::protocols::jolt::verifier_relation_outputs::RelationOutputPlan] {
                &self.relation_outputs
            }
            fn point_exprs(&self) -> Vec<$crate::protocols::jolt::verifier_plan::VerifierPointExprPlan> {
                $crate::protocols::jolt::verifier_plan::impl_verifier_plan_source_traits!(
                    @point_exprs self $(, $point_zero)?
                )
            }
            fn opening_claims(&self) -> &[$crate::protocols::jolt::verifier_opening_rows::CpuOpeningClaimPlan] { &self.opening_claims }
            fn opening_equalities(&self) -> &[$crate::protocols::jolt::verifier_opening_rows::CpuOpeningClaimEqualityPlan] { &self.opening_equalities }
            fn opening_batches(&self) -> &[$crate::protocols::jolt::verifier_opening_rows::CpuOpeningBatchPlan] { &self.opening_batches }
        }

        impl $crate::protocols::jolt::verifier_plan::VerifierProgramStepSource for $step {
            fn kind(&self) -> &str { &self.kind }
            fn symbol(&self) -> &str { &self.symbol }
        }

        impl $crate::protocols::jolt::verifier_plan::VerifierTranscriptSqueezeSource for $squeeze {
            fn symbol(&self) -> &str { &self.symbol }
            fn label(&self) -> &str { &self.label }
            fn kind(&self) -> &str { &self.kind }
            fn count(&self) -> usize { self.count }
        }

        $(
        impl $crate::protocols::jolt::verifier_plan::VerifierTranscriptAbsorbBytesSource for $absorb {
            fn symbol(&self) -> &str { &self.symbol }
            fn label(&self) -> &str { &self.label }
            fn payload(&self) -> &str { &self.payload }
        }
        )?

        impl $crate::protocols::jolt::verifier_plan::VerifierOpeningInputSource for $opening_input {
            fn symbol(&self) -> &str { &self.symbol }
            fn source_stage(&self) -> &str { &self.source_stage }
            fn source_claim(&self) -> &str { &self.source_claim }
            fn oracle(&self) -> &str { &self.oracle }
            fn domain(&self) -> &str { &self.domain }
            fn point_arity(&self) -> usize { self.point_arity }
            fn claim_kind(&self) -> &str { &self.claim_kind }
        }

        impl $crate::protocols::jolt::verifier_plan::VerifierFieldConstantSource for $field_constant {
            fn symbol(&self) -> &str { &self.symbol }
            fn field(&self) -> &str { &self.field }
            fn value(&self) -> usize { self.value }
        }

        impl $crate::protocols::jolt::verifier_plan::VerifierFieldExprSource for $field_expr {
            fn symbol(&self) -> &str { &self.symbol }
            fn formula(&self) -> &str { &self.formula }
            fn operands(&self) -> &[String] { &self.operands }
        }

        impl $crate::protocols::jolt::verifier_plan::VerifierScalarExprSource for $scalar_expr {
            fn symbol(&self) -> &str { &self.symbol }
            fn formula(&self) -> &str { &self.formula }
            fn operands(&self) -> &[String] { &self.operands }
        }

        impl $crate::protocols::jolt::verifier_plan::VerifierSumcheckEvalSource for $eval {
            fn symbol(&self) -> &str { &self.symbol }
            fn source(&self) -> &str { &self.source }
            fn name(&self) -> &str { &self.name }
            fn index(&self) -> usize { self.index }
            fn oracle(&self) -> &str { &self.oracle }
        }

        impl $crate::protocols::jolt::verifier_plan::VerifierSumcheckClaimSource for $claim {
            fn symbol(&self) -> &str { &self.symbol }
            fn stage(&self) -> &str { &self.stage }
            fn domain(&self) -> &str { &self.domain }
            fn num_rounds(&self) -> usize { self.num_rounds }
            fn degree(&self) -> usize { self.degree }
            fn claim(&self) -> &str { &self.claim }
            fn relation(&self) -> Option<&str> { self.relation.as_deref() }
            fn claim_value(&self) -> &str { &self.claim_value }
        }

        impl $crate::protocols::jolt::verifier_plan::VerifierSumcheckBatchSource for $batch {
            fn symbol(&self) -> &str { &self.symbol }
            fn stage(&self) -> &str { &self.stage }
            fn proof_slot(&self) -> &str { &self.proof_slot }
            fn policy(&self) -> &str { &self.policy }
            fn count(&self) -> usize { self.count }
            fn claim_operands(&self) -> &[String] { &self.claim_operands }
            fn claim_label(&self) -> &str { &self.claim_label }
            fn round_label(&self) -> &str { &self.round_label }
            fn round_schedule(&self) -> &[usize] { &self.round_schedule }
        }

        impl $crate::protocols::jolt::verifier_plan::VerifierSumcheckDriverSource for $driver {
            fn symbol(&self) -> &str { &self.symbol }
            fn stage(&self) -> &str { &self.stage }
            fn proof_slot(&self) -> &str { &self.proof_slot }
            fn relation(&self) -> Option<&str> { self.relation.as_deref() }
            fn batch(&self) -> &str { &self.batch }
            fn policy(&self) -> &str { &self.policy }
            fn round_schedule(&self) -> &[usize] { &self.round_schedule }
            fn claim_label(&self) -> &str { &self.claim_label }
            fn round_label(&self) -> &str { &self.round_label }
            fn num_rounds(&self) -> usize { self.num_rounds }
            fn degree(&self) -> usize { self.degree }
        }

        impl $crate::protocols::jolt::verifier_plan::VerifierSumcheckInstanceResultSource for $instance {
            fn symbol(&self) -> &str { &self.symbol }
            fn source(&self) -> &str { &self.source }
            fn claim(&self) -> &str { &self.claim }
            fn relation(&self) -> &str { &self.relation }
            fn index(&self) -> usize { self.index }
            fn point_arity(&self) -> usize { self.point_arity }
            fn num_rounds(&self) -> usize { self.num_rounds }
            fn round_offset(&self) -> usize { self.round_offset }
            fn point_order(&self) -> &str { &self.point_order }
            fn degree(&self) -> usize { self.degree }
        }

        $(
        impl $crate::protocols::jolt::verifier_plan::VerifierPointZeroSource for $point_zero {
            fn symbol(&self) -> &str { &self.symbol }
            fn field(&self) -> &str { &self.field }
            fn arity(&self) -> usize { self.arity }
        }
        )?

        impl $crate::protocols::jolt::verifier_plan::VerifierPointSliceSource for $point_slice {
            fn symbol(&self) -> &str { &self.symbol }
            fn offset(&self) -> usize { self.offset }
            fn length(&self) -> usize { self.length }
            fn input(&self) -> &str { &self.input }
        }

        impl $crate::protocols::jolt::verifier_plan::VerifierPointConcatSource for $point_concat {
            fn symbol(&self) -> &str { &self.symbol }
            fn layout(&self) -> &str { &self.layout }
            fn arity(&self) -> usize { self.arity }
            fn inputs(&self) -> &[String] { &self.inputs }
        }

    };
    (@transcript_absorb_bytes $self:ident, $absorb:ty) => {
        $crate::protocols::jolt::verifier_plan::transcript_absorb_bytes_from_cpu(
            &$self.transcript_absorb_bytes,
        )
    };
    (@transcript_absorb_bytes $self:ident) => {
        Vec::new()
    };
    (@point_exprs $self:ident, $point_zero:ty) => {
        $crate::protocols::jolt::verifier_plan::point_exprs_with_zeros_from_cpu(
            &$self.point_zeros,
            &$self.point_slices,
            &$self.point_concats,
        )
    };
    (@point_exprs $self:ident) => {
        $crate::protocols::jolt::verifier_plan::point_exprs_from_cpu(
            &$self.point_slices,
            &$self.point_concats,
        )
    };
    (@indexed_eval_families $self:ident, $indexed_eval_families:ident) => {
        &$self.$indexed_eval_families
    };
    (@indexed_eval_families $self:ident) => {
        &[]
    };
    (@relation_output_eval_families $self:ident, $relation_output_eval_families:ident) => {
        &$self.$relation_output_eval_families
    };
    (@relation_output_eval_families $self:ident) => {
        &[]
    };
    (@relation_output_product_families $self:ident, $relation_output_product_families:ident) => {
        &$self.$relation_output_product_families
    };
    (@relation_output_product_families $self:ident) => {
        &[]
    };
    (@relation_output_function_families $self:ident, $relation_output_function_families:ident) => {
        &$self.$relation_output_function_families
    };
    (@relation_output_function_families $self:ident) => {
        &[]
    };
}

pub(crate) use impl_verifier_plan_source_traits;

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

pub(crate) fn field_expr_kind_expr(stage_type_prefix: &str, kind: &FieldExprKind) -> String {
    format!(
        "{stage_type_prefix}FieldExprKind::{}",
        kind.rust_variant_expr()
    )
}

pub(crate) fn scalar_expr_kind_expr(stage_type_prefix: &str, kind: &ScalarExprKind) -> String {
    format!(
        "{stage_type_prefix}ScalarExprKind::{}",
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

pub(crate) fn emit_transcript_absorb_bytes_constants(
    stage_type_prefix: &str,
    const_prefix: &str,
    absorbs: &[VerifierTranscriptAbsorbBytesPlan],
) -> String {
    let absorbs = absorbs
        .iter()
        .map(|absorb| {
            format!(
                "    {stage_type_prefix}TranscriptAbsorbBytesPlan {{ symbol: {}, label: {}, payload: {} }},",
                rust_str(&absorb.symbol),
                rust_str(&absorb.label),
                rust_str(&absorb.payload),
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "pub const {const_prefix}_TRANSCRIPT_ABSORB_BYTES: &[{stage_type_prefix}TranscriptAbsorbBytesPlan] = &[\n{absorbs}\n];\n\n"
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
                field_expr_kind_expr(stage_type_prefix, &expr.kind),
                rust_str_slice_expr(&expr.operands),
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "pub const {const_prefix}_FIELD_EXPRS: &[{stage_type_prefix}FieldExprPlan] = &[\n{exprs}\n];\n"
    )
}

pub(crate) fn emit_scalar_expr_constants(
    stage_type_prefix: &str,
    const_prefix: &str,
    exprs: &[VerifierScalarExprPlan],
) -> String {
    let exprs = exprs
        .iter()
        .map(|expr| {
            format!(
                "    {stage_type_prefix}ScalarExprPlan {{ symbol: {}, kind: {}, operands: {} }},",
                rust_str(&expr.symbol),
                scalar_expr_kind_expr(stage_type_prefix, &expr.kind),
                rust_str_slice_expr(&expr.operands),
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "pub const {const_prefix}_SCALAR_EXPRS: &[{stage_type_prefix}ScalarExprPlan] = &[\n{exprs}\n];\n"
    )
}

pub(crate) fn emit_field_expr_constants_chunked(
    stage_type_prefix: &str,
    const_prefix: &str,
    helper_name: &str,
    exprs: &[VerifierFieldExprPlan],
    chunk_size: usize,
) -> String {
    let rows = exprs
        .chunks(chunk_size)
        .map(|chunk| {
            let exprs = chunk
                .iter()
                .map(|expr| {
                    format!(
                        "{helper_name}({}, {}, {})",
                        rust_str(&expr.symbol),
                        field_expr_kind_expr(stage_type_prefix, &expr.kind),
                        rust_str_slice_expr(&expr.operands)
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!("    {exprs},")
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "const fn {helper_name}(symbol: &'static str, kind: {stage_type_prefix}FieldExprKind, operands: &'static [&'static str]) -> {stage_type_prefix}FieldExprPlan {{\n    {stage_type_prefix}FieldExprPlan {{ symbol, kind, operands }}\n}}\n\n#[rustfmt::skip]\npub const {const_prefix}_FIELD_EXPRS: &[{stage_type_prefix}FieldExprPlan] = &[\n{rows}\n];\n"
    )
}

pub(crate) fn emit_scalar_expr_constants_chunked(
    stage_type_prefix: &str,
    const_prefix: &str,
    helper_name: &str,
    exprs: &[VerifierScalarExprPlan],
    chunk_size: usize,
) -> String {
    let rows = exprs
        .chunks(chunk_size)
        .map(|chunk| {
            let exprs = chunk
                .iter()
                .map(|expr| {
                    format!(
                        "{helper_name}({}, {}, {})",
                        rust_str(&expr.symbol),
                        scalar_expr_kind_expr(stage_type_prefix, &expr.kind),
                        rust_str_slice_expr(&expr.operands)
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!("    {exprs},")
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "const fn {helper_name}(symbol: &'static str, kind: {stage_type_prefix}ScalarExprKind, operands: &'static [&'static str]) -> {stage_type_prefix}ScalarExprPlan {{\n    {stage_type_prefix}ScalarExprPlan {{ symbol, kind, operands }}\n}}\n\n#[rustfmt::skip]\npub const {const_prefix}_SCALAR_EXPRS: &[{stage_type_prefix}ScalarExprPlan] = &[\n{rows}\n];\n"
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

pub(crate) fn emit_sumcheck_batch_constants(
    stage_type_prefix: &str,
    const_prefix: &str,
    batches: &[VerifierSumcheckBatchPlan],
) -> String {
    let mut source = String::new();
    for (index, batch) in batches.iter().enumerate() {
        emit_str_array_if_not_inline(
            &mut source,
            &format!("{const_prefix}_SUMCHECK_BATCH_{index}_CLAIM_OPERANDS"),
            &batch.claim_operands,
            4,
        );
        source.push_str(&emit_usize_array(
            &format!("{const_prefix}_SUMCHECK_BATCH_{index}_ROUND_SCHEDULE"),
            &batch.round_schedule,
        ));
    }
    let batches = batches
        .iter()
        .enumerate()
        .map(|(index, batch)| {
            let claim_operands = str_slice_ref_expr(
                &format!("{const_prefix}_SUMCHECK_BATCH_{index}_CLAIM_OPERANDS"),
                &batch.claim_operands,
                4,
            );
            format!(
                "    {stage_type_prefix}SumcheckBatchPlan {{ symbol: {}, stage: {}, proof_slot: {}, policy: {}, count: {}, claim_operands: {}, claim_label: {}, round_label: {}, round_schedule: {const_prefix}_SUMCHECK_BATCH_{index}_ROUND_SCHEDULE }},",
                rust_str(&batch.symbol),
                rust_str(&batch.stage),
                rust_str(&batch.proof_slot),
                rust_str(&batch.policy),
                batch.count,
                claim_operands,
                rust_str(&batch.claim_label),
                rust_str(&batch.round_label)
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let _ = write!(
        source,
        "pub const {const_prefix}_SUMCHECK_BATCHES: &[{stage_type_prefix}SumcheckBatchPlan] = &[\n{batches}\n];\n"
    );
    source
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

pub(crate) fn point_expr_kind_expr(
    stage_type_prefix: &str,
    kind: &VerifierPointExprKind,
) -> String {
    match kind {
        VerifierPointExprKind::Zero { field, arity } => format!(
            "{stage_type_prefix}PointExprKind::Zero {{ field: {}, arity: {arity} }}",
            rust_str(field)
        ),
        VerifierPointExprKind::Slice { offset, length } => {
            format!(
                "{stage_type_prefix}PointExprKind::Slice {{ offset: {offset}, length: {length} }}"
            )
        }
        VerifierPointExprKind::Concat { layout, arity } => format!(
            "{stage_type_prefix}PointExprKind::Concat {{ layout: {}, arity: {arity} }}",
            rust_str(layout)
        ),
    }
}

pub(crate) fn emit_point_expr_constants(
    stage_type_prefix: &str,
    const_prefix: &str,
    exprs: &[VerifierPointExprPlan],
) -> String {
    let exprs = exprs
        .iter()
        .map(|expr| {
            format!(
                "    {stage_type_prefix}PointExprPlan {{ symbol: {}, kind: {}, operands: {} }},",
                rust_str(&expr.symbol),
                point_expr_kind_expr(stage_type_prefix, &expr.kind),
                rust_str_slice_expr(&expr.operands)
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "pub const {const_prefix}_POINT_EXPRS: &[{stage_type_prefix}PointExprPlan] = &[\n{exprs}\n];\n\n"
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

pub(crate) fn emit_opening_batch_constants(
    stage_type_prefix: &str,
    const_prefix: &str,
    batches: &[VerifierOpeningBatchPlan],
) -> String {
    let batches = batches
        .iter()
        .map(|batch| {
            format!(
                "    {stage_type_prefix}OpeningBatchPlan {{ symbol: {}, stage: {}, proof_slot: {}, policy: {}, count: {}, ordered_claims: {}, claim_operands: {} }},",
                rust_str(&batch.symbol),
                rust_str(&batch.stage),
                rust_str(&batch.proof_slot),
                rust_str(&batch.policy),
                batch.count,
                rust_str_slice_expr(&batch.ordered_claims),
                rust_str_slice_expr(&batch.claim_operands)
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "pub const {const_prefix}_OPENING_BATCHES: &[{stage_type_prefix}OpeningBatchPlan] = &[\n{batches}\n];\n"
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

fn str_slice_ref_expr(name: &str, values: &[String], inline_limit: usize) -> String {
    if values.len() <= inline_limit {
        return rust_str_slice_expr(values);
    }
    name.to_owned()
}

fn emit_str_array_if_not_inline(
    source: &mut String,
    name: &str,
    values: &[String],
    inline_limit: usize,
) {
    if values.len() <= inline_limit {
        return;
    }
    let entries = values
        .iter()
        .map(|value| rust_str(value))
        .collect::<Vec<_>>()
        .join(",\n    ");
    let _ = write!(
        source,
        "pub const {name}: &[&str] = &[\n    {entries},\n];\n\n"
    );
}

fn rust_str(value: &str) -> String {
    format!("{value:?}")
}

fn plan_error(error: RustTargetPlanError) -> EmitError {
    EmitError::new(error.to_string())
}
