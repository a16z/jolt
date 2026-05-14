#![expect(
    clippy::too_many_arguments,
    reason = "generated verifier helpers mirror staged protocol ABIs"
)]

use std::fmt;
use std::marker::PhantomData;

use jolt_field::{Field, Fr, MulPow2, RingCore};
use jolt_poly::{lagrange::lagrange_evals, EqPolynomial};
use jolt_sumcheck::{
    CompressedLabeledRoundPoly, SumcheckClaim, SumcheckError, SumcheckProof, SumcheckVerifier,
};
use jolt_transcript::{Label, Transcript};
use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageParams {
    pub field: &'static str,
    pub pcs: &'static str,
    pub transcript: &'static str,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TypedPlanSymbol<Tag> {
    symbol: &'static str,
    _tag: PhantomData<fn() -> Tag>,
}

impl<Tag> TypedPlanSymbol<Tag> {
    pub const fn new(symbol: &'static str) -> Self {
        Self {
            symbol,
            _tag: PhantomData,
        }
    }

    pub fn as_str(self) -> &'static str {
        self.symbol
    }
}

impl<Tag> fmt::Debug for TypedPlanSymbol<Tag> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("TypedPlanSymbol")
            .field(&self.symbol)
            .finish()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KernelPlan {
    pub symbol: &'static str,
    pub relation: &'static str,
    pub kind: &'static str,
    pub backend: &'static str,
    pub abi: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[rustfmt::skip]
pub enum RelationKind { Stage1OuterUniskip, Stage1OuterRemaining, Stage2ProductVirtualUniskip, Stage2RamReadWrite, Stage2ProductVirtualRemainder, Stage2InstructionLookupClaimReduction, Stage2RamRafEvaluation, Stage2RamOutputCheck, Stage2Batched, Stage3SpartanShift, Stage3InstructionInput, Stage3RegistersClaimReduction, Stage3Batched, Stage4RegistersReadWrite, Stage4RamValCheck, Stage4Batched, Stage5InstructionReadRaf, Stage5RamRaClaimReduction, Stage5RegistersValEvaluation, Stage5Batched, Stage6BytecodeReadRaf, Stage6Booleanity, Stage6HammingBooleanity, Stage6RamRaVirtual, Stage6InstructionRaVirtual, Stage6IncClaimReduction, Stage6Batched, Stage7HammingWeightClaimReduction, Stage7Batched }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TranscriptSqueezeKind {
    ChallengeScalar,
    ChallengeVector,
    Scalar,
}

impl TranscriptSqueezeKind {
    pub fn is_scalar(self) -> bool {
        matches!(self, Self::ChallengeScalar | Self::Scalar)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TranscriptSqueezePlan {
    pub symbol: &'static str,
    pub label: &'static str,
    pub kind: TranscriptSqueezeKind,
    pub count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TranscriptAbsorbBytesPlan {
    pub symbol: &'static str,
    pub label: &'static str,
    pub payload: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProgramStepKind {
    TranscriptSqueeze,
    TranscriptAbsorbBytes,
    SumcheckDriver,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProgramStepPlan {
    pub kind: ProgramStepKind,
    pub symbol: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClaimKind {
    Committed,
    Virtual,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceStage {
    Stage6,
    Stage7,
}

impl SourceStage {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stage6 => "stage6",
            Self::Stage7 => "stage7",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PcsProofMode {
    Open,
    Verify,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpeningInputPlan {
    pub symbol: &'static str,
    pub source_stage: &'static str,
    pub source_claim: &'static str,
    pub oracle: &'static str,
    pub domain: &'static str,
    pub point_arity: usize,
    pub claim_kind: ClaimKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FieldConstantPlan {
    pub symbol: &'static str,
    pub field: &'static str,
    pub value: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldExprKind {
    OpeningEval,
    Add,
    Sub,
    Mul,
    Neg,
    Pow(usize),
    LagrangeBasisEval(i64, usize, usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FieldExprPlan {
    pub symbol: &'static str,
    pub kind: FieldExprKind,
    pub operands: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SumcheckClaimPlan {
    pub symbol: &'static str,
    pub stage: &'static str,
    pub domain: &'static str,
    pub num_rounds: usize,
    pub degree: usize,
    pub claim: &'static str,
    pub kernel: Option<&'static str>,
    pub relation: Option<RelationKind>,
    pub claim_value: &'static str,
    pub input_openings: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SumcheckBatchPlan {
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
pub struct SumcheckDriverPlan {
    pub symbol: &'static str,
    pub stage: &'static str,
    pub proof_slot: &'static str,
    pub kernel: Option<&'static str>,
    pub relation: Option<RelationKind>,
    pub batch: &'static str,
    pub policy: &'static str,
    pub round_schedule: &'static [usize],
    pub claim_label: &'static str,
    pub round_label: &'static str,
    pub num_rounds: usize,
    pub degree: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SumcheckInstanceResultPlan {
    pub symbol: &'static str,
    pub source: &'static str,
    pub claim: &'static str,
    pub relation: RelationKind,
    pub index: usize,
    pub point_arity: usize,
    pub num_rounds: usize,
    pub round_offset: usize,
    pub point_order: &'static str,
    pub degree: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SumcheckEvalPlan {
    pub symbol: &'static str,
    pub source: &'static str,
    pub name: &'static str,
    pub index: usize,
    pub oracle: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PointZeroPlan {
    pub symbol: &'static str,
    pub field: &'static str,
    pub arity: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PointSlicePlan {
    pub symbol: &'static str,
    pub source: &'static str,
    pub offset: usize,
    pub length: usize,
    pub input: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PointConcatPlan {
    pub symbol: &'static str,
    pub layout: &'static str,
    pub arity: usize,
    pub inputs: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpeningClaimPlan {
    pub symbol: &'static str,
    pub oracle: &'static str,
    pub domain: &'static str,
    pub point_arity: usize,
    pub claim_kind: ClaimKind,
    pub point_source: &'static str,
    pub eval_source: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpeningEqualityMode {
    PointAndEval,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpeningClaimEqualityPlan {
    pub symbol: &'static str,
    pub mode: OpeningEqualityMode,
    pub lhs: &'static str,
    pub rhs: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpeningBatchPlan {
    pub symbol: &'static str,
    pub stage: &'static str,
    pub proof_slot: &'static str,
    pub policy: &'static str,
    pub count: usize,
    pub ordered_claims: &'static [&'static str],
    pub claim_operands: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageProgramPlan {
    pub role: &'static str,
    pub params: StageParams,
    pub steps: &'static [ProgramStepPlan],
    pub transcript_squeezes: &'static [TranscriptSqueezePlan],
    pub transcript_absorb_bytes: &'static [TranscriptAbsorbBytesPlan],
    pub opening_inputs: &'static [OpeningInputPlan],
    pub field_constants: &'static [FieldConstantPlan],
    pub field_exprs: &'static [FieldExprPlan],
    pub kernels: &'static [KernelPlan],
    pub claims: &'static [SumcheckClaimPlan],
    pub batches: &'static [SumcheckBatchPlan],
    pub drivers: &'static [SumcheckDriverPlan],
    pub instance_results: &'static [SumcheckInstanceResultPlan],
    pub evals: &'static [SumcheckEvalPlan],
    pub point_zeros: &'static [PointZeroPlan],
    pub point_slices: &'static [PointSlicePlan],
    pub point_concats: &'static [PointConcatPlan],
    pub opening_claims: &'static [OpeningClaimPlan],
    pub opening_equalities: &'static [OpeningClaimEqualityPlan],
    pub opening_batches: &'static [OpeningBatchPlan],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageProgramPlanNoPointZeros {
    pub role: &'static str,
    pub params: StageParams,
    pub steps: &'static [ProgramStepPlan],
    pub transcript_squeezes: &'static [TranscriptSqueezePlan],
    pub transcript_absorb_bytes: &'static [TranscriptAbsorbBytesPlan],
    pub opening_inputs: &'static [OpeningInputPlan],
    pub field_constants: &'static [FieldConstantPlan],
    pub field_exprs: &'static [FieldExprPlan],
    pub kernels: &'static [KernelPlan],
    pub claims: &'static [SumcheckClaimPlan],
    pub batches: &'static [SumcheckBatchPlan],
    pub drivers: &'static [SumcheckDriverPlan],
    pub instance_results: &'static [SumcheckInstanceResultPlan],
    pub evals: &'static [SumcheckEvalPlan],
    pub point_slices: &'static [PointSlicePlan],
    pub point_concats: &'static [PointConcatPlan],
    pub opening_claims: &'static [OpeningClaimPlan],
    pub opening_equalities: &'static [OpeningClaimEqualityPlan],
    pub opening_batches: &'static [OpeningBatchPlan],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageVerifierProgramPlan {
    pub params: StageParams,
    pub steps: &'static [ProgramStepPlan],
    pub transcript_squeezes: &'static [TranscriptSqueezePlan],
    pub opening_inputs: &'static [OpeningInputPlan],
    pub field_constants: &'static [FieldConstantPlan],
    pub field_exprs: &'static [FieldExprPlan],
    pub claims: &'static [SumcheckClaimPlan],
    pub batches: &'static [SumcheckBatchPlan],
    pub drivers: &'static [SumcheckDriverPlan],
    pub instance_results: &'static [SumcheckInstanceResultPlan],
    pub evals: &'static [SumcheckEvalPlan],
    pub point_slices: &'static [PointSlicePlan],
    pub point_concats: &'static [PointConcatPlan],
    pub opening_claims: &'static [OpeningClaimPlan],
    pub opening_equalities: &'static [OpeningClaimEqualityPlan],
    pub opening_batches: &'static [OpeningBatchPlan],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageVerifierProgramPlanNoEqualities {
    pub params: StageParams,
    pub steps: &'static [ProgramStepPlan],
    pub transcript_squeezes: &'static [TranscriptSqueezePlan],
    pub opening_inputs: &'static [OpeningInputPlan],
    pub field_constants: &'static [FieldConstantPlan],
    pub field_exprs: &'static [FieldExprPlan],
    pub claims: &'static [SumcheckClaimPlan],
    pub batches: &'static [SumcheckBatchPlan],
    pub drivers: &'static [SumcheckDriverPlan],
    pub instance_results: &'static [SumcheckInstanceResultPlan],
    pub evals: &'static [SumcheckEvalPlan],
    pub point_slices: &'static [PointSlicePlan],
    pub point_concats: &'static [PointConcatPlan],
    pub opening_claims: &'static [OpeningClaimPlan],
    pub opening_batches: &'static [OpeningBatchPlan],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifierProgramPlanMinimal {
    pub params: StageParams,
    pub transcript_squeezes: &'static [TranscriptSqueezePlan],
    pub claims: &'static [SumcheckClaimPlan],
    pub batches: &'static [SumcheckBatchPlan],
    pub drivers: &'static [SumcheckDriverPlan],
    pub instance_results: &'static [SumcheckInstanceResultPlan],
    pub evals: &'static [SumcheckEvalPlan],
    pub opening_claims: &'static [OpeningClaimPlan],
    pub opening_batches: &'static [OpeningBatchPlan],
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct StageNamedEval<F: Field> {
    pub name: &'static str,
    pub oracle: &'static str,
    pub value: F,
}

#[derive(Clone, Debug, Serialize)]
pub struct StageSumcheckOutput<F: Field> {
    pub driver: &'static str,
    pub point: Vec<F>,
    pub evals: Vec<StageNamedEval<F>>,
    pub proof: SumcheckProof<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageChallengeVector<F: Field> {
    pub symbol: &'static str,
    pub values: Vec<F>,
}

#[derive(Clone, Debug)]
pub struct StageExecutionArtifacts<F: Field> {
    pub challenge_vectors: Vec<StageChallengeVector<F>>,
    pub sumchecks: Vec<StageSumcheckOutput<F>>,
    pub opening_batches: Vec<&'static OpeningBatchPlan>,
}

impl<F: Field> Default for StageExecutionArtifacts<F> {
    fn default() -> Self {
        Self {
            challenge_vectors: Vec::new(),
            sumchecks: Vec::new(),
            opening_batches: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct StageProof<F: Field> {
    pub sumchecks: Vec<StageSumcheckOutput<F>>,
}

#[derive(Clone, Debug)]
pub struct StageOpeningInputValue<F: Field> {
    pub symbol: &'static str,
    pub point: Vec<F>,
    pub eval: F,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimePlanError {
    MissingBatch {
        driver: &'static str,
        batch: &'static str,
    },
    MissingClaim {
        batch: &'static str,
        claim: &'static str,
    },
    MissingValue {
        symbol: &'static str,
    },
    InvalidInputLength {
        input: &'static str,
        expected: usize,
        actual: usize,
    },
    InvalidProof {
        driver: &'static str,
        reason: &'static str,
    },
}

macro_rules! impl_runtime_plan_error_conversion {
    ($error:ident) => {
        impl From<super::common::RuntimePlanError> for $error {
            fn from(error: super::common::RuntimePlanError) -> Self {
                match error {
                    super::common::RuntimePlanError::MissingBatch { driver, batch } => {
                        Self::MissingBatch { driver, batch }
                    }
                    super::common::RuntimePlanError::MissingClaim { batch, claim } => {
                        Self::MissingClaim { batch, claim }
                    }
                    super::common::RuntimePlanError::MissingValue { symbol } => {
                        Self::MissingValue { symbol }
                    }
                    super::common::RuntimePlanError::InvalidInputLength {
                        input,
                        expected,
                        actual,
                    } => Self::InvalidInputLength {
                        input,
                        expected,
                        actual,
                    },
                    super::common::RuntimePlanError::InvalidProof { driver, reason } => {
                        Self::InvalidProof { driver, reason }
                    }
                }
            }
        }
    };
}

pub(crate) use impl_runtime_plan_error_conversion;

pub trait SymbolPlan {
    fn symbol(&self) -> &'static str;
}

impl SymbolPlan for TranscriptSqueezePlan {
    fn symbol(&self) -> &'static str {
        self.symbol
    }
}

impl SymbolPlan for TranscriptAbsorbBytesPlan {
    fn symbol(&self) -> &'static str {
        self.symbol
    }
}

impl SymbolPlan for SumcheckBatchPlan {
    fn symbol(&self) -> &'static str {
        self.symbol
    }
}

impl SymbolPlan for SumcheckClaimPlan {
    fn symbol(&self) -> &'static str {
        self.symbol
    }
}

impl SymbolPlan for SumcheckDriverPlan {
    fn symbol(&self) -> &'static str {
        self.symbol
    }
}

impl SymbolPlan for OpeningClaimPlan {
    fn symbol(&self) -> &'static str {
        self.symbol
    }
}

pub trait SumcheckClaimInfo: SymbolPlan {
    fn num_rounds(&self) -> usize;
    fn claim_value(&self) -> &'static str;
}

impl SumcheckClaimInfo for SumcheckClaimPlan {
    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn claim_value(&self) -> &'static str {
        self.claim_value
    }
}

pub trait SumcheckDriverInfo: SymbolPlan {
    fn batch(&self) -> &'static str;
    fn num_rounds(&self) -> usize;
    fn degree(&self) -> usize;
    fn round_label(&self) -> &'static str;
}

impl SumcheckDriverInfo for SumcheckDriverPlan {
    fn batch(&self) -> &'static str {
        self.batch
    }

    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn degree(&self) -> usize {
        self.degree
    }

    fn round_label(&self) -> &'static str {
        self.round_label
    }
}

#[derive(Clone, Debug, Default)]
pub struct ValueStore<F: Field> {
    scalars: Vec<(&'static str, F)>,
    points: Vec<(&'static str, Vec<F>)>,
}

impl<F: Field> ValueStore<F> {
    pub fn with_opening_inputs(
        inputs: &[StageOpeningInputValue<F>],
        expected_inputs: &[OpeningInputPlan],
    ) -> Result<Self, RuntimePlanError> {
        if inputs.len() != expected_inputs.len() {
            return Err(RuntimePlanError::InvalidInputLength {
                input: "opening_inputs",
                expected: expected_inputs.len(),
                actual: inputs.len(),
            });
        }
        for expected in expected_inputs {
            let matching_count = inputs
                .iter()
                .filter(|input| input.symbol == expected.symbol)
                .count();
            if matching_count != 1 {
                return Err(RuntimePlanError::InvalidInputLength {
                    input: expected.symbol,
                    expected: 1,
                    actual: matching_count,
                });
            }
            if let Some(input) = inputs.iter().find(|input| input.symbol == expected.symbol) {
                if input.point.len() != expected.point_arity {
                    return Err(RuntimePlanError::InvalidInputLength {
                        input: expected.symbol,
                        expected: expected.point_arity,
                        actual: input.point.len(),
                    });
                }
            }
        }
        let mut store = Self::default();
        for input in inputs {
            store.insert_scalar(input.symbol, input.eval);
            store.insert_point(input.symbol, input.point.clone());
        }
        Ok(store)
    }

    pub fn seed_constants(&mut self, constants: &[FieldConstantPlan]) {
        for constant in constants {
            self.insert_scalar(constant.symbol, F::from_u64(constant.value as u64));
        }
    }

    pub fn seed_point_zeros(&mut self, point_zeros: &[PointZeroPlan]) {
        for zero in point_zeros {
            self.insert_point(zero.symbol, vec![F::from_u64(0); zero.arity]);
        }
    }

    pub fn observe_challenge_vector<E>(
        &mut self,
        plan: &TranscriptSqueezePlan,
        values: &[F],
        invalid_input_length: impl Fn(&'static str, usize, usize) -> E,
    ) -> Result<(), E> {
        self.insert_point(plan.symbol, values.to_vec());
        if plan.kind.is_scalar() {
            if values.len() != 1 {
                return Err(invalid_input_length(plan.symbol, 1, values.len()));
            }
            self.insert_scalar(plan.symbol, values[0]);
        }
        Ok(())
    }

    pub fn observe_sumcheck_output<E>(
        &mut self,
        instance_results: &[SumcheckInstanceResultPlan],
        evals: &[SumcheckEvalPlan],
        output: &StageSumcheckOutput<F>,
        normalize_point: impl Fn(&SumcheckInstanceResultPlan, Vec<F>) -> Result<Vec<F>, E>,
        invalid_input_length: impl Fn(&'static str, usize, usize) -> E,
        missing_value: impl Fn(&'static str) -> E,
    ) -> Result<(), E> {
        self.insert_point(output.driver, output.point.clone());
        for instance in instance_results
            .iter()
            .filter(|instance| instance.source == output.driver)
        {
            let end = instance.round_offset + instance.point_arity;
            let point = output
                .point
                .get(instance.round_offset..end)
                .ok_or_else(|| invalid_input_length(instance.symbol, end, output.point.len()))?
                .to_vec();
            self.insert_point(instance.symbol, normalize_point(instance, point)?);
        }
        for eval in evals.iter().filter(|eval| eval.source == output.driver) {
            let value = output
                .evals
                .iter()
                .find(|value| value.name == eval.name)
                .or_else(|| output.evals.get(eval.index))
                .ok_or_else(|| missing_value(eval.symbol))?
                .value;
            self.insert_scalar(eval.symbol, value);
            self.insert_scalar(eval.name, value);
        }
        Ok(())
    }

    pub fn evaluate_available_points<E>(
        &mut self,
        point_slices: &[PointSlicePlan],
        point_concats: &[PointConcatPlan],
        invalid_input_length: impl Fn(&'static str, usize, usize) -> E,
    ) -> Result<(), E> {
        loop {
            let mut progress = 0usize;
            for slice in point_slices {
                if self.try_point(slice.symbol).is_some() {
                    continue;
                }
                let Some(input) = self.try_point(slice.input) else {
                    continue;
                };
                let end = slice.offset + slice.length;
                let point = input
                    .get(slice.offset..end)
                    .ok_or_else(|| invalid_input_length(slice.symbol, end, input.len()))?
                    .to_vec();
                self.insert_point(slice.symbol, point);
                progress += 1;
            }
            for concat in point_concats {
                if self.try_point(concat.symbol).is_some() {
                    continue;
                }
                let Some(point) = self.try_concat_point(concat) else {
                    continue;
                };
                if point.len() != concat.arity {
                    return Err(invalid_input_length(
                        concat.symbol,
                        concat.arity,
                        point.len(),
                    ));
                }
                self.insert_point(concat.symbol, point);
                progress += 1;
            }
            if progress == 0 {
                return Ok(());
            }
        }
    }

    pub fn evaluate_available_field_exprs<E>(
        &mut self,
        field_exprs: &[FieldExprPlan],
        evaluate: impl Fn(&FieldExprPlan, &[F]) -> Result<F, E>,
    ) -> Result<(), E> {
        loop {
            let mut progress = 0usize;
            for expr in field_exprs {
                if self.try_scalar(expr.symbol).is_some() {
                    continue;
                }
                let Some(operands) = self.try_expr_operands(expr) else {
                    continue;
                };
                self.insert_scalar(expr.symbol, evaluate(expr, &operands)?);
                progress += 1;
            }
            if progress == 0 {
                return Ok(());
            }
        }
    }

    pub fn verify_opening_equalities<E>(
        &self,
        opening_equalities: &[OpeningClaimEqualityPlan],
        invalid_proof: impl Fn(&'static str, &'static str) -> E,
        missing_value: impl Fn(&'static str) -> E,
    ) -> Result<(), E> {
        for equality in opening_equalities {
            match equality.mode {
                OpeningEqualityMode::PointAndEval => {
                    if self.point_or(equality.lhs, &missing_value)?
                        != self.point_or(equality.rhs, &missing_value)?
                        || self.scalar_or(equality.lhs, &missing_value)?
                            != self.scalar_or(equality.rhs, &missing_value)?
                    {
                        return Err(invalid_proof(
                            equality.symbol,
                            "opening claim equality failed",
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    pub fn insert_scalar(&mut self, symbol: &'static str, value: F) {
        if let Some((_, existing)) = self.scalars.iter_mut().find(|(name, _)| *name == symbol) {
            *existing = value;
        } else {
            self.scalars.push((symbol, value));
        }
    }

    pub fn insert_point(&mut self, symbol: &'static str, point: Vec<F>) {
        if let Some((_, existing)) = self.points.iter_mut().find(|(name, _)| *name == symbol) {
            *existing = point;
        } else {
            self.points.push((symbol, point));
        }
    }

    pub fn scalar_or<E>(
        &self,
        symbol: &'static str,
        missing_value: impl FnOnce(&'static str) -> E,
    ) -> Result<F, E> {
        self.try_scalar(symbol).ok_or_else(|| missing_value(symbol))
    }

    pub fn try_scalar(&self, symbol: &str) -> Option<F> {
        self.scalars
            .iter()
            .find(|(name, _)| *name == symbol)
            .map(|(_, value)| *value)
    }

    pub fn point_or<E>(
        &self,
        symbol: &'static str,
        missing_value: impl FnOnce(&'static str) -> E,
    ) -> Result<&[F], E> {
        self.try_point(symbol).ok_or_else(|| missing_value(symbol))
    }

    pub fn try_point(&self, symbol: &str) -> Option<&[F]> {
        self.points
            .iter()
            .find(|(name, _)| *name == symbol)
            .map(|(_, point)| point.as_slice())
    }

    fn try_expr_operands(&self, expr: &FieldExprPlan) -> Option<Vec<F>> {
        expr.operands
            .iter()
            .map(|operand| self.try_scalar(operand))
            .collect()
    }

    fn try_concat_point(&self, concat: &PointConcatPlan) -> Option<Vec<F>> {
        let mut point = Vec::with_capacity(concat.arity);
        for input in symbol_list(concat.inputs) {
            point.extend_from_slice(self.try_point(input)?);
        }
        Some(point)
    }
}

pub fn symbol_list(symbols: &'static str) -> impl Iterator<Item = &'static str> {
    symbols.split('|').filter(|symbol| !symbol.is_empty())
}

pub fn find_plan<'a, T: SymbolPlan>(plans: &'a [T], symbol: &str) -> Option<&'a T> {
    plans.iter().find(|plan| plan.symbol() == symbol)
}

pub fn find_batch<'a>(
    batches: &'a [SumcheckBatchPlan],
    driver: &'static str,
    batch: &'static str,
) -> Result<&'a SumcheckBatchPlan, RuntimePlanError> {
    find_plan(batches, batch).ok_or(RuntimePlanError::MissingBatch { driver, batch })
}

pub fn batch_claims<'a, C: SymbolPlan>(
    claims: &'a [C],
    batch: &SumcheckBatchPlan,
) -> Result<Vec<&'a C>, RuntimePlanError> {
    batch
        .claim_operands
        .iter()
        .copied()
        .map(|symbol| {
            find_plan(claims, symbol).ok_or(RuntimePlanError::MissingClaim {
                batch: batch.symbol,
                claim: symbol,
            })
        })
        .collect()
}

pub fn batch_claim_values<C: SumcheckClaimInfo>(
    claims: &[&C],
    field_exprs: &[FieldExprPlan],
    store: &mut ValueStore<Fr>,
) -> Result<Vec<Fr>, RuntimePlanError> {
    claims
        .iter()
        .map(|claim| {
            store.evaluate_available_field_exprs(field_exprs, evaluate_field_expr)?;
            store.scalar_or(claim.claim_value(), |symbol| {
                RuntimePlanError::MissingValue { symbol }
            })
        })
        .collect()
}

pub fn verify_batched_sumcheck<T, E, C, D, Expected, Observe, MapSumcheck>(
    driver: &'static D,
    proof: &StageSumcheckOutput<Fr>,
    claims: &'static [C],
    batches: &'static [SumcheckBatchPlan],
    field_exprs: &'static [FieldExprPlan],
    opening_inputs: &'static [OpeningInputPlan],
    opening_claims: &'static [OpeningClaimPlan],
    opening_batches: &'static [OpeningBatchPlan],
    store: &mut ValueStore<Fr>,
    transcript: &mut T,
    expected_output: Expected,
    observe_output: Observe,
    map_sumcheck: MapSumcheck,
) -> Result<StageSumcheckOutput<Fr>, E>
where
    T: Transcript<Challenge = Fr>,
    E: From<RuntimePlanError>,
    C: SumcheckClaimInfo,
    D: SumcheckDriverInfo,
    Expected: FnOnce(&ValueStore<Fr>, &[StageNamedEval<Fr>], &[Fr], &[Fr]) -> Result<Fr, E>,
    Observe: FnOnce(&mut ValueStore<Fr>, &StageSumcheckOutput<Fr>) -> Result<(), E>,
    MapSumcheck: FnOnce(&'static str, SumcheckError<Fr>) -> E,
{
    if proof.driver != driver.symbol() {
        return Err(RuntimePlanError::InvalidProof {
            driver: driver.symbol(),
            reason: "driver symbol mismatch",
        }
        .into());
    }
    let batch = find_batch(batches, driver.symbol(), driver.batch())?;
    let claims = batch_claims(claims, batch)?;
    let input_claims = batch_claim_values(&claims, field_exprs, store)?;
    for claim in &input_claims {
        append_labeled_scalar(transcript, batch.claim_label, claim);
    }
    let batching_coeffs = transcript.challenge_vector(claims.len());
    let claimed_sum = input_claims
        .iter()
        .zip(claims.iter())
        .zip(&batching_coeffs)
        .map(|((claim, plan), coefficient)| {
            claim.mul_pow_2(driver.num_rounds() - plan.num_rounds()) * *coefficient
        })
        .sum::<Fr>();
    let claim = SumcheckClaim::new(driver.num_rounds(), driver.degree(), claimed_sum);
    let round_proofs = proof
        .proof
        .round_polynomials
        .iter()
        .map(|poly| CompressedLabeledRoundPoly::new(poly, driver.round_label().as_bytes()))
        .collect::<Vec<_>>();
    let output = SumcheckVerifier::verify_optimized(&claim, &round_proofs, transcript)
        .map_err(|error| map_sumcheck(driver.symbol(), error))?;
    if !proof.point.is_empty() && proof.point != output.point {
        return Err(RuntimePlanError::InvalidProof {
            driver: driver.symbol(),
            reason: "batched point mismatch",
        }
        .into());
    }
    let expected = expected_output(store, &proof.evals, &output.point, &batching_coeffs)?;
    if output.value != expected {
        return Err(RuntimePlanError::InvalidProof {
            driver: driver.symbol(),
            reason: "batched output claim mismatch",
        }
        .into());
    }
    let verified = StageSumcheckOutput {
        driver: driver.symbol(),
        point: output.point,
        evals: proof.evals.clone(),
        proof: proof.proof.clone(),
    };
    observe_output(store, &verified)?;
    append_opening_claims(
        opening_inputs,
        opening_claims,
        opening_batches,
        store,
        transcript,
        &verified.evals,
        |batch, claim| RuntimePlanError::MissingClaim { batch, claim },
        |symbol| RuntimePlanError::MissingValue { symbol },
    )?;
    Ok(verified)
}

pub fn eval_by_name<F: Field>(
    evals: &[StageNamedEval<F>],
    name: &'static str,
) -> Result<F, RuntimePlanError> {
    evals
        .iter()
        .find(|eval| eval.name == name)
        .map(|eval| eval.value)
        .ok_or(RuntimePlanError::MissingValue { symbol: name })
}

pub fn indexed_evals_by_prefix<F: Field>(
    evals: &[StageNamedEval<F>],
    prefix: &'static str,
    count: usize,
) -> Result<Vec<F>, RuntimePlanError> {
    let mut values = vec![None; count];
    for eval in evals {
        let Some(suffix) = eval.name.strip_prefix(prefix) else {
            continue;
        };
        let index = suffix
            .parse::<usize>()
            .map_err(|_| RuntimePlanError::InvalidProof {
                driver: prefix,
                reason: "invalid indexed eval suffix",
            })?;
        if index >= count || values[index].is_some() {
            return Err(RuntimePlanError::InvalidProof {
                driver: prefix,
                reason: "invalid indexed eval",
            });
        }
        values[index] = Some(eval.value);
    }
    values
        .into_iter()
        .map(|value| value.ok_or(RuntimePlanError::MissingValue { symbol: prefix }))
        .collect()
}

pub fn indexed_evals_by_prefix_any<F: Field>(
    evals: &[StageNamedEval<F>],
    prefix: &'static str,
) -> Result<Vec<F>, RuntimePlanError> {
    let mut indexed_values = Vec::new();
    for eval in evals {
        let Some(suffix) = eval.name.strip_prefix(prefix) else {
            continue;
        };
        let index = suffix
            .parse::<usize>()
            .map_err(|_| RuntimePlanError::InvalidProof {
                driver: prefix,
                reason: "invalid indexed eval suffix",
            })?;
        if indexed_values
            .iter()
            .any(|(existing_index, _)| *existing_index == index)
        {
            return Err(RuntimePlanError::InvalidProof {
                driver: prefix,
                reason: "duplicate indexed eval",
            });
        }
        indexed_values.push((index, eval.value));
    }
    if indexed_values.is_empty() {
        return Err(RuntimePlanError::MissingValue { symbol: prefix });
    }
    indexed_values.sort_by_key(|(index, _)| *index);
    for (expected, (actual, _)) in indexed_values.iter().enumerate() {
        if *actual != expected {
            return Err(RuntimePlanError::InvalidProof {
                driver: prefix,
                reason: "non-contiguous indexed eval",
            });
        }
    }
    Ok(indexed_values.into_iter().map(|(_, value)| value).collect())
}

pub fn single_operand<F: Field>(
    symbol: &'static str,
    operands: &[F],
) -> Result<F, RuntimePlanError> {
    require_operand_count(symbol, 1, operands.len())?;
    Ok(operands[0])
}

pub fn require_operand_count(
    input: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), RuntimePlanError> {
    if expected == actual {
        Ok(())
    } else {
        Err(RuntimePlanError::InvalidInputLength {
            input,
            expected,
            actual,
        })
    }
}

pub fn evaluate_field_expr<F: Field>(
    expr: &FieldExprPlan,
    operands: &[F],
) -> Result<F, RuntimePlanError> {
    match expr.kind {
        FieldExprKind::OpeningEval => Ok(single_operand(expr.symbol, operands)?),
        FieldExprKind::Add => {
            require_operand_count(expr.symbol, 2, operands.len())?;
            Ok(operands[0] + operands[1])
        }
        FieldExprKind::Sub => {
            require_operand_count(expr.symbol, 2, operands.len())?;
            Ok(operands[0] - operands[1])
        }
        FieldExprKind::Mul => {
            require_operand_count(expr.symbol, 2, operands.len())?;
            Ok(operands[0] * operands[1])
        }
        FieldExprKind::Neg => {
            require_operand_count(expr.symbol, 1, operands.len())?;
            Ok(-operands[0])
        }
        FieldExprKind::Pow(exponent) => {
            require_operand_count(expr.symbol, 1, operands.len())?;
            Ok(pow_field(operands[0], exponent))
        }
        FieldExprKind::LagrangeBasisEval(domain_start, domain_size, index) => {
            require_operand_count(expr.symbol, 1, operands.len())?;
            let weights = lagrange_evals(domain_start, domain_size, operands[0]);
            weights
                .get(index)
                .copied()
                .ok_or(RuntimePlanError::InvalidInputLength {
                    input: expr.symbol,
                    expected: index + 1,
                    actual: weights.len(),
                })
        }
    }
}

pub fn bytecode_gamma_powers(gamma: Fr) -> [Fr; 8] {
    let mut powers = [Fr::from_u64(1); 8];
    for index in 1..powers.len() {
        powers[index] = powers[index - 1] * gamma;
    }
    powers
}

pub fn indexed_boolean_eq(index: usize, point: &[Fr]) -> Fr {
    point
        .iter()
        .enumerate()
        .map(|(bit, value)| {
            if (index >> (point.len() - 1 - bit)) & 1 == 1 {
                *value
            } else {
                Fr::from_u64(1) - *value
            }
        })
        .product()
}

pub fn field_powers(base: Fr, count: usize) -> Vec<Fr> {
    let mut powers = Vec::with_capacity(count);
    let mut power = Fr::from_u64(1);
    for _ in 0..count {
        powers.push(power);
        power *= base;
    }
    powers
}

pub fn prefix_point<'a, F: Field>(
    point: &'a [F],
    length: usize,
    input: &'static str,
) -> Result<&'a [F], RuntimePlanError> {
    point
        .get(..length)
        .filter(|prefix| prefix.len() == length)
        .ok_or(RuntimePlanError::InvalidInputLength {
            input,
            expected: length,
            actual: point.len(),
        })
}

pub fn suffix_point<'a, F: Field>(
    point: &'a [F],
    length: usize,
    input: &'static str,
) -> Result<&'a [F], RuntimePlanError> {
    point
        .get(point.len().saturating_sub(length)..)
        .filter(|suffix| suffix.len() == length)
        .ok_or(RuntimePlanError::InvalidInputLength {
            input,
            expected: length,
            actual: point.len(),
        })
}

pub fn normalize_bytecode_read_raf_point<F: Field>(
    point: &[F],
    log_t: usize,
    input: &'static str,
) -> Result<Vec<F>, RuntimePlanError> {
    let log_k = point
        .len()
        .checked_sub(log_t)
        .ok_or(RuntimePlanError::InvalidInputLength {
            input,
            expected: log_t,
            actual: point.len(),
        })?;
    let mut normalized = point.to_vec();
    normalized[..log_k].reverse();
    normalized[log_k..].reverse();
    Ok(normalized)
}

pub fn normalize_instruction_read_raf_point<F: Field>(
    point: &[F],
    input: &'static str,
) -> Result<Vec<F>, RuntimePlanError> {
    const LOG_K: usize = 128;
    if point.len() < LOG_K {
        return Err(RuntimePlanError::InvalidInputLength {
            input,
            expected: LOG_K,
            actual: point.len(),
        });
    }
    let mut normalized = point.to_vec();
    normalized[LOG_K..].reverse();
    Ok(normalized)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage67RelationSymbols {
    pub hamming_booleanity_instance: &'static str,
    pub booleanity_point: &'static str,
    pub stage5_instruction_ra0: &'static str,
    pub booleanity_combined_point: &'static str,
    pub booleanity_gamma: &'static str,
    pub booleanity_instruction_ra_prefix: &'static str,
    pub booleanity_bytecode_ra_prefix: &'static str,
    pub booleanity_ram_ra_prefix: &'static str,
    pub hamming_weight_eval: &'static str,
    pub hamming_lookup_output: &'static str,
    pub ram_ra_virtual_cycle: &'static str,
    pub ram_ra_virtual_eval_prefix: &'static str,
    pub instruction_ra_virtual_cycle: &'static str,
    pub instruction_ra_virtual_eval_prefix: &'static str,
    pub instruction_ra_virtual_input_prefix: &'static str,
    pub instruction_ra_virtual_gamma: &'static str,
    pub inc_ram_stage2: &'static str,
    pub inc_ram_stage4: &'static str,
    pub inc_rd_stage4: &'static str,
    pub inc_rd_stage5: &'static str,
    pub inc_gamma: &'static str,
    pub inc_ram_eval: &'static str,
    pub inc_rd_eval: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage67BytecodeSymbols {
    pub point: &'static str,
    pub gamma: &'static str,
    pub bytecode_ra_eval_prefix: &'static str,
    pub entries: &'static str,
    pub entry_bytecode_index: &'static str,
    pub stage_gammas: [&'static str; 5],
    pub stage_cycle_points: [&'static str; 5],
    pub stage4_register_point: &'static str,
    pub stage5_register_point: &'static str,
    pub entry_rd: &'static str,
    pub entry_rs1: &'static str,
    pub entry_rs2: &'static str,
    pub entry_lookup_table: &'static str,
}

pub trait Stage67BytecodeEntry {
    fn address(&self) -> Fr;
    fn imm(&self) -> Fr;
    fn circuit_flags(&self) -> &[bool; 14];
    fn rd(&self) -> Option<usize>;
    fn rs1(&self) -> Option<usize>;
    fn rs2(&self) -> Option<usize>;
    fn lookup_table(&self) -> Option<usize>;
    fn is_interleaved(&self) -> bool;
    fn is_branch(&self) -> bool;
    fn left_is_rs1(&self) -> bool;
    fn left_is_pc(&self) -> bool;
    fn right_is_rs2(&self) -> bool;
    fn right_is_imm(&self) -> bool;
    fn is_noop(&self) -> bool;
}

pub fn store_scalar(store: &ValueStore<Fr>, symbol: &'static str) -> Result<Fr, RuntimePlanError> {
    store.scalar_or(symbol, |symbol| RuntimePlanError::MissingValue { symbol })
}

pub fn store_point<'a>(
    store: &'a ValueStore<Fr>,
    symbol: &'static str,
) -> Result<&'a [Fr], RuntimePlanError> {
    store.point_or(symbol, |symbol| RuntimePlanError::MissingValue { symbol })
}

pub fn stage67_trace_rounds(
    instance_results: &[SumcheckInstanceResultPlan],
    symbols: &Stage67RelationSymbols,
) -> Result<usize, RuntimePlanError> {
    instance_results
        .iter()
        .find(|instance| instance.relation == RelationKind::Stage6HammingBooleanity)
        .map(|instance| instance.num_rounds)
        .ok_or(RuntimePlanError::MissingValue {
            symbol: symbols.hamming_booleanity_instance,
        })
}

pub fn expected_stage67_bytecode_read_raf<E: Stage67BytecodeEntry>(
    entries: &[E],
    entry_bytecode_index: usize,
    num_lookup_tables: usize,
    store: &ValueStore<Fr>,
    evals: &[StageNamedEval<Fr>],
    local_point: &[Fr],
    log_t: usize,
    symbols: &Stage67BytecodeSymbols,
) -> Result<Fr, RuntimePlanError> {
    let opening_point = normalize_bytecode_read_raf_point(local_point, log_t, symbols.point)?;
    let log_k = opening_point.len() - log_t;
    let (r_address_prime, r_cycle_prime) = opening_point.split_at(log_k);

    let gamma = store_scalar(store, symbols.gamma)?;
    let gamma_powers = bytecode_gamma_powers(gamma);
    let int_eval = identity_polynomial_eval(r_address_prime);
    let stage_value_evals = stage67_bytecode_stage_value_evals(
        entries,
        entry_bytecode_index,
        num_lookup_tables,
        store,
        r_address_prime,
        r_cycle_prime.len(),
        symbols,
    )?;
    let stage_cycle_points =
        stage67_bytecode_stage_cycle_points(store, r_cycle_prime.len(), symbols)?;
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
        .map(|index| Fr::from_u64(((entry_bytecode_index >> (log_k - 1 - index)) & 1) as u64))
        .collect::<Vec<_>>();
    let zero_cycle = vec![Fr::from_u64(0); r_cycle_prime.len()];
    let entry_contrib = gamma_powers[7]
        * EqPolynomial::<Fr>::mle(&entry_bits, r_address_prime)
        * EqPolynomial::<Fr>::mle(&zero_cycle, r_cycle_prime);
    let bytecode_ra = indexed_evals_by_prefix_any(evals, symbols.bytecode_ra_eval_prefix)?
        .into_iter()
        .product::<Fr>();
    Ok((val + entry_contrib) * bytecode_ra)
}

pub fn expected_stage67_booleanity(
    store: &ValueStore<Fr>,
    evals: &[StageNamedEval<Fr>],
    local_point: &[Fr],
    log_t: usize,
    symbols: &Stage67RelationSymbols,
) -> Result<Fr, RuntimePlanError> {
    let log_k_chunk =
        local_point
            .len()
            .checked_sub(log_t)
            .ok_or(RuntimePlanError::InvalidInputLength {
                input: symbols.booleanity_point,
                expected: log_t,
                actual: local_point.len(),
            })?;
    let stage5_point = store_point(store, symbols.stage5_instruction_ra0)?;
    let stage5_address_len =
        stage5_point
            .len()
            .checked_sub(log_t)
            .ok_or(RuntimePlanError::InvalidInputLength {
                input: symbols.stage5_instruction_ra0,
                expected: log_t,
                actual: stage5_point.len(),
            })?;
    if stage5_address_len < log_k_chunk {
        return Err(RuntimePlanError::InvalidInputLength {
            input: symbols.stage5_instruction_ra0,
            expected: log_k_chunk + log_t,
            actual: stage5_point.len(),
        });
    }

    let mut stage5_addr = stage5_point[..stage5_address_len].to_vec();
    stage5_addr.reverse();
    let mut combined_r = stage5_addr[stage5_address_len - log_k_chunk..].to_vec();
    combined_r.extend(stage5_point[stage5_address_len..].iter().rev().copied());
    if combined_r.len() != local_point.len() {
        return Err(RuntimePlanError::InvalidInputLength {
            input: symbols.booleanity_combined_point,
            expected: local_point.len(),
            actual: combined_r.len(),
        });
    }
    let mut verifier_point = combined_r[..log_k_chunk].to_vec();
    verifier_point.reverse();
    verifier_point.extend(combined_r[log_k_chunk..].iter().rev().copied());
    let eq_eval = EqPolynomial::<Fr>::mle(local_point, &verifier_point);

    let gamma = store_scalar(store, symbols.booleanity_gamma)?;
    let gamma_sq = gamma.square();
    let mut gamma_power = Fr::from_u64(1);
    let mut booleanity = Fr::from_u64(0);
    for ra in stage67_booleanity_evals(evals, symbols)? {
        booleanity += gamma_power * (ra.square() - ra);
        gamma_power *= gamma_sq;
    }
    Ok(eq_eval * booleanity)
}

pub fn expected_stage67_hamming_booleanity(
    store: &ValueStore<Fr>,
    evals: &[StageNamedEval<Fr>],
    local_point: &[Fr],
    symbols: &Stage67RelationSymbols,
) -> Result<Fr, RuntimePlanError> {
    let hamming = eval_by_name(evals, symbols.hamming_weight_eval)?;
    let lookup_output_point = reverse_slice(store_point(store, symbols.hamming_lookup_output)?);
    if lookup_output_point.len() != local_point.len() {
        return Err(RuntimePlanError::InvalidInputLength {
            input: symbols.hamming_lookup_output,
            expected: local_point.len(),
            actual: lookup_output_point.len(),
        });
    }
    let eq_eval = EqPolynomial::<Fr>::mle(local_point, &lookup_output_point);
    Ok((hamming.square() - hamming) * eq_eval)
}

pub fn expected_stage67_ram_ra_virtual(
    store: &ValueStore<Fr>,
    evals: &[StageNamedEval<Fr>],
    local_point: &[Fr],
    symbols: &Stage67RelationSymbols,
) -> Result<Fr, RuntimePlanError> {
    let r_cycle_reduced = reverse_slice(local_point);
    let r_cycle = suffix_point(
        store_point(store, symbols.ram_ra_virtual_cycle)?,
        r_cycle_reduced.len(),
        symbols.ram_ra_virtual_cycle,
    )?;
    let eq_eval = EqPolynomial::<Fr>::mle(r_cycle, &r_cycle_reduced);
    let ram_ra = indexed_evals_by_prefix_any(evals, symbols.ram_ra_virtual_eval_prefix)?
        .into_iter()
        .product::<Fr>();
    Ok(eq_eval * ram_ra)
}

pub fn expected_stage67_instruction_ra_virtual(
    opening_inputs: &[OpeningInputPlan],
    store: &ValueStore<Fr>,
    evals: &[StageNamedEval<Fr>],
    local_point: &[Fr],
    symbols: &Stage67RelationSymbols,
) -> Result<Fr, RuntimePlanError> {
    let r_cycle_reduced = reverse_slice(local_point);
    let r_cycle = suffix_point(
        store_point(store, symbols.instruction_ra_virtual_cycle)?,
        r_cycle_reduced.len(),
        symbols.instruction_ra_virtual_cycle,
    )?;
    let eq_eval = EqPolynomial::<Fr>::mle(r_cycle, &r_cycle_reduced);
    let committed_ra =
        indexed_evals_by_prefix_any(evals, symbols.instruction_ra_virtual_eval_prefix)?;
    let virtual_count = opening_inputs
        .iter()
        .filter(|input| {
            input
                .symbol
                .starts_with(symbols.instruction_ra_virtual_input_prefix)
        })
        .count();
    if virtual_count == 0 || committed_ra.len() % virtual_count != 0 {
        return Err(RuntimePlanError::InvalidInputLength {
            input: symbols.instruction_ra_virtual_eval_prefix,
            expected: virtual_count,
            actual: committed_ra.len(),
        });
    }
    let committed_per_virtual = committed_ra.len() / virtual_count;
    let gamma = store_scalar(store, symbols.instruction_ra_virtual_gamma)?;
    let mut gamma_power = Fr::from_u64(1);
    let mut value = Fr::from_u64(0);
    for chunk in committed_ra.chunks(committed_per_virtual) {
        value += gamma_power * chunk.iter().copied().product::<Fr>();
        gamma_power *= gamma;
    }
    Ok(eq_eval * value)
}

pub fn expected_stage67_inc_claim_reduction(
    store: &ValueStore<Fr>,
    evals: &[StageNamedEval<Fr>],
    local_point: &[Fr],
    symbols: &Stage67RelationSymbols,
) -> Result<Fr, RuntimePlanError> {
    let r_cycle_reduced = reverse_slice(local_point);
    let ram_inc_stage2 = suffix_point(
        store_point(store, symbols.inc_ram_stage2)?,
        r_cycle_reduced.len(),
        symbols.inc_ram_stage2,
    )?;
    let ram_inc_stage4 = suffix_point(
        store_point(store, symbols.inc_ram_stage4)?,
        r_cycle_reduced.len(),
        symbols.inc_ram_stage4,
    )?;
    let rd_inc_stage4 = suffix_point(
        store_point(store, symbols.inc_rd_stage4)?,
        r_cycle_reduced.len(),
        symbols.inc_rd_stage4,
    )?;
    let rd_inc_stage5 = suffix_point(
        store_point(store, symbols.inc_rd_stage5)?,
        r_cycle_reduced.len(),
        symbols.inc_rd_stage5,
    )?;
    let gamma = store_scalar(store, symbols.inc_gamma)?;
    let eq_ram_combined = EqPolynomial::<Fr>::mle(ram_inc_stage2, &r_cycle_reduced)
        + gamma * EqPolynomial::<Fr>::mle(ram_inc_stage4, &r_cycle_reduced);
    let eq_rd_combined = EqPolynomial::<Fr>::mle(rd_inc_stage4, &r_cycle_reduced)
        + gamma * EqPolynomial::<Fr>::mle(rd_inc_stage5, &r_cycle_reduced);
    let ram_inc = eval_by_name(evals, symbols.inc_ram_eval)?;
    let rd_inc = eval_by_name(evals, symbols.inc_rd_eval)?;
    Ok(ram_inc * eq_ram_combined + gamma.square() * rd_inc * eq_rd_combined)
}

fn stage67_booleanity_evals(
    evals: &[StageNamedEval<Fr>],
    symbols: &Stage67RelationSymbols,
) -> Result<Vec<Fr>, RuntimePlanError> {
    let mut values = indexed_evals_by_prefix_any(evals, symbols.booleanity_instruction_ra_prefix)?;
    values.extend(indexed_evals_by_prefix_any(
        evals,
        symbols.booleanity_bytecode_ra_prefix,
    )?);
    values.extend(indexed_evals_by_prefix_any(
        evals,
        symbols.booleanity_ram_ra_prefix,
    )?);
    Ok(values)
}

fn stage67_bytecode_stage_cycle_points(
    store: &ValueStore<Fr>,
    log_t: usize,
    symbols: &Stage67BytecodeSymbols,
) -> Result<[Vec<Fr>; 5], RuntimePlanError> {
    let point = |index| {
        let symbol = symbols.stage_cycle_points[index];
        suffix_point(store_point(store, symbol)?, log_t, symbol).map(|point| point.to_vec())
    };
    Ok([point(0)?, point(1)?, point(2)?, point(3)?, point(4)?])
}

fn stage67_bytecode_stage_value_evals<E: Stage67BytecodeEntry>(
    entries: &[E],
    entry_bytecode_index: usize,
    num_lookup_tables: usize,
    store: &ValueStore<Fr>,
    r_address: &[Fr],
    log_t: usize,
    symbols: &Stage67BytecodeSymbols,
) -> Result<[Fr; 5], RuntimePlanError> {
    let expected_len =
        1usize
            .checked_shl(r_address.len() as u32)
            .ok_or(RuntimePlanError::InvalidInputLength {
                input: symbols.entries,
                expected: usize::BITS as usize,
                actual: r_address.len(),
            })?;
    if entries.len() != expected_len {
        return Err(RuntimePlanError::InvalidInputLength {
            input: symbols.entries,
            expected: expected_len,
            actual: entries.len(),
        });
    }
    if entry_bytecode_index >= expected_len {
        return Err(RuntimePlanError::InvalidInputLength {
            input: symbols.entry_bytecode_index,
            expected: expected_len,
            actual: entry_bytecode_index + 1,
        });
    }

    let stage1_gamma_powers = field_powers(store_scalar(store, symbols.stage_gammas[0])?, 16);
    let stage2_gamma_powers = field_powers(store_scalar(store, symbols.stage_gammas[1])?, 4);
    let stage3_gamma_powers = field_powers(store_scalar(store, symbols.stage_gammas[2])?, 9);
    let stage4_gamma_powers = field_powers(store_scalar(store, symbols.stage_gammas[3])?, 3);
    let stage5_gamma_powers = field_powers(
        store_scalar(store, symbols.stage_gammas[4])?,
        num_lookup_tables + 2,
    );

    let stage4_register_point =
        stage67_register_prefix_point(store, symbols.stage4_register_point, log_t)?;
    let stage5_register_point =
        stage67_register_prefix_point(store, symbols.stage5_register_point, log_t)?;

    let mut evals = [Fr::from_u64(0); 5];
    for (index, entry) in entries.iter().enumerate() {
        let eq = indexed_boolean_eq(index, r_address);
        let values = stage67_bytecode_entry_stage_values(
            entry,
            num_lookup_tables,
            stage4_register_point,
            stage5_register_point,
            &stage1_gamma_powers,
            &stage2_gamma_powers,
            &stage3_gamma_powers,
            &stage4_gamma_powers,
            &stage5_gamma_powers,
            symbols,
        )?;
        for stage in 0..evals.len() {
            evals[stage] += eq * values[stage];
        }
    }
    Ok(evals)
}

fn stage67_bytecode_entry_stage_values<E: Stage67BytecodeEntry>(
    entry: &E,
    num_lookup_tables: usize,
    stage4_register_point: &[Fr],
    stage5_register_point: &[Fr],
    stage1_gamma_powers: &[Fr],
    stage2_gamma_powers: &[Fr],
    stage3_gamma_powers: &[Fr],
    stage4_gamma_powers: &[Fr],
    stage5_gamma_powers: &[Fr],
    symbols: &Stage67BytecodeSymbols,
) -> Result<[Fr; 5], RuntimePlanError> {
    let flags = entry.circuit_flags();
    let mut stage1 = entry.address() + entry.imm() * stage1_gamma_powers[1];
    for (flag, gamma) in flags.iter().zip(stage1_gamma_powers.iter().skip(2)) {
        if *flag {
            stage1 += *gamma;
        }
    }

    let mut stage2 = Fr::from_u64(0);
    if flags[5] {
        stage2 += stage2_gamma_powers[0];
    }
    if entry.is_branch() {
        stage2 += stage2_gamma_powers[1];
    }
    if flags[6] {
        stage2 += stage2_gamma_powers[2];
    }
    if flags[7] {
        stage2 += stage2_gamma_powers[3];
    }

    let mut stage3 = entry.imm() + entry.address() * stage3_gamma_powers[1];
    if entry.left_is_rs1() {
        stage3 += stage3_gamma_powers[2];
    }
    if entry.left_is_pc() {
        stage3 += stage3_gamma_powers[3];
    }
    if entry.right_is_rs2() {
        stage3 += stage3_gamma_powers[4];
    }
    if entry.right_is_imm() {
        stage3 += stage3_gamma_powers[5];
    }
    if entry.is_noop() {
        stage3 += stage3_gamma_powers[6];
    }
    if flags[7] {
        stage3 += stage3_gamma_powers[7];
    }
    if flags[12] {
        stage3 += stage3_gamma_powers[8];
    }

    let stage4 = stage67_register_eq(entry.rd(), stage4_register_point, symbols.entry_rd)?
        * stage4_gamma_powers[0]
        + stage67_register_eq(entry.rs1(), stage4_register_point, symbols.entry_rs1)?
            * stage4_gamma_powers[1]
        + stage67_register_eq(entry.rs2(), stage4_register_point, symbols.entry_rs2)?
            * stage4_gamma_powers[2];

    let mut stage5 = stage67_register_eq(entry.rd(), stage5_register_point, symbols.entry_rd)?
        * stage5_gamma_powers[0];
    if !entry.is_interleaved() {
        stage5 += stage5_gamma_powers[1];
    }
    if let Some(table) = entry.lookup_table() {
        if table >= num_lookup_tables {
            return Err(RuntimePlanError::InvalidInputLength {
                input: symbols.entry_lookup_table,
                expected: num_lookup_tables,
                actual: table + 1,
            });
        }
        stage5 += stage5_gamma_powers[2 + table];
    }

    Ok([stage1, stage2, stage3, stage4, stage5])
}

fn stage67_register_eq(
    index: Option<usize>,
    point: &[Fr],
    input: &'static str,
) -> Result<Fr, RuntimePlanError> {
    let Some(index) = index else {
        return Ok(Fr::from_u64(0));
    };
    let register_count =
        1usize
            .checked_shl(point.len() as u32)
            .ok_or(RuntimePlanError::InvalidInputLength {
                input,
                expected: usize::BITS as usize,
                actual: point.len(),
            })?;
    if index >= register_count {
        return Err(RuntimePlanError::InvalidInputLength {
            input,
            expected: register_count,
            actual: index + 1,
        });
    }
    Ok(indexed_boolean_eq(index, point))
}

fn stage67_register_prefix_point<'a>(
    store: &'a ValueStore<Fr>,
    symbol: &'static str,
    log_t: usize,
) -> Result<&'a [Fr], RuntimePlanError> {
    let point = store_point(store, symbol)?;
    let register_len =
        point
            .len()
            .checked_sub(log_t)
            .ok_or(RuntimePlanError::InvalidInputLength {
                input: symbol,
                expected: log_t,
                actual: point.len(),
            })?;
    prefix_point(point, register_len, symbol)
}

pub fn operand_polynomial_eval(point: &[Fr], left: bool) -> Fr {
    let stride_offset = usize::from(!left);
    let operand_bits = point.len() / 2;
    (0..operand_bits)
        .map(|index| point[2 * index + stride_offset].mul_pow_2(operand_bits - 1 - index))
        .sum()
}

pub fn identity_polynomial_eval(point: &[Fr]) -> Fr {
    point
        .iter()
        .enumerate()
        .map(|(index, value)| value.mul_pow_2(point.len() - 1 - index))
        .sum()
}

pub fn append_labeled_scalar<T>(transcript: &mut T, label: &'static str, scalar: &Fr)
where
    T: Transcript<Challenge = Fr>,
{
    transcript.append(&Label(label.as_bytes()));
    transcript.append(scalar);
}

pub fn append_opening_claims<T, E>(
    opening_inputs: &[OpeningInputPlan],
    opening_claims: &[OpeningClaimPlan],
    opening_batches: &[OpeningBatchPlan],
    store: &mut ValueStore<Fr>,
    transcript: &mut T,
    evals: &[StageNamedEval<Fr>],
    missing_claim: impl Fn(&'static str, &'static str) -> E,
    missing_value: impl Fn(&'static str) -> E,
) -> Result<(), E>
where
    T: Transcript<Challenge = Fr>,
{
    if opening_batches.is_empty() {
        for eval in evals {
            append_labeled_scalar(transcript, "opening_claim", &eval.value);
        }
        return Ok(());
    }
    let mut seen = opening_inputs
        .iter()
        .filter_map(|input| {
            store
                .try_point(input.symbol)
                .map(|point| (input.claim_kind, input.oracle, point.to_vec()))
        })
        .collect::<Vec<_>>();
    for batch in opening_batches {
        for &symbol in batch.claim_operands {
            let claim = opening_claims
                .iter()
                .find(|claim| claim.symbol == symbol)
                .ok_or_else(|| missing_claim(batch.symbol, symbol))?;
            let point = store.point_or(claim.point_source, &missing_value)?.to_vec();
            if seen.iter().any(|(kind, oracle, seen_point)| {
                *kind == claim.claim_kind && *oracle == claim.oracle && seen_point == &point
            }) {
                continue;
            }
            let value = store.scalar_or(claim.eval_source, &missing_value)?;
            append_labeled_scalar(transcript, "opening_claim", &value);
            seen.push((claim.claim_kind, claim.oracle, point));
        }
    }
    Ok(())
}

pub fn lt_polynomial_eval(x: &[Fr], y: &[Fr]) -> Fr {
    let mut lt_eval = Fr::from_u64(0);
    let mut eq_term = Fr::from_u64(1);
    for (x_i, y_i) in x.iter().zip(y.iter()) {
        lt_eval += (Fr::from_u64(1) - *x_i) * *y_i * eq_term;
        eq_term *= Fr::from_u64(1) - *x_i - *y_i + *x_i * *y_i + *x_i * *y_i;
    }
    lt_eval
}

pub fn pow_field<F: Field>(base: F, mut exponent: usize) -> F {
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

pub fn reverse_slice(values: &[Fr]) -> Vec<Fr> {
    values.iter().rev().copied().collect()
}
