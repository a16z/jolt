#![expect(
    clippy::too_many_arguments,
    reason = "generated verifier helpers mirror staged protocol ABIs"
)]

use std::collections::BTreeMap;
use std::fmt;
use std::marker::PhantomData;

use jolt_field::{Field, Fr, MulPow2};
use jolt_poly::{lagrange::lagrange_evals, EqPlusOnePolynomial, EqPolynomial};
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

pub trait ProtocolRelation: Copy + Eq + fmt::Debug + 'static {}

impl<T: Copy + Eq + fmt::Debug + 'static> ProtocolRelation for T {}

pub trait VerifierProgramSlot: Copy + Eq + fmt::Debug + 'static {}

impl<T: Copy + Eq + fmt::Debug + 'static> VerifierProgramSlot for T {}

pub trait VerifierProgramCheckpoint: Copy + Eq + fmt::Debug + 'static {}

impl<T: Copy + Eq + fmt::Debug + 'static> VerifierProgramCheckpoint for T {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerifierProgramStepKind {
    ReceiveCommitments,
    VerifySumcheckStage,
    VerifyPcsOpening,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifierProgramStepPlan<S: VerifierProgramSlot> {
    pub kind: VerifierProgramStepKind,
    pub slot: S,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerifierEvaluationPolicy {
    Skip,
    VerifyIfPresent,
    Required,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifierTarget<C: VerifierProgramCheckpoint> {
    pub checkpoint: C,
    pub evaluation: VerifierEvaluationPolicy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifierTargetPlan<C: VerifierProgramCheckpoint> {
    pub target: VerifierTarget<C>,
    pub step_count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifierProgramPlan<S: VerifierProgramSlot, C: VerifierProgramCheckpoint> {
    pub steps: &'static [VerifierProgramStepPlan<S>],
    pub targets: &'static [VerifierTargetPlan<C>],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerifierProgramError<S: VerifierProgramSlot, T: Copy + Eq + fmt::Debug + 'static> {
    MissingTarget {
        target: T,
    },
    InvalidStepCount {
        target: T,
        step_count: usize,
        total_steps: usize,
    },
    MissingArtifact {
        slot: S,
    },
    UnsupportedStep {
        step: VerifierProgramStepPlan<S>,
        reason: &'static str,
    },
}

impl<S: VerifierProgramSlot, C: VerifierProgramCheckpoint> VerifierProgramPlan<S, C> {
    pub fn steps_for(
        &self,
        target: VerifierTarget<C>,
    ) -> Result<&'static [VerifierProgramStepPlan<S>], VerifierProgramError<S, VerifierTarget<C>>>
    {
        let target_plan = self
            .targets
            .iter()
            .find(|plan| plan.target == target)
            .ok_or(VerifierProgramError::MissingTarget { target })?;
        let step_count = target_plan.step_count;
        if step_count > self.steps.len() {
            return Err(VerifierProgramError::InvalidStepCount {
                target,
                step_count,
                total_steps: self.steps.len(),
            });
        }
        Ok(&self.steps[..step_count])
    }
}

pub fn execute_verifier_program<S, C, E>(
    program: &VerifierProgramPlan<S, C>,
    target: VerifierTarget<C>,
    mut execute_step: impl FnMut(VerifierProgramStepPlan<S>) -> Result<(), E>,
) -> Result<(), E>
where
    S: VerifierProgramSlot,
    C: VerifierProgramCheckpoint,
    E: From<VerifierProgramError<S, VerifierTarget<C>>>,
{
    for step in program.steps_for(target).map_err(E::from)? {
        execute_step(*step)?;
    }
    Ok(())
}

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
pub struct SumcheckClaimPlan<R: ProtocolRelation> {
    pub symbol: &'static str,
    pub stage: &'static str,
    pub domain: &'static str,
    pub num_rounds: usize,
    pub degree: usize,
    pub claim: &'static str,
    pub kernel: Option<&'static str>,
    pub relation: Option<R>,
    pub claim_value: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SumcheckBatchPlan {
    pub symbol: &'static str,
    pub stage: &'static str,
    pub proof_slot: &'static str,
    pub policy: &'static str,
    pub count: usize,
    pub claim_operands: &'static [&'static str],
    pub claim_label: &'static str,
    pub round_label: &'static str,
    pub round_schedule: &'static [usize],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SumcheckDriverPlan<R: ProtocolRelation> {
    pub symbol: &'static str,
    pub stage: &'static str,
    pub proof_slot: &'static str,
    pub kernel: Option<&'static str>,
    pub relation: Option<R>,
    pub batch: &'static str,
    pub policy: &'static str,
    pub round_schedule: &'static [usize],
    pub claim_label: &'static str,
    pub round_label: &'static str,
    pub num_rounds: usize,
    pub degree: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SumcheckInstanceResultPlan<R: ProtocolRelation> {
    pub symbol: &'static str,
    pub source: &'static str,
    pub claim: &'static str,
    pub relation: R,
    pub index: usize,
    pub point_arity: usize,
    pub num_rounds: usize,
    pub round_offset: usize,
    pub point_order: &'static str,
    pub degree: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SumcheckOutputPointOrder {
    AsIs,
    Reverse,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SumcheckOutputPointSegment {
    Full,
    Prefix,
    Suffix,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SumcheckOutputPointLength {
    Full,
    LocalPoint,
    OpeningPoint,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SumcheckOutputPointPlan {
    pub source: &'static str,
    pub segment: SumcheckOutputPointSegment,
    pub length: SumcheckOutputPointLength,
    pub order: SumcheckOutputPointOrder,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SumcheckOutputValueKind {
    EqMle,
    EqPlusOne,
    Lt,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SumcheckOutputValuePlan {
    pub symbol: &'static str,
    pub kind: SumcheckOutputValueKind,
    pub local_point: SumcheckOutputPointPlan,
    pub opening_point: SumcheckOutputPointPlan,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SumcheckOutputClaimPlan<R: ProtocolRelation> {
    pub relation: R,
    pub local_values: &'static [SumcheckOutputValuePlan],
    pub claim_value: &'static str,
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
    pub inputs: &'static [&'static str],
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
pub struct StageProgramPlan<R: ProtocolRelation> {
    pub role: &'static str,
    pub params: StageParams,
    pub steps: &'static [ProgramStepPlan],
    pub transcript_squeezes: &'static [TranscriptSqueezePlan],
    pub transcript_absorb_bytes: &'static [TranscriptAbsorbBytesPlan],
    pub opening_inputs: &'static [OpeningInputPlan],
    pub field_constants: &'static [FieldConstantPlan],
    pub field_exprs: &'static [FieldExprPlan],
    pub kernels: &'static [KernelPlan],
    pub claims: &'static [SumcheckClaimPlan<R>],
    pub batches: &'static [SumcheckBatchPlan],
    pub drivers: &'static [SumcheckDriverPlan<R>],
    pub instance_results: &'static [SumcheckInstanceResultPlan<R>],
    pub evals: &'static [SumcheckEvalPlan],
    pub point_zeros: &'static [PointZeroPlan],
    pub point_slices: &'static [PointSlicePlan],
    pub point_concats: &'static [PointConcatPlan],
    pub opening_claims: &'static [OpeningClaimPlan],
    pub opening_equalities: &'static [OpeningClaimEqualityPlan],
    pub opening_batches: &'static [OpeningBatchPlan],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageProgramPlanNoPointZeros<R: ProtocolRelation> {
    pub role: &'static str,
    pub params: StageParams,
    pub steps: &'static [ProgramStepPlan],
    pub transcript_squeezes: &'static [TranscriptSqueezePlan],
    pub transcript_absorb_bytes: &'static [TranscriptAbsorbBytesPlan],
    pub opening_inputs: &'static [OpeningInputPlan],
    pub field_constants: &'static [FieldConstantPlan],
    pub field_exprs: &'static [FieldExprPlan],
    pub kernels: &'static [KernelPlan],
    pub claims: &'static [SumcheckClaimPlan<R>],
    pub batches: &'static [SumcheckBatchPlan],
    pub drivers: &'static [SumcheckDriverPlan<R>],
    pub instance_results: &'static [SumcheckInstanceResultPlan<R>],
    pub evals: &'static [SumcheckEvalPlan],
    pub output_claims: &'static [SumcheckOutputClaimPlan<R>],
    pub point_slices: &'static [PointSlicePlan],
    pub point_concats: &'static [PointConcatPlan],
    pub opening_claims: &'static [OpeningClaimPlan],
    pub opening_equalities: &'static [OpeningClaimEqualityPlan],
    pub opening_batches: &'static [OpeningBatchPlan],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageVerifierProgramPlan<R: ProtocolRelation> {
    pub params: StageParams,
    pub steps: &'static [ProgramStepPlan],
    pub transcript_squeezes: &'static [TranscriptSqueezePlan],
    pub opening_inputs: &'static [OpeningInputPlan],
    pub field_constants: &'static [FieldConstantPlan],
    pub field_exprs: &'static [FieldExprPlan],
    pub claims: &'static [SumcheckClaimPlan<R>],
    pub batches: &'static [SumcheckBatchPlan],
    pub drivers: &'static [SumcheckDriverPlan<R>],
    pub instance_results: &'static [SumcheckInstanceResultPlan<R>],
    pub evals: &'static [SumcheckEvalPlan],
    pub output_claims: &'static [SumcheckOutputClaimPlan<R>],
    pub point_slices: &'static [PointSlicePlan],
    pub point_concats: &'static [PointConcatPlan],
    pub opening_claims: &'static [OpeningClaimPlan],
    pub opening_equalities: &'static [OpeningClaimEqualityPlan],
    pub opening_batches: &'static [OpeningBatchPlan],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageVerifierProgramPlanNoEqualities<R: ProtocolRelation> {
    pub params: StageParams,
    pub steps: &'static [ProgramStepPlan],
    pub transcript_squeezes: &'static [TranscriptSqueezePlan],
    pub opening_inputs: &'static [OpeningInputPlan],
    pub field_constants: &'static [FieldConstantPlan],
    pub field_exprs: &'static [FieldExprPlan],
    pub claims: &'static [SumcheckClaimPlan<R>],
    pub batches: &'static [SumcheckBatchPlan],
    pub drivers: &'static [SumcheckDriverPlan<R>],
    pub instance_results: &'static [SumcheckInstanceResultPlan<R>],
    pub evals: &'static [SumcheckEvalPlan],
    pub point_slices: &'static [PointSlicePlan],
    pub point_concats: &'static [PointConcatPlan],
    pub opening_claims: &'static [OpeningClaimPlan],
    pub opening_batches: &'static [OpeningBatchPlan],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifierProgramPlanMinimal<R: ProtocolRelation> {
    pub params: StageParams,
    pub transcript_squeezes: &'static [TranscriptSqueezePlan],
    pub claims: &'static [SumcheckClaimPlan<R>],
    pub batches: &'static [SumcheckBatchPlan],
    pub drivers: &'static [SumcheckDriverPlan<R>],
    pub instance_results: &'static [SumcheckInstanceResultPlan<R>],
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

#[macro_export]
macro_rules! impl_runtime_plan_error_conversion {
    ($error:ident) => {
        impl From<$crate::RuntimePlanError> for $error {
            fn from(error: $crate::RuntimePlanError) -> Self {
                match error {
                    $crate::RuntimePlanError::MissingBatch { driver, batch } => {
                        Self::MissingBatch { driver, batch }
                    }
                    $crate::RuntimePlanError::MissingClaim { batch, claim } => {
                        Self::MissingClaim { batch, claim }
                    }
                    $crate::RuntimePlanError::MissingValue { symbol } => {
                        Self::MissingValue { symbol }
                    }
                    $crate::RuntimePlanError::InvalidInputLength {
                        input,
                        expected,
                        actual,
                    } => Self::InvalidInputLength {
                        input,
                        expected,
                        actual,
                    },
                    $crate::RuntimePlanError::InvalidProof { driver, reason } => {
                        Self::InvalidProof { driver, reason }
                    }
                }
            }
        }
    };
}

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

impl<R: ProtocolRelation> SymbolPlan for SumcheckClaimPlan<R> {
    fn symbol(&self) -> &'static str {
        self.symbol
    }
}

impl<R: ProtocolRelation> SymbolPlan for SumcheckDriverPlan<R> {
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

impl<R: ProtocolRelation> SumcheckClaimInfo for SumcheckClaimPlan<R> {
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

impl<R: ProtocolRelation> SumcheckDriverInfo for SumcheckDriverPlan<R> {
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

    pub fn observe_sumcheck_output<E, R: ProtocolRelation>(
        &mut self,
        instance_results: &[SumcheckInstanceResultPlan<R>],
        evals: &[SumcheckEvalPlan],
        output: &StageSumcheckOutput<F>,
        normalize_point: impl Fn(&SumcheckInstanceResultPlan<R>, Vec<F>) -> Result<Vec<F>, E>,
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
        for input in concat.inputs {
            point.extend_from_slice(self.try_point(input)?);
        }
        Some(point)
    }
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

pub fn evaluate_sumcheck_output_claim<R: ProtocolRelation>(
    plan: &SumcheckOutputClaimPlan<R>,
    field_exprs: &[FieldExprPlan],
    store: &ValueStore<Fr>,
    instance_symbol: &'static str,
    evals: &[StageNamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, RuntimePlanError> {
    let mut scratch = ScratchScalars::default();
    for eval in evals {
        scratch.insert(eval.name, eval.value);
    }
    for local_value in plan.local_values {
        if local_value.local_point.source != instance_symbol {
            return Err(RuntimePlanError::InvalidProof {
                driver: instance_symbol,
                reason: "sumcheck output value source mismatch",
            });
        }
        let opening_point = store.point_or(local_value.opening_point.source, |symbol| {
            RuntimePlanError::MissingValue { symbol }
        })?;
        let local_output_point = evaluate_sumcheck_output_point(
            local_value.local_point,
            local_point,
            local_point,
            opening_point,
        )?;
        let opening_output_point = evaluate_sumcheck_output_point(
            local_value.opening_point,
            opening_point,
            local_point,
            opening_point,
        )?;
        let value = match local_value.kind {
            SumcheckOutputValueKind::EqMle => {
                EqPolynomial::<Fr>::mle(&local_output_point, &opening_output_point)
            }
            SumcheckOutputValueKind::EqPlusOne => {
                EqPlusOnePolynomial::<Fr>::new(opening_output_point).evaluate(&local_output_point)
            }
            SumcheckOutputValueKind::Lt => {
                lt_polynomial_eval(&local_output_point, &opening_output_point)?
            }
        };
        scratch.insert(local_value.symbol, value);
    }
    evaluate_available_field_exprs_with_scratch(field_exprs, store, &mut scratch)?;
    scratch
        .scalar_or(store, plan.claim_value)
        .ok_or(RuntimePlanError::MissingValue {
            symbol: plan.claim_value,
        })
}

fn evaluate_sumcheck_output_point(
    plan: SumcheckOutputPointPlan,
    raw_point: &[Fr],
    local_point: &[Fr],
    opening_point: &[Fr],
) -> Result<Vec<Fr>, RuntimePlanError> {
    if matches!(plan.segment, SumcheckOutputPointSegment::Full)
        && !matches!(plan.length, SumcheckOutputPointLength::Full)
    {
        return Err(RuntimePlanError::InvalidProof {
            driver: plan.source,
            reason: "full output point segment requires full length",
        });
    }
    let length = match plan.length {
        SumcheckOutputPointLength::Full => raw_point.len(),
        SumcheckOutputPointLength::LocalPoint => local_point.len(),
        SumcheckOutputPointLength::OpeningPoint => opening_point.len(),
    };
    let segment = match plan.segment {
        SumcheckOutputPointSegment::Full => raw_point,
        SumcheckOutputPointSegment::Prefix => raw_point
            .get(..length)
            .filter(|prefix| prefix.len() == length)
            .ok_or(RuntimePlanError::InvalidInputLength {
                input: plan.source,
                expected: length,
                actual: raw_point.len(),
            })?,
        SumcheckOutputPointSegment::Suffix => raw_point
            .get(raw_point.len().saturating_sub(length)..)
            .filter(|suffix| suffix.len() == length)
            .ok_or(RuntimePlanError::InvalidInputLength {
                input: plan.source,
                expected: length,
                actual: raw_point.len(),
            })?,
    };
    Ok(match plan.order {
        SumcheckOutputPointOrder::AsIs => segment.to_vec(),
        SumcheckOutputPointOrder::Reverse => reverse_slice(segment),
    })
}

fn lt_polynomial_eval(x: &[Fr], y: &[Fr]) -> Result<Fr, RuntimePlanError> {
    if x.len() != y.len() {
        return Err(RuntimePlanError::InvalidInputLength {
            input: "sumcheck_output.lt",
            expected: x.len(),
            actual: y.len(),
        });
    }
    let mut lt_eval = Fr::from_u64(0);
    let mut eq_term = Fr::from_u64(1);
    for (x_i, y_i) in x.iter().zip(y) {
        lt_eval += (Fr::from_u64(1) - *x_i) * *y_i * eq_term;
        eq_term *= Fr::from_u64(1) - *x_i - *y_i + *x_i * *y_i + *x_i * *y_i;
    }
    Ok(lt_eval)
}

#[derive(Default)]
struct ScratchScalars {
    values: BTreeMap<&'static str, Fr>,
}

impl ScratchScalars {
    fn insert(&mut self, symbol: &'static str, value: Fr) {
        let _ = self.values.insert(symbol, value);
    }

    fn scalar_or(&self, store: &ValueStore<Fr>, symbol: &'static str) -> Option<Fr> {
        self.values
            .get(symbol)
            .copied()
            .or_else(|| store.try_scalar(symbol))
    }
}

fn evaluate_available_field_exprs_with_scratch(
    field_exprs: &[FieldExprPlan],
    store: &ValueStore<Fr>,
    scratch: &mut ScratchScalars,
) -> Result<(), RuntimePlanError> {
    loop {
        let mut progress = 0usize;
        for expr in field_exprs {
            if scratch.scalar_or(store, expr.symbol).is_some() {
                continue;
            }
            let Some(operands) = expr
                .operands
                .iter()
                .map(|operand| scratch.scalar_or(store, operand))
                .collect::<Option<Vec<_>>>()
            else {
                continue;
            };
            scratch.insert(expr.symbol, evaluate_field_expr(expr, &operands)?);
            progress += 1;
        }
        if progress == 0 {
            return Ok(());
        }
    }
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

pub fn store_scalar(store: &ValueStore<Fr>, symbol: &'static str) -> Result<Fr, RuntimePlanError> {
    store.scalar_or(symbol, |symbol| RuntimePlanError::MissingValue { symbol })
}

pub fn store_point<'a>(
    store: &'a ValueStore<Fr>,
    symbol: &'static str,
) -> Result<&'a [Fr], RuntimePlanError> {
    store.point_or(symbol, |symbol| RuntimePlanError::MissingValue { symbol })
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
