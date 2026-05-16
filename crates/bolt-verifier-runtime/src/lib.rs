#![expect(
    clippy::too_many_arguments,
    reason = "generated verifier helpers mirror staged protocol ABIs"
)]

use std::collections::BTreeMap;
use std::fmt;
use std::marker::PhantomData;

use jolt_field::{Field, Fr, MulPow2};
use jolt_poly::{
    lagrange::{lagrange_evals, lagrange_kernel_eval},
    EqPlusOnePolynomial, EqPolynomial, LtPolynomial,
};
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
    Sum,
    Product,
    Neg,
    Pow(usize),
    LagrangeBasisEval(i64, usize, usize),
    LagrangeKernelEval(i64, usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FieldExprPlan {
    pub symbol: &'static str,
    pub kind: FieldExprKind,
    pub operands: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarExprKind {
    FieldVectorSum,
    FieldVectorProduct,
    StructuredPolynomial {
        polynomial: StructuredPolynomialKind,
        x_point: StructuredPolynomialPointTransform,
        y_point: StructuredPolynomialPointTransform,
    },
    PowerStridedWeightedSum {
        row_count: usize,
        power_stride: usize,
        value_term_offsets: &'static [usize],
        shared_term_offsets: &'static [usize],
        row_term_offsets: &'static [usize],
    },
    PointElement {
        index: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScalarExprPlan {
    pub symbol: &'static str,
    pub kind: ScalarExprKind,
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
    pub point_order: SumcheckPointOrder,
    pub degree: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SumcheckPointOrder {
    AsIs,
    Reverse,
    Stage4RegistersReadWrite,
    InstructionReadRaf,
    BytecodeReadRaf,
    Stage6Booleanity,
}

impl SumcheckPointOrder {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AsIs => "as_is",
            Self::Reverse => "reverse",
            Self::Stage4RegistersReadWrite => "stage4_registers_rw",
            Self::InstructionReadRaf => "instruction_read_raf",
            Self::BytecodeReadRaf => "bytecode_read_raf",
            Self::Stage6Booleanity => "stage6_booleanity",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StructuredPolynomialPointOrder {
    AsIs,
    Reverse,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StructuredPolynomialPointSegment {
    Full,
    Prefix,
    Suffix,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StructuredPolynomialPointLength {
    Full,
    XPoint,
    YPoint,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct StructuredPolynomialPointPlan {
    pub source: &'static str,
    pub segment: StructuredPolynomialPointSegment,
    pub length: StructuredPolynomialPointLength,
    pub order: StructuredPolynomialPointOrder,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StructuredPolynomialPointTransform {
    pub segment: StructuredPolynomialPointSegment,
    pub length: StructuredPolynomialPointLength,
    pub order: StructuredPolynomialPointOrder,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StructuredPolynomialKind {
    Eq,
    EqPlusOne,
    Lt,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RelationOutputPlan<R: ProtocolRelation> {
    pub relation: R,
    pub local_scalars: &'static [&'static str],
    pub expected_output: &'static str,
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
pub enum PointExprKind {
    Zero { field: &'static str, arity: usize },
    Slice { offset: usize, length: usize },
    Concat { layout: &'static str, arity: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PointExprPlan {
    pub symbol: &'static str,
    pub kind: PointExprKind,
    pub operands: &'static [&'static str],
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
    pub scalar_exprs: &'static [ScalarExprPlan],
    pub kernels: &'static [KernelPlan],
    pub claims: &'static [SumcheckClaimPlan<R>],
    pub batches: &'static [SumcheckBatchPlan],
    pub drivers: &'static [SumcheckDriverPlan<R>],
    pub instance_results: &'static [SumcheckInstanceResultPlan<R>],
    pub evals: &'static [SumcheckEvalPlan],
    pub indexed_eval_families: &'static [NamedEvalFamilyPlan],
    pub relation_outputs: &'static [RelationOutputPlan<R>],
    pub point_exprs: &'static [PointExprPlan],
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
    pub scalar_exprs: &'static [ScalarExprPlan],
    pub claims: &'static [SumcheckClaimPlan<R>],
    pub batches: &'static [SumcheckBatchPlan],
    pub drivers: &'static [SumcheckDriverPlan<R>],
    pub instance_results: &'static [SumcheckInstanceResultPlan<R>],
    pub evals: &'static [SumcheckEvalPlan],
    pub relation_outputs: &'static [RelationOutputPlan<R>],
    pub point_exprs: &'static [PointExprPlan],
    pub opening_claims: &'static [OpeningClaimPlan],
    pub opening_equalities: &'static [OpeningClaimEqualityPlan],
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NamedScalar<F: Field> {
    pub symbol: &'static str,
    pub value: F,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NamedPoint<'a, F: Field> {
    pub symbol: &'static str,
    pub point: &'a [F],
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct NamedFieldVector<F: Field> {
    symbol: &'static str,
    values: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelationOutputInputs<'a, F: Field> {
    pub scalars: Vec<NamedScalar<F>>,
    pub points: Vec<NamedPoint<'a, F>>,
}

impl<F: Field> Default for RelationOutputInputs<'_, F> {
    fn default() -> Self {
        Self {
            scalars: Vec::new(),
            points: Vec::new(),
        }
    }
}

impl<F: Field> RelationOutputInputs<'_, F> {
    pub fn empty() -> Self {
        Self::default()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NamedEvalFamilyPlan {
    pub symbol: &'static str,
    pub evals: &'static [&'static str],
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
    field_vectors: Vec<(&'static str, Vec<F>)>,
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
        point_exprs: &[PointExprPlan],
        invalid_input_length: impl Fn(&'static str, usize, usize) -> E,
    ) -> Result<(), E> {
        loop {
            let mut progress = 0usize;
            for expr in point_exprs {
                if self.try_point(expr.symbol).is_some() {
                    continue;
                }
                let Some(point) = self.try_point_expr(expr, &invalid_input_length)? else {
                    continue;
                };
                self.insert_point(expr.symbol, point);
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
                let Some(operands) = self.try_field_expr_operands(expr) else {
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

    pub fn evaluate_available_exprs(
        &mut self,
        field_exprs: &[FieldExprPlan],
        scalar_exprs: &[ScalarExprPlan],
    ) -> Result<(), RuntimePlanError> {
        loop {
            let mut progress = 0usize;
            for expr in field_exprs {
                if self.try_scalar(expr.symbol).is_some() {
                    continue;
                }
                let Some(operands) = self.try_field_expr_operands(expr) else {
                    continue;
                };
                self.insert_scalar(expr.symbol, evaluate_field_expr(expr, &operands)?);
                progress += 1;
            }
            for expr in scalar_exprs {
                if self.try_scalar(expr.symbol).is_some() {
                    continue;
                }
                let Some(operands) = self.try_scalar_expr_operands(expr) else {
                    continue;
                };
                self.insert_scalar(expr.symbol, evaluate_scalar_expr(expr, &operands)?);
                progress += 1;
            }
            if progress == 0 {
                return Ok(());
            }
        }
    }

    pub fn evaluate_named_eval_families(
        &mut self,
        families: &[NamedEvalFamilyPlan],
    ) -> Result<(), RuntimePlanError> {
        for family in families {
            if self.try_field_vector(family.symbol).is_some() {
                continue;
            }
            let values = family
                .evals
                .iter()
                .map(|eval| {
                    self.try_scalar(eval)
                        .ok_or(RuntimePlanError::MissingValue { symbol: eval })
                })
                .collect::<Result<Vec<_>, RuntimePlanError>>()?;
            self.insert_field_vector(family.symbol, values);
        }
        Ok(())
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

    pub fn insert_field_vector(&mut self, symbol: &'static str, values: Vec<F>) {
        if let Some((_, existing)) = self
            .field_vectors
            .iter_mut()
            .find(|(name, _)| *name == symbol)
        {
            *existing = values;
        } else {
            self.field_vectors.push((symbol, values));
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

    pub fn field_vector_or<E>(
        &self,
        symbol: &'static str,
        missing_value: impl FnOnce(&'static str) -> E,
    ) -> Result<&[F], E> {
        self.try_field_vector(symbol)
            .ok_or_else(|| missing_value(symbol))
    }

    pub fn try_field_vector(&self, symbol: &str) -> Option<&[F]> {
        self.field_vectors
            .iter()
            .find(|(name, _)| *name == symbol)
            .map(|(_, values)| values.as_slice())
    }

    fn try_field_expr_operands(&self, expr: &FieldExprPlan) -> Option<Vec<F>> {
        expr.operands
            .iter()
            .map(|operand| self.try_scalar(operand))
            .collect()
    }

    fn try_scalar_expr_operands(&self, expr: &ScalarExprPlan) -> Option<Vec<F>> {
        match expr.kind {
            ScalarExprKind::FieldVectorSum | ScalarExprKind::FieldVectorProduct => {
                let [symbol] = expr.operands else {
                    return Some(Vec::new());
                };
                self.try_field_vector(symbol).map(|values| values.to_vec())
            }
            ScalarExprKind::StructuredPolynomial { .. } => None,
            ScalarExprKind::PowerStridedWeightedSum { .. } => expr
                .operands
                .iter()
                .map(|operand| self.try_scalar(operand))
                .collect(),
            ScalarExprKind::PointElement { index } => {
                let [symbol] = expr.operands else {
                    return Some(Vec::new());
                };
                self.try_point(symbol)
                    .map(|point| point.get(index).copied().into_iter().collect())
            }
        }
    }

    fn try_point_expr<E>(
        &self,
        expr: &PointExprPlan,
        invalid_input_length: &impl Fn(&'static str, usize, usize) -> E,
    ) -> Result<Option<Vec<F>>, E> {
        match expr.kind {
            PointExprKind::Zero { arity, .. } => Ok(Some(vec![F::from_u64(0); arity])),
            PointExprKind::Slice { offset, length } => {
                let [input] = expr.operands else {
                    return Err(invalid_input_length(expr.symbol, 1, expr.operands.len()));
                };
                let Some(point) = self.try_point(input) else {
                    return Ok(None);
                };
                let end = offset + length;
                let point = point
                    .get(offset..end)
                    .ok_or_else(|| invalid_input_length(expr.symbol, end, point.len()))?
                    .to_vec();
                Ok(Some(point))
            }
            PointExprKind::Concat { arity, .. } => {
                let mut point = Vec::with_capacity(arity);
                for input in expr.operands {
                    let Some(input) = self.try_point(input) else {
                        return Ok(None);
                    };
                    point.extend_from_slice(input);
                }
                if point.len() != arity {
                    return Err(invalid_input_length(expr.symbol, arity, point.len()));
                }
                Ok(Some(point))
            }
        }
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
    scalar_exprs: &[ScalarExprPlan],
    store: &mut ValueStore<Fr>,
) -> Result<Vec<Fr>, RuntimePlanError> {
    claims
        .iter()
        .map(|claim| {
            store.evaluate_available_exprs(field_exprs, scalar_exprs)?;
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
    scalar_exprs: &'static [ScalarExprPlan],
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
    let input_claims = batch_claim_values(&claims, field_exprs, scalar_exprs, store)?;
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

pub fn eval_family_values<F: Field>(
    evals: &[StageNamedEval<F>],
    family: &NamedEvalFamilyPlan,
) -> Result<Vec<F>, RuntimePlanError> {
    family
        .evals
        .iter()
        .map(|&name| eval_by_name(evals, name))
        .collect()
}

pub fn try_eval_family_values<F: Field>(
    evals: &[StageNamedEval<F>],
    family: &NamedEvalFamilyPlan,
) -> Option<Vec<F>> {
    family
        .evals
        .iter()
        .map(|&name| {
            evals
                .iter()
                .find(|eval| eval.name == name)
                .map(|eval| eval.value)
        })
        .collect()
}

pub fn evaluate_relation_output<R: ProtocolRelation>(
    plan: &RelationOutputPlan<R>,
    field_exprs: &[FieldExprPlan],
    scalar_exprs: &[ScalarExprPlan],
    eval_families: &[NamedEvalFamilyPlan],
    store: &ValueStore<Fr>,
    instance_symbol: &'static str,
    evals: &[StageNamedEval<Fr>],
    local_scalars: &[NamedScalar<Fr>],
    local_points: &[NamedPoint<'_, Fr>],
    local_point: &[Fr],
) -> Result<Fr, RuntimePlanError> {
    let mut scratch = ScratchScalars::default();
    for scalar in local_scalars {
        scratch.insert(scalar.symbol, scalar.value);
    }
    for symbol in plan.local_scalars {
        if !local_scalars.iter().any(|scalar| scalar.symbol == *symbol) {
            return Err(RuntimePlanError::MissingValue { symbol });
        }
    }
    for eval in evals {
        scratch.insert(eval.name, eval.value);
    }
    let local_field_vectors = eval_families
        .iter()
        .filter_map(|family| {
            try_eval_family_values(evals, family).map(|values| NamedFieldVector {
                symbol: family.symbol,
                values,
            })
        })
        .collect::<Vec<_>>();
    let context = RelationOutputContext {
        instance_symbol,
        local_points,
        local_point,
        local_field_vectors: &local_field_vectors,
    };
    evaluate_available_exprs_with_scratch(field_exprs, scalar_exprs, store, &mut scratch, context)?;
    scratch
        .scalar_or(store, plan.expected_output)
        .ok_or(RuntimePlanError::MissingValue {
            symbol: plan.expected_output,
        })
}

pub fn evaluate_relation_output_for_instance<R: ProtocolRelation>(
    relation_outputs: &[RelationOutputPlan<R>],
    field_exprs: &[FieldExprPlan],
    scalar_exprs: &[ScalarExprPlan],
    eval_families: &[NamedEvalFamilyPlan],
    store: &ValueStore<Fr>,
    instance: &SumcheckInstanceResultPlan<R>,
    evals: &[StageNamedEval<Fr>],
    local_scalars: &[NamedScalar<Fr>],
    local_points: &[NamedPoint<'_, Fr>],
    local_point: &[Fr],
) -> Result<Fr, RuntimePlanError> {
    let relation_output = relation_outputs
        .iter()
        .find(|relation_output| relation_output.relation == instance.relation)
        .ok_or(RuntimePlanError::InvalidProof {
            driver: instance.symbol,
            reason: "missing relation output for relation",
        })?;
    evaluate_relation_output(
        relation_output,
        field_exprs,
        scalar_exprs,
        eval_families,
        store,
        instance.symbol,
        evals,
        local_scalars,
        local_points,
        local_point,
    )
}

pub fn evaluate_relation_output_batch<R, E, LocalInputs>(
    driver: &SumcheckDriverPlan<R>,
    batches: &[SumcheckBatchPlan],
    claims: &[SumcheckClaimPlan<R>],
    instance_results: &[SumcheckInstanceResultPlan<R>],
    relation_outputs: &[RelationOutputPlan<R>],
    field_exprs: &[FieldExprPlan],
    scalar_exprs: &[ScalarExprPlan],
    eval_families: &[NamedEvalFamilyPlan],
    store: &ValueStore<Fr>,
    evals: &[StageNamedEval<Fr>],
    point: &[Fr],
    batching_coeffs: &[Fr],
    mut local_inputs: LocalInputs,
) -> Result<Fr, E>
where
    R: ProtocolRelation,
    E: From<RuntimePlanError>,
    LocalInputs: for<'a> FnMut(
        &SumcheckInstanceResultPlan<R>,
        &'a [Fr],
    ) -> Result<RelationOutputInputs<'a, Fr>, E>,
{
    let batch = find_batch(batches, driver.symbol, driver.batch)?;
    let claims = batch_claims(claims, batch)?;
    let mut expected = Fr::from_u64(0);
    for (claim, coefficient) in claims.iter().zip(batching_coeffs) {
        let instance = instance_results
            .iter()
            .find(|instance| instance.claim == claim.symbol && instance.source == driver.symbol)
            .ok_or(RuntimePlanError::MissingClaim {
                batch: batch.symbol,
                claim: claim.symbol,
            })?;
        let local_point = point
            .get(instance.round_offset..instance.round_offset + instance.num_rounds)
            .ok_or(RuntimePlanError::InvalidInputLength {
                input: instance.symbol,
                expected: instance.round_offset + instance.num_rounds,
                actual: point.len(),
            })?;
        let inputs = local_inputs(instance, local_point)?;
        let value = evaluate_relation_output_for_instance(
            relation_outputs,
            field_exprs,
            scalar_exprs,
            eval_families,
            store,
            instance,
            evals,
            &inputs.scalars,
            &inputs.points,
            local_point,
        )?;
        expected += *coefficient * value;
    }
    Ok(expected)
}

fn try_relation_output_x_point_source<'a>(
    source: &'static str,
    instance_symbol: &'static str,
    local_points: &'a [NamedPoint<'a, Fr>],
    local_point: &'a [Fr],
    store: &'a ValueStore<Fr>,
) -> Option<&'a [Fr]> {
    local_points
        .iter()
        .find(|point| point.symbol == source)
        .map(|point| point.point)
        .or_else(|| (source == instance_symbol).then_some(local_point))
        .or_else(|| store.try_point(source))
}

#[derive(Clone, Copy)]
struct RelationOutputContext<'a> {
    instance_symbol: &'static str,
    local_points: &'a [NamedPoint<'a, Fr>],
    local_point: &'a [Fr],
    local_field_vectors: &'a [NamedFieldVector<Fr>],
}

fn evaluate_structured_polynomial_point(
    plan: StructuredPolynomialPointPlan,
    raw_point: &[Fr],
    x_point: &[Fr],
    y_point: &[Fr],
) -> Result<Vec<Fr>, RuntimePlanError> {
    if matches!(plan.segment, StructuredPolynomialPointSegment::Full)
        && !matches!(plan.length, StructuredPolynomialPointLength::Full)
    {
        return Err(RuntimePlanError::InvalidProof {
            driver: plan.source,
            reason: "full output point segment requires full length",
        });
    }
    let length = match plan.length {
        StructuredPolynomialPointLength::Full => raw_point.len(),
        StructuredPolynomialPointLength::XPoint => x_point.len(),
        StructuredPolynomialPointLength::YPoint => y_point.len(),
    };
    let segment = match plan.segment {
        StructuredPolynomialPointSegment::Full => raw_point,
        StructuredPolynomialPointSegment::Prefix => raw_point
            .get(..length)
            .filter(|prefix| prefix.len() == length)
            .ok_or(RuntimePlanError::InvalidInputLength {
                input: plan.source,
                expected: length,
                actual: raw_point.len(),
            })?,
        StructuredPolynomialPointSegment::Suffix => raw_point
            .get(raw_point.len().saturating_sub(length)..)
            .filter(|suffix| suffix.len() == length)
            .ok_or(RuntimePlanError::InvalidInputLength {
                input: plan.source,
                expected: length,
                actual: raw_point.len(),
            })?,
    };
    Ok(match plan.order {
        StructuredPolynomialPointOrder::AsIs => segment.to_vec(),
        StructuredPolynomialPointOrder::Reverse => reverse_slice(segment),
    })
}

fn evaluate_structured_polynomial(
    polynomial: StructuredPolynomialKind,
    x: &[Fr],
    y: Vec<Fr>,
) -> Result<Fr, RuntimePlanError> {
    Ok(match polynomial {
        StructuredPolynomialKind::Eq => evaluate_eq_polynomial_mle(x, &y)?,
        StructuredPolynomialKind::EqPlusOne => evaluate_eq_plus_one_polynomial_mle(x, y)?,
        StructuredPolynomialKind::Lt => evaluate_lt_polynomial_mle(x, &y)?,
    })
}

fn evaluate_eq_polynomial_mle(x: &[Fr], y: &[Fr]) -> Result<Fr, RuntimePlanError> {
    EqPolynomial::<Fr>::try_mle(x, y).ok_or(RuntimePlanError::InvalidInputLength {
        input: "sumcheck_output.eq",
        expected: x.len(),
        actual: y.len(),
    })
}

fn evaluate_eq_plus_one_polynomial_mle(x: &[Fr], y: Vec<Fr>) -> Result<Fr, RuntimePlanError> {
    let y_len = y.len();
    EqPlusOnePolynomial::<Fr>::new(y)
        .try_evaluate(x)
        .ok_or(RuntimePlanError::InvalidInputLength {
            input: "sumcheck_output.eq_plus_one",
            expected: x.len(),
            actual: y_len,
        })
}

fn evaluate_lt_polynomial_mle(x: &[Fr], y: &[Fr]) -> Result<Fr, RuntimePlanError> {
    LtPolynomial::try_evaluate(x, y).ok_or(RuntimePlanError::InvalidInputLength {
        input: "sumcheck_output.lt",
        expected: x.len(),
        actual: y.len(),
    })
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

fn evaluate_available_exprs_with_scratch(
    field_exprs: &[FieldExprPlan],
    scalar_exprs: &[ScalarExprPlan],
    store: &ValueStore<Fr>,
    scratch: &mut ScratchScalars,
    context: RelationOutputContext<'_>,
) -> Result<(), RuntimePlanError> {
    loop {
        let mut progress = 0usize;
        for expr in field_exprs {
            if scratch.scalar_or(store, expr.symbol).is_some() {
                continue;
            }
            let Some(operands) = relation_output_expr_operands(expr, store, scratch) else {
                continue;
            };
            scratch.insert(expr.symbol, evaluate_field_expr(expr, &operands)?);
            progress += 1;
        }
        for expr in scalar_exprs {
            if scratch.scalar_or(store, expr.symbol).is_some() {
                continue;
            }
            let Some(value) = evaluate_relation_output_scalar_expr(expr, store, scratch, context)?
            else {
                continue;
            };
            scratch.insert(expr.symbol, value);
            progress += 1;
        }
        if progress == 0 {
            return Ok(());
        }
    }
}

fn relation_output_expr_operands(
    expr: &FieldExprPlan,
    store: &ValueStore<Fr>,
    scratch: &ScratchScalars,
) -> Option<Vec<Fr>> {
    expr.operands
        .iter()
        .map(|operand| scratch.scalar_or(store, operand))
        .collect()
}

fn evaluate_relation_output_scalar_expr(
    expr: &ScalarExprPlan,
    store: &ValueStore<Fr>,
    scratch: &ScratchScalars,
    context: RelationOutputContext<'_>,
) -> Result<Option<Fr>, RuntimePlanError> {
    match expr.kind {
        ScalarExprKind::FieldVectorSum | ScalarExprKind::FieldVectorProduct => {
            let [symbol] = expr.operands else {
                return evaluate_scalar_expr(expr, &[]).map(Some);
            };
            let Some(values) = context
                .local_field_vectors
                .iter()
                .find(|field_vector| field_vector.symbol == *symbol)
                .map(|field_vector| field_vector.values.as_slice())
                .or_else(|| store.try_field_vector(symbol))
            else {
                return Ok(None);
            };
            evaluate_scalar_expr(expr, values).map(Some)
        }
        ScalarExprKind::StructuredPolynomial {
            polynomial,
            x_point,
            y_point,
        } => evaluate_structured_polynomial_scalar(
            expr.symbol,
            expr.operands,
            polynomial,
            x_point,
            y_point,
            store,
            context,
        ),
        ScalarExprKind::PowerStridedWeightedSum { .. } => expr
            .operands
            .iter()
            .map(|operand| scratch.scalar_or(store, operand))
            .collect::<Option<Vec<_>>>()
            .map(|operands| evaluate_scalar_expr(expr, &operands).map(Some))
            .transpose()
            .map(Option::flatten),
        ScalarExprKind::PointElement { index } => {
            let [source] = expr.operands else {
                return evaluate_scalar_expr(expr, &[]).map(Some);
            };
            let Some(point) = try_relation_output_x_point_source(
                source,
                context.instance_symbol,
                context.local_points,
                context.local_point,
                store,
            ) else {
                return Ok(None);
            };
            let value = point
                .get(index)
                .copied()
                .ok_or(RuntimePlanError::InvalidInputLength {
                    input: expr.symbol,
                    expected: index + 1,
                    actual: point.len(),
                })?;
            Ok(Some(value))
        }
    }
}

fn evaluate_structured_polynomial_scalar(
    symbol: &'static str,
    operands: &[&'static str],
    polynomial: StructuredPolynomialKind,
    x_transform: StructuredPolynomialPointTransform,
    y_transform: StructuredPolynomialPointTransform,
    store: &ValueStore<Fr>,
    context: RelationOutputContext<'_>,
) -> Result<Option<Fr>, RuntimePlanError> {
    require_operand_count(symbol, 2, operands.len())?;
    let x_source = operands[0];
    let y_source = operands[1];
    let Some(x_raw_point) = try_relation_output_x_point_source(
        x_source,
        context.instance_symbol,
        context.local_points,
        context.local_point,
        store,
    ) else {
        return Ok(None);
    };
    let Some(y_raw_point) = store.try_point(y_source) else {
        return Ok(None);
    };
    let x_plan = StructuredPolynomialPointPlan {
        source: x_source,
        segment: x_transform.segment,
        length: x_transform.length,
        order: x_transform.order,
    };
    let y_plan = StructuredPolynomialPointPlan {
        source: y_source,
        segment: y_transform.segment,
        length: y_transform.length,
        order: y_transform.order,
    };
    let x_point =
        evaluate_structured_polynomial_point(x_plan, x_raw_point, x_raw_point, y_raw_point)?;
    let y_point =
        evaluate_structured_polynomial_point(y_plan, y_raw_point, x_raw_point, y_raw_point)?;
    evaluate_structured_polynomial(polynomial, &x_point, y_point).map(Some)
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

pub fn require_min_operand_count(
    input: &'static str,
    minimum: usize,
    actual: usize,
) -> Result<(), RuntimePlanError> {
    if actual >= minimum {
        Ok(())
    } else {
        Err(RuntimePlanError::InvalidInputLength {
            input,
            expected: minimum,
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
        FieldExprKind::Sum => {
            require_min_operand_count(expr.symbol, 1, operands.len())?;
            Ok(operands
                .iter()
                .copied()
                .fold(F::from_u64(0), |acc, operand| acc + operand))
        }
        FieldExprKind::Product => {
            require_min_operand_count(expr.symbol, 1, operands.len())?;
            Ok(operands
                .iter()
                .copied()
                .fold(F::from_u64(1), |acc, operand| acc * operand))
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
        FieldExprKind::LagrangeKernelEval(domain_start, domain_size) => {
            require_operand_count(expr.symbol, 2, operands.len())?;
            Ok(lagrange_kernel_eval(
                domain_start,
                domain_size,
                operands[0],
                operands[1],
            ))
        }
    }
}

pub fn evaluate_scalar_expr<F: Field>(
    expr: &ScalarExprPlan,
    operands: &[F],
) -> Result<F, RuntimePlanError> {
    match expr.kind {
        ScalarExprKind::FieldVectorSum => {
            require_min_operand_count(expr.symbol, 1, operands.len())?;
            Ok(operands
                .iter()
                .copied()
                .fold(F::from_u64(0), |acc, operand| acc + operand))
        }
        ScalarExprKind::FieldVectorProduct => {
            require_min_operand_count(expr.symbol, 1, operands.len())?;
            Ok(operands
                .iter()
                .copied()
                .fold(F::from_u64(1), |acc, operand| acc * operand))
        }
        ScalarExprKind::StructuredPolynomial { .. } => Err(RuntimePlanError::InvalidProof {
            driver: expr.symbol,
            reason:
                "structured polynomial scalar expressions require relation-output point context",
        }),
        ScalarExprKind::PowerStridedWeightedSum {
            row_count,
            power_stride,
            value_term_offsets,
            shared_term_offsets,
            row_term_offsets,
        } => evaluate_power_strided_weighted_sum(
            expr.symbol,
            operands,
            row_count,
            power_stride,
            value_term_offsets,
            shared_term_offsets,
            row_term_offsets,
        ),
        ScalarExprKind::PointElement { index: _ } => {
            require_operand_count(expr.symbol, 1, operands.len())?;
            Ok(operands[0])
        }
    }
}

fn evaluate_power_strided_weighted_sum<F: Field>(
    symbol: &'static str,
    operands: &[F],
    row_count: usize,
    power_stride: usize,
    value_term_offsets: &[usize],
    shared_term_offsets: &[usize],
    row_term_offsets: &[usize],
) -> Result<F, RuntimePlanError> {
    let term_count = value_term_offsets.len() + shared_term_offsets.len() + row_term_offsets.len();
    if row_count == 0 || term_count == 0 {
        return Err(RuntimePlanError::InvalidInputLength {
            input: symbol,
            expected: 1,
            actual: 0,
        });
    }
    let expected_operands =
        1 + row_count + shared_term_offsets.len() + row_count * row_term_offsets.len();
    require_operand_count(symbol, expected_operands, operands.len())?;

    let gamma = operands[0];
    let evals_start = 1;
    let shared_start = evals_start + row_count;
    let item_start = shared_start + shared_term_offsets.len();
    let evals = &operands[evals_start..shared_start];
    let shared_factors = &operands[shared_start..item_start];
    let row_factors = &operands[item_start..];

    let value_offset_powers = value_term_offsets
        .iter()
        .map(|&offset| pow_field(gamma, offset))
        .collect::<Vec<_>>();
    let shared_terms = shared_term_offsets
        .iter()
        .zip(shared_factors.iter())
        .map(|(&offset, &factor)| (pow_field(gamma, offset), factor))
        .collect::<Vec<_>>();
    let row_offset_powers = row_term_offsets
        .iter()
        .map(|&offset| pow_field(gamma, offset))
        .collect::<Vec<_>>();

    let gamma_stride = pow_field(gamma, power_stride);
    let mut gamma_base = F::from_u64(1);
    let mut result = F::from_u64(0);
    for (eval_index, &eval) in evals.iter().enumerate() {
        let weighted_eval = eval * gamma_base;
        for offset_power in &value_offset_powers {
            result += weighted_eval * *offset_power;
        }
        for (offset_power, factor) in &shared_terms {
            result += weighted_eval * *factor * *offset_power;
        }
        for (term_index, offset_power) in row_offset_powers.iter().enumerate() {
            let factor = row_factors[term_index * row_count + eval_index];
            result += weighted_eval * factor * *offset_power;
        }
        gamma_base *= gamma_stride;
    }
    Ok(result)
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

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "tests assert the success and error paths directly"
)]
mod tests {
    use super::{
        evaluate_relation_output_for_instance, evaluate_scalar_expr, FieldExprKind, FieldExprPlan,
        Fr, NamedEvalFamilyPlan, RelationOutputPlan, RuntimePlanError, ScalarExprKind,
        ScalarExprPlan, StageNamedEval, StructuredPolynomialKind, StructuredPolynomialPointLength,
        StructuredPolynomialPointOrder, StructuredPolynomialPointSegment,
        StructuredPolynomialPointTransform, SumcheckInstanceResultPlan, SumcheckPointOrder,
        ValueStore,
    };

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum TestRelation {
        Output,
        Sibling,
    }

    #[test]
    fn value_store_evaluates_named_eval_families_as_field_vectors() {
        let mut store = ValueStore::default();
        store.insert_scalar("eval.a", Fr::from_u64(2));
        store.insert_scalar("eval.b", Fr::from_u64(3));

        store
            .evaluate_named_eval_families(&[NamedEvalFamilyPlan {
                symbol: "family.ab",
                evals: &["eval.a", "eval.b"],
            }])
            .unwrap();

        assert_eq!(
            store.try_field_vector("family.ab"),
            Some([Fr::from_u64(2), Fr::from_u64(3)].as_slice())
        );
    }

    #[test]
    fn value_store_rejects_named_eval_family_missing_eval() {
        let mut store = ValueStore::<Fr>::default();
        let error = store
            .evaluate_named_eval_families(&[NamedEvalFamilyPlan {
                symbol: "family.ab",
                evals: &["eval.missing"],
            }])
            .unwrap_err();

        assert_eq!(
            error,
            RuntimePlanError::MissingValue {
                symbol: "eval.missing"
            }
        );
    }

    #[test]
    fn value_store_evaluates_field_vector_exprs() {
        let mut store = ValueStore::default();
        store.insert_field_vector("family.ab", vec![Fr::from_u64(2), Fr::from_u64(3)]);

        store
            .evaluate_available_exprs(
                &[],
                &[
                    ScalarExprPlan {
                        symbol: "family.ab.product",
                        kind: ScalarExprKind::FieldVectorProduct,
                        operands: &["family.ab"],
                    },
                    ScalarExprPlan {
                        symbol: "family.ab.sum",
                        kind: ScalarExprKind::FieldVectorSum,
                        operands: &["family.ab"],
                    },
                ],
            )
            .unwrap();

        assert_eq!(store.try_scalar("family.ab.product"), Some(Fr::from_u64(6)));
        assert_eq!(store.try_scalar("family.ab.sum"), Some(Fr::from_u64(5)));
    }

    #[test]
    fn value_store_evaluates_point_element_and_lagrange_kernel_exprs() {
        let mut store = ValueStore::default();
        store.insert_point("point.r", vec![Fr::from_u64(2)]);
        store.insert_scalar("tau", Fr::from_u64(5));

        store
            .evaluate_available_exprs(
                &[FieldExprPlan {
                    symbol: "kernel",
                    kind: FieldExprKind::LagrangeKernelEval(-1, 3),
                    operands: &["tau", "r0"],
                }],
                &[ScalarExprPlan {
                    symbol: "r0",
                    kind: ScalarExprKind::PointElement { index: 0 },
                    operands: &["point.r"],
                }],
            )
            .unwrap();

        assert_eq!(store.try_scalar("r0"), Some(Fr::from_u64(2)));
        assert_eq!(
            store.try_scalar("kernel"),
            Some(jolt_poly::lagrange::lagrange_kernel_eval(
                -1,
                3,
                Fr::from_u64(5),
                Fr::from_u64(2),
            ))
        );
    }

    #[test]
    fn scalar_expr_evaluates_power_strided_weighted_sum() {
        let value = evaluate_scalar_expr(
            &ScalarExprPlan {
                symbol: "family.weighted",
                kind: ScalarExprKind::PowerStridedWeightedSum {
                    row_count: 2,
                    power_stride: 3,
                    value_term_offsets: &[0],
                    shared_term_offsets: &[1],
                    row_term_offsets: &[2],
                },
                operands: &["gamma", "eval.a", "eval.b", "shared.eq", "item.a", "item.b"],
            },
            &[
                Fr::from_u64(2),
                Fr::from_u64(3),
                Fr::from_u64(5),
                Fr::from_u64(7),
                Fr::from_u64(11),
                Fr::from_u64(13),
            ],
        )
        .unwrap();

        assert_eq!(value, Fr::from_u64(2857));
    }

    #[test]
    fn relation_output_evaluates_structured_polynomial_scalar_exprs() {
        let x = [Fr::from_u64(1), Fr::from_u64(0)];
        let mut store = ValueStore::default();
        store.insert_point("point.y", x.to_vec());

        let value = evaluate_relation_output_for_instance(
            &[RelationOutputPlan {
                relation: TestRelation::Output,
                local_scalars: &[],
                expected_output: "eq.xy",
            }],
            &[],
            &[ScalarExprPlan {
                symbol: "eq.xy",
                kind: ScalarExprKind::StructuredPolynomial {
                    polynomial: StructuredPolynomialKind::Eq,
                    x_point: StructuredPolynomialPointTransform {
                        segment: StructuredPolynomialPointSegment::Full,
                        length: StructuredPolynomialPointLength::Full,
                        order: StructuredPolynomialPointOrder::AsIs,
                    },
                    y_point: StructuredPolynomialPointTransform {
                        segment: StructuredPolynomialPointSegment::Full,
                        length: StructuredPolynomialPointLength::Full,
                        order: StructuredPolynomialPointOrder::AsIs,
                    },
                },
                operands: &["instance", "point.y"],
            }],
            &[],
            &store,
            &SumcheckInstanceResultPlan {
                symbol: "instance",
                source: "driver",
                claim: "claim",
                relation: TestRelation::Output,
                index: 0,
                point_arity: 2,
                num_rounds: 2,
                round_offset: 0,
                point_order: SumcheckPointOrder::AsIs,
                degree: 2,
            },
            &[],
            &[],
            &[],
            &x,
        )
        .unwrap();

        assert_eq!(value, Fr::from_u64(1));
    }

    #[test]
    fn relation_output_skips_unavailable_sibling_instance_exprs() {
        let x = [Fr::from_u64(1), Fr::from_u64(0)];
        let mut store = ValueStore::default();
        store.insert_point("point.y", x.to_vec());

        let structured_eq = ScalarExprKind::StructuredPolynomial {
            polynomial: StructuredPolynomialKind::Eq,
            x_point: StructuredPolynomialPointTransform {
                segment: StructuredPolynomialPointSegment::Full,
                length: StructuredPolynomialPointLength::Full,
                order: StructuredPolynomialPointOrder::AsIs,
            },
            y_point: StructuredPolynomialPointTransform {
                segment: StructuredPolynomialPointSegment::Full,
                length: StructuredPolynomialPointLength::Full,
                order: StructuredPolynomialPointOrder::AsIs,
            },
        };
        let value = evaluate_relation_output_for_instance(
            &[
                RelationOutputPlan {
                    relation: TestRelation::Output,
                    local_scalars: &[],
                    expected_output: "eq.current",
                },
                RelationOutputPlan {
                    relation: TestRelation::Sibling,
                    local_scalars: &[],
                    expected_output: "eq.sibling",
                },
            ],
            &[],
            &[
                ScalarExprPlan {
                    symbol: "eq.sibling",
                    kind: structured_eq,
                    operands: &["sibling.instance", "point.y"],
                },
                ScalarExprPlan {
                    symbol: "eq.current",
                    kind: structured_eq,
                    operands: &["instance", "point.y"],
                },
            ],
            &[],
            &store,
            &SumcheckInstanceResultPlan {
                symbol: "instance",
                source: "driver",
                claim: "claim",
                relation: TestRelation::Output,
                index: 0,
                point_arity: 2,
                num_rounds: 2,
                round_offset: 0,
                point_order: SumcheckPointOrder::AsIs,
                degree: 2,
            },
            &[],
            &[],
            &[],
            &x,
        )
        .unwrap();

        assert_eq!(value, Fr::from_u64(1));
    }

    #[test]
    fn relation_output_evaluates_eval_family_vectors_from_proof_evals() {
        let value = evaluate_relation_output_for_instance(
            &[RelationOutputPlan {
                relation: TestRelation::Output,
                local_scalars: &[],
                expected_output: "family.product",
            }],
            &[],
            &[ScalarExprPlan {
                symbol: "family.product",
                kind: ScalarExprKind::FieldVectorProduct,
                operands: &["family.ab"],
            }],
            &[NamedEvalFamilyPlan {
                symbol: "family.ab",
                evals: &["eval.a", "eval.b"],
            }],
            &ValueStore::default(),
            &SumcheckInstanceResultPlan {
                symbol: "instance",
                source: "driver",
                claim: "claim",
                relation: TestRelation::Output,
                index: 0,
                point_arity: 0,
                num_rounds: 0,
                round_offset: 0,
                point_order: SumcheckPointOrder::AsIs,
                degree: 2,
            },
            &[
                StageNamedEval {
                    name: "eval.a",
                    oracle: "A",
                    value: Fr::from_u64(2),
                },
                StageNamedEval {
                    name: "eval.b",
                    oracle: "B",
                    value: Fr::from_u64(3),
                },
            ],
            &[],
            &[],
            &[],
        )
        .unwrap();

        assert_eq!(value, Fr::from_u64(6));
    }

    #[test]
    fn relation_output_rejects_structured_polynomial_dimension_mismatch() {
        let x = [Fr::from_u64(1)];
        let y = [Fr::from_u64(1), Fr::from_u64(0)];
        let mut store = ValueStore::default();
        store.insert_point("point.y", y.to_vec());

        let error = evaluate_relation_output_for_instance(
            &[RelationOutputPlan {
                relation: TestRelation::Output,
                local_scalars: &[],
                expected_output: "eq.xy",
            }],
            &[],
            &[ScalarExprPlan {
                symbol: "eq.xy",
                kind: ScalarExprKind::StructuredPolynomial {
                    polynomial: StructuredPolynomialKind::Eq,
                    x_point: StructuredPolynomialPointTransform {
                        segment: StructuredPolynomialPointSegment::Full,
                        length: StructuredPolynomialPointLength::Full,
                        order: StructuredPolynomialPointOrder::AsIs,
                    },
                    y_point: StructuredPolynomialPointTransform {
                        segment: StructuredPolynomialPointSegment::Full,
                        length: StructuredPolynomialPointLength::Full,
                        order: StructuredPolynomialPointOrder::AsIs,
                    },
                },
                operands: &["instance", "point.y"],
            }],
            &[],
            &store,
            &SumcheckInstanceResultPlan {
                symbol: "instance",
                source: "driver",
                claim: "claim",
                relation: TestRelation::Output,
                index: 0,
                point_arity: 1,
                num_rounds: 1,
                round_offset: 0,
                point_order: SumcheckPointOrder::AsIs,
                degree: 2,
            },
            &[],
            &[],
            &[],
            &x,
        )
        .unwrap_err();

        assert_eq!(
            error,
            RuntimePlanError::InvalidInputLength {
                input: "sumcheck_output.eq",
                expected: 1,
                actual: 2
            }
        );
    }
}
