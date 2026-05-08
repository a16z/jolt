//! Stage 7 coarse-kernel ABI used by Bolt-generated Jolt prover code.
//!
//! Stage 7 fuses hamming-weight claim reduction with the address reduction for
//! all RA one-hot polynomials. This module owns the prover-side runtime behind
//! the generated `jolt.stage7.*` CPU ABI.

use std::error::Error;
use std::fmt::{self, Display, Formatter};

use crate::dense::bind_dense_evals_reuse;
use jolt_field::Field;
use jolt_poly::{EqPolynomial, UnivariatePoly};
use jolt_sumcheck::SumcheckProof;
use jolt_transcript::{Label, LabelWithCount, Transcript};
use jolt_witness::Stage6WitnessSlices;
use rayon::prelude::*;

pub use crate::stage6::{
    Stage6ChallengeVector as Stage7ChallengeVector,
    Stage6ExecutionArtifacts as Stage7ExecutionArtifacts,
    Stage6ExecutionMode as Stage7ExecutionMode, Stage6FieldConstantPlan as Stage7FieldConstantPlan,
    Stage6FieldExprPlan as Stage7FieldExprPlan, Stage6NamedEval as Stage7NamedEval,
    Stage6OpeningBatchPlan as Stage7OpeningBatchPlan,
    Stage6OpeningClaimEqualityPlan as Stage7OpeningClaimEqualityPlan,
    Stage6OpeningClaimPlan as Stage7OpeningClaimPlan,
    Stage6OpeningClaimValue as Stage7OpeningClaimValue,
    Stage6OpeningInputPlan as Stage7OpeningInputPlan,
    Stage6OpeningInputValue as Stage7OpeningInputValue, Stage6Params as Stage7Params,
    Stage6PointConcatPlan as Stage7PointConcatPlan, Stage6PointSlicePlan as Stage7PointSlicePlan,
    Stage6PointZeroPlan as Stage7PointZeroPlan, Stage6ProgramStepPlan as Stage7ProgramStepPlan,
    Stage6Proof as Stage7Proof, Stage6SumcheckBatchPlan as Stage7SumcheckBatchPlan,
    Stage6SumcheckClaimPlan as Stage7SumcheckClaimPlan,
    Stage6SumcheckDriverPlan as Stage7SumcheckDriverPlan,
    Stage6SumcheckEvalPlan as Stage7SumcheckEvalPlan,
    Stage6SumcheckInstanceResultPlan as Stage7SumcheckInstanceResultPlan,
    Stage6SumcheckOutput as Stage7SumcheckOutput,
    Stage6TranscriptAbsorbBytesPlan as Stage7TranscriptAbsorbBytesPlan,
    Stage6TranscriptSqueezePlan as Stage7TranscriptSqueezePlan,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage7KernelPlan {
    pub symbol: &'static str,
    pub relation: &'static str,
    pub kind: &'static str,
    pub backend: &'static str,
    pub abi: &'static str,
}

impl Stage7KernelPlan {
    pub fn relation_kind(&self) -> Result<Stage7Relation, Stage7KernelError> {
        Stage7Relation::from_symbol(self.relation).ok_or(Stage7KernelError::UnknownRelation {
            relation: self.relation,
        })
    }

    pub fn abi_kind(&self) -> Result<Stage7KernelAbi, Stage7KernelError> {
        Stage7KernelAbi::from_name(self.abi)
            .ok_or(Stage7KernelError::UnknownKernelAbi { abi: self.abi })
    }
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

impl Stage7CpuProgramPlan {
    pub fn claim(&self, symbol: &str) -> Option<&Stage7SumcheckClaimPlan> {
        self.claims.iter().find(|claim| claim.symbol == symbol)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage7Relation {
    HammingWeightClaimReduction,
    Batched,
}

impl Stage7Relation {
    pub fn from_symbol(symbol: &str) -> Option<Self> {
        match symbol {
            "jolt.stage7.hamming_weight_claim_reduction" => Some(Self::HammingWeightClaimReduction),
            "jolt.stage7.batched" => Some(Self::Batched),
            _ => None,
        }
    }

    pub fn symbol(self) -> &'static str {
        match self {
            Self::HammingWeightClaimReduction => "jolt.stage7.hamming_weight_claim_reduction",
            Self::Batched => "jolt.stage7.batched",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage7KernelAbi {
    HammingWeightClaimReduction,
    Batched,
}

impl Stage7KernelAbi {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "jolt_stage7_hamming_weight_claim_reduction" => Some(Self::HammingWeightClaimReduction),
            "jolt_stage7_batched" => Some(Self::Batched),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::HammingWeightClaimReduction => "jolt_stage7_hamming_weight_claim_reduction",
            Self::Batched => "jolt_stage7_batched",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage7KernelError {
    MissingClaim {
        batch: &'static str,
        claim: &'static str,
    },
    MissingValue {
        symbol: &'static str,
    },
    MissingDynamicValue {
        symbol: String,
    },
    MissingDriver {
        driver: &'static str,
    },
    MissingKernel {
        driver: &'static str,
        kernel: &'static str,
    },
    MissingBatch {
        driver: &'static str,
        batch: &'static str,
    },
    UnknownRelation {
        relation: &'static str,
    },
    UnknownKernelAbi {
        abi: &'static str,
    },
    PlanCountMismatch {
        artifact: &'static str,
        expected: usize,
        actual: usize,
    },
    InvalidInputLength {
        input: &'static str,
        expected: usize,
        actual: usize,
    },
    UnsupportedFieldExpr {
        symbol: &'static str,
        formula: &'static str,
    },
    KernelNotImplemented {
        abi: &'static str,
    },
    WrongExecutorMode {
        driver: &'static str,
        expected: Stage7ExecutionMode,
        actual: Stage7ExecutionMode,
    },
    MissingProof {
        driver: &'static str,
    },
    MissingKernelInput {
        kernel: &'static str,
        input: &'static str,
    },
    InvalidProgramStep {
        symbol: &'static str,
        kind: &'static str,
    },
    InvalidProof {
        driver: &'static str,
        reason: &'static str,
    },
}

impl Display for Stage7KernelError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingClaim { batch, claim } => {
                write!(
                    formatter,
                    "stage7 batch @{batch} references missing claim @{claim}"
                )
            }
            Self::MissingValue { symbol } => {
                write!(formatter, "stage7 value @{symbol} is not available")
            }
            Self::MissingDynamicValue { symbol } => {
                write!(formatter, "stage7 value @{symbol} is not available")
            }
            Self::MissingDriver { driver } => {
                write!(formatter, "stage7 driver @{driver} is not available")
            }
            Self::MissingKernel { driver, kernel } => {
                write!(
                    formatter,
                    "stage7 driver @{driver} references missing kernel @{kernel}"
                )
            }
            Self::MissingBatch { driver, batch } => {
                write!(
                    formatter,
                    "stage7 driver @{driver} references missing batch @{batch}"
                )
            }
            Self::UnknownRelation { relation } => {
                write!(formatter, "unknown stage7 relation `{relation}`")
            }
            Self::UnknownKernelAbi { abi } => {
                write!(formatter, "unknown stage7 kernel ABI `{abi}`")
            }
            Self::PlanCountMismatch {
                artifact,
                expected,
                actual,
            } => {
                write!(
                    formatter,
                    "stage7 {artifact} plan count mismatch: expected {expected}, got {actual}"
                )
            }
            Self::InvalidInputLength {
                input,
                expected,
                actual,
            } => {
                write!(
                    formatter,
                    "stage7 input `{input}` has length {actual}, expected {expected}"
                )
            }
            Self::UnsupportedFieldExpr { symbol, formula } => {
                write!(
                    formatter,
                    "stage7 field expr @{symbol} uses unsupported formula `{formula}`"
                )
            }
            Self::KernelNotImplemented { abi } => {
                write!(formatter, "stage7 kernel ABI `{abi}` is not implemented")
            }
            Self::WrongExecutorMode {
                driver,
                expected,
                actual,
            } => {
                write!(
                    formatter,
                    "stage7 driver @{driver} expected {expected:?} executor, got {actual:?}"
                )
            }
            Self::MissingProof { driver } => {
                write!(formatter, "stage7 proof for driver @{driver} is missing")
            }
            Self::MissingKernelInput { kernel, input } => {
                write!(
                    formatter,
                    "stage7 kernel `{kernel}` is missing required input `{input}`"
                )
            }
            Self::InvalidProgramStep { symbol, kind } => {
                write!(
                    formatter,
                    "stage7 program step @{symbol} has invalid kind `{kind}`"
                )
            }
            Self::InvalidProof { driver, reason } => {
                write!(
                    formatter,
                    "stage7 proof for driver @{driver} is invalid: {reason}"
                )
            }
        }
    }
}

impl Error for Stage7KernelError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage7RaChunkLayout {
    CycleMajor,
    AddressMajor,
}

#[derive(Clone, Copy)]
pub struct Stage7RaChunks<'a, F: Field> {
    pub chunks: &'a [&'a [F]],
    pub layout: Stage7RaChunkLayout,
}

#[derive(Clone, Copy)]
pub struct Stage7HammingWeightClaimReductionWitness<'a, F: Field> {
    pub instruction_ra: Stage7RaChunks<'a, F>,
    pub bytecode_ra: Stage7RaChunks<'a, F>,
    pub ram_ra: Stage7RaChunks<'a, F>,
}

#[derive(Clone, Copy)]
pub struct Stage7RaIndexChunks<'a> {
    pub chunks: &'a [&'a [Option<u8>]],
}

#[derive(Clone, Copy)]
pub struct Stage7HammingWeightClaimReductionIndexWitness<'a> {
    pub instruction_ra: Stage7RaIndexChunks<'a>,
    pub bytecode_ra: Stage7RaIndexChunks<'a>,
    pub ram_ra: Stage7RaIndexChunks<'a>,
}

#[derive(Clone, Copy)]
pub struct Stage7ProverInputs<'a, F: Field> {
    pub opening_inputs: &'a [Stage7OpeningInputValue<F>],
    pub hamming_weight_claim_reduction: Option<Stage7HammingWeightClaimReductionWitness<'a, F>>,
    pub hamming_weight_claim_reduction_indices:
        Option<Stage7HammingWeightClaimReductionIndexWitness<'a>>,
}

impl<'a, F: Field> Stage7ProverInputs<'a, F> {
    pub fn new(opening_inputs: &'a [Stage7OpeningInputValue<F>]) -> Self {
        Self {
            opening_inputs,
            hamming_weight_claim_reduction: None,
            hamming_weight_claim_reduction_indices: None,
        }
    }

    pub fn empty() -> Self {
        Self {
            opening_inputs: &[],
            hamming_weight_claim_reduction: None,
            hamming_weight_claim_reduction_indices: None,
        }
    }

    pub fn with_hamming_weight_claim_reduction(
        mut self,
        witness: Stage7HammingWeightClaimReductionWitness<'a, F>,
    ) -> Self {
        self.hamming_weight_claim_reduction = Some(witness);
        self
    }

    pub fn with_hamming_weight_claim_reduction_indices(
        mut self,
        witness: Stage7HammingWeightClaimReductionIndexWitness<'a>,
    ) -> Self {
        self.hamming_weight_claim_reduction_indices = Some(witness);
        self
    }

    pub fn with_stage6_witness_indices(self, slices: &'a Stage6WitnessSlices<'a, F>) -> Self {
        self.with_hamming_weight_claim_reduction_indices(
            Stage7HammingWeightClaimReductionIndexWitness {
                instruction_ra: Stage7RaIndexChunks {
                    chunks: &slices.instruction_ra_index_chunks,
                },
                bytecode_ra: Stage7RaIndexChunks {
                    chunks: &slices.bytecode_ra_index_chunks,
                },
                ram_ra: Stage7RaIndexChunks {
                    chunks: &slices.ram_ra_index_chunks,
                },
            },
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Stage7KernelContext<'a> {
    pub mode: Stage7ExecutionMode,
    pub program: &'static Stage7CpuProgramPlan,
    pub kernel: &'a Stage7KernelPlan,
    pub batch: &'a Stage7SumcheckBatchPlan,
    pub driver: &'a Stage7SumcheckDriverPlan,
}

impl Stage7KernelContext<'_> {
    pub fn relation_kind(&self) -> Result<Stage7Relation, Stage7KernelError> {
        self.kernel.relation_kind()
    }

    pub fn abi_kind(&self) -> Result<Stage7KernelAbi, Stage7KernelError> {
        self.kernel.abi_kind()
    }

    pub fn batch_claims(&self) -> Result<Vec<&'static Stage7SumcheckClaimPlan>, Stage7KernelError> {
        self.batch
            .claim_operands
            .iter()
            .map(|symbol| {
                self.program
                    .claim(symbol)
                    .ok_or(Stage7KernelError::MissingClaim {
                        batch: self.batch.symbol,
                        claim: symbol,
                    })
            })
            .collect()
    }
}

pub trait Stage7KernelExecutor<F: Field> {
    fn observe_challenge_vector(
        &mut self,
        _plan: &'static Stage7TranscriptSqueezePlan,
        _values: &[F],
    ) -> Result<(), Stage7KernelError> {
        Ok(())
    }

    fn observe_sumcheck_output(
        &mut self,
        _output: &Stage7SumcheckOutput<F>,
    ) -> Result<(), Stage7KernelError> {
        Ok(())
    }

    fn prove_sumcheck<T>(
        &mut self,
        context: Stage7KernelContext<'_>,
        transcript: &mut T,
    ) -> Result<Stage7SumcheckOutput<F>, Stage7KernelError>
    where
        T: Transcript<Challenge = F>;

    fn verify_sumcheck<T>(
        &mut self,
        context: Stage7KernelContext<'_>,
        transcript: &mut T,
    ) -> Result<Stage7SumcheckOutput<F>, Stage7KernelError>
    where
        T: Transcript<Challenge = F>;
}

#[derive(Clone, Debug, Default)]
pub struct UnsupportedStage7KernelExecutor;

impl<F: Field> Stage7KernelExecutor<F> for UnsupportedStage7KernelExecutor {
    fn prove_sumcheck<T>(
        &mut self,
        context: Stage7KernelContext<'_>,
        _transcript: &mut T,
    ) -> Result<Stage7SumcheckOutput<F>, Stage7KernelError>
    where
        T: Transcript<Challenge = F>,
    {
        let abi = context.abi_kind()?;
        let _ = context.relation_kind()?;
        Err(Stage7KernelError::KernelNotImplemented { abi: abi.name() })
    }

    fn verify_sumcheck<T>(
        &mut self,
        context: Stage7KernelContext<'_>,
        _transcript: &mut T,
    ) -> Result<Stage7SumcheckOutput<F>, Stage7KernelError>
    where
        T: Transcript<Challenge = F>,
    {
        let abi = context.abi_kind()?;
        let _ = context.relation_kind()?;
        Err(Stage7KernelError::KernelNotImplemented { abi: abi.name() })
    }
}

#[derive(Clone)]
pub struct Stage7ProverKernelExecutor<'a, F: Field> {
    pub inputs: Stage7ProverInputs<'a, F>,
    challenge_vectors: Vec<Stage7ChallengeVector<F>>,
    completed_sumchecks: Vec<Stage7SumcheckOutput<F>>,
}

impl<'a, F: Field> Stage7ProverKernelExecutor<'a, F> {
    pub fn new(inputs: Stage7ProverInputs<'a, F>) -> Self {
        Self {
            inputs,
            challenge_vectors: Vec::new(),
            completed_sumchecks: Vec::new(),
        }
    }

    fn value_store(
        &self,
        program: &'static Stage7CpuProgramPlan,
    ) -> Result<Stage7ValueStore<F>, Stage7KernelError> {
        value_store_from_observations(
            program,
            self.inputs.opening_inputs,
            &self.challenge_vectors,
            &self.completed_sumchecks,
        )
    }
}

impl<F: Field> Stage7KernelExecutor<F> for Stage7ProverKernelExecutor<'_, F> {
    fn observe_challenge_vector(
        &mut self,
        plan: &'static Stage7TranscriptSqueezePlan,
        values: &[F],
    ) -> Result<(), Stage7KernelError> {
        self.challenge_vectors.push(Stage7ChallengeVector {
            symbol: plan.symbol,
            values: values.to_vec(),
        });
        Ok(())
    }

    fn observe_sumcheck_output(
        &mut self,
        output: &Stage7SumcheckOutput<F>,
    ) -> Result<(), Stage7KernelError> {
        self.completed_sumchecks.push(output.clone());
        Ok(())
    }

    fn prove_sumcheck<T>(
        &mut self,
        context: Stage7KernelContext<'_>,
        transcript: &mut T,
    ) -> Result<Stage7SumcheckOutput<F>, Stage7KernelError>
    where
        T: Transcript<Challenge = F>,
    {
        prove_stage7_kernel(
            context,
            &self.inputs,
            self.value_store(context.program)?,
            transcript,
        )
    }

    fn verify_sumcheck<T>(
        &mut self,
        context: Stage7KernelContext<'_>,
        _transcript: &mut T,
    ) -> Result<Stage7SumcheckOutput<F>, Stage7KernelError>
    where
        T: Transcript<Challenge = F>,
    {
        Err(Stage7KernelError::WrongExecutorMode {
            driver: context.driver.symbol,
            expected: Stage7ExecutionMode::Prover,
            actual: Stage7ExecutionMode::Verifier,
        })
    }
}

#[derive(Clone)]
pub struct Stage7ProofCarryingKernelExecutor<'a, F: Field> {
    pub proof: &'a Stage7Proof<F>,
    pub opening_inputs: &'a [Stage7OpeningInputValue<F>],
    pub cursor: usize,
    challenge_vectors: Vec<Stage7ChallengeVector<F>>,
    completed_sumchecks: Vec<Stage7SumcheckOutput<F>>,
}

impl<'a, F: Field> Stage7ProofCarryingKernelExecutor<'a, F> {
    pub fn new(
        proof: &'a Stage7Proof<F>,
        opening_inputs: &'a [Stage7OpeningInputValue<F>],
    ) -> Self {
        Self {
            proof,
            opening_inputs,
            cursor: 0,
            challenge_vectors: Vec::new(),
            completed_sumchecks: Vec::new(),
        }
    }

    fn value_store(
        &self,
        program: &'static Stage7CpuProgramPlan,
    ) -> Result<Stage7ValueStore<F>, Stage7KernelError> {
        value_store_from_observations(
            program,
            self.opening_inputs,
            &self.challenge_vectors,
            &self.completed_sumchecks,
        )
    }

    fn next_proof(
        &mut self,
        driver: &'static str,
    ) -> Result<&'a Stage7SumcheckOutput<F>, Stage7KernelError> {
        let proof = self
            .proof
            .sumchecks
            .get(self.cursor)
            .ok_or(Stage7KernelError::MissingProof { driver })?;
        self.cursor += 1;
        Ok(proof)
    }
}

impl<F: Field> Stage7KernelExecutor<F> for Stage7ProofCarryingKernelExecutor<'_, F> {
    fn observe_challenge_vector(
        &mut self,
        plan: &'static Stage7TranscriptSqueezePlan,
        values: &[F],
    ) -> Result<(), Stage7KernelError> {
        self.challenge_vectors.push(Stage7ChallengeVector {
            symbol: plan.symbol,
            values: values.to_vec(),
        });
        Ok(())
    }

    fn observe_sumcheck_output(
        &mut self,
        output: &Stage7SumcheckOutput<F>,
    ) -> Result<(), Stage7KernelError> {
        self.completed_sumchecks.push(output.clone());
        Ok(())
    }

    fn prove_sumcheck<T>(
        &mut self,
        context: Stage7KernelContext<'_>,
        transcript: &mut T,
    ) -> Result<Stage7SumcheckOutput<F>, Stage7KernelError>
    where
        T: Transcript<Challenge = F>,
    {
        let proof = self.next_proof(context.driver.symbol)?;
        verify_stage7_kernel(
            context,
            self.value_store(context.program)?,
            proof,
            transcript,
        )
    }

    fn verify_sumcheck<T>(
        &mut self,
        context: Stage7KernelContext<'_>,
        transcript: &mut T,
    ) -> Result<Stage7SumcheckOutput<F>, Stage7KernelError>
    where
        T: Transcript<Challenge = F>,
    {
        let proof = self.next_proof(context.driver.symbol)?;
        verify_stage7_kernel(
            context,
            self.value_store(context.program)?,
            proof,
            transcript,
        )
    }
}

#[derive(Clone, Debug, Default)]
struct Stage7ValueStore<F: Field> {
    scalars: Vec<(&'static str, F)>,
    points: Vec<(&'static str, Vec<F>)>,
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

    fn seed_constants(&mut self, program: &'static Stage7CpuProgramPlan) {
        for constant in program.field_constants {
            self.insert_scalar(constant.symbol, F::from_u64(constant.value as u64));
        }
        for zero in program.point_zeros {
            self.insert_point(zero.symbol, vec![F::from_u64(0); zero.arity]);
        }
    }

    fn observe_challenge_vector(
        &mut self,
        plan: &'static Stage7TranscriptSqueezePlan,
        values: &[F],
    ) -> Result<(), Stage7KernelError> {
        self.insert_point(plan.symbol, values.to_vec());
        if matches!(plan.kind, "challenge_scalar" | "scalar") {
            require_operand_count(plan.symbol, 1, values.len())?;
            self.insert_scalar(plan.symbol, values[0]);
        }
        Ok(())
    }

    fn observe_sumcheck_output(
        &mut self,
        program: &'static Stage7CpuProgramPlan,
        output: &Stage7SumcheckOutput<F>,
    ) -> Result<(), Stage7KernelError> {
        self.observe_sumcheck_values(program, output.driver, &output.point, &output.evals)
    }

    fn observe_sumcheck_values(
        &mut self,
        program: &'static Stage7CpuProgramPlan,
        driver: &'static str,
        point: &[F],
        evals: &[Stage7NamedEval<F>],
    ) -> Result<(), Stage7KernelError> {
        self.insert_point(driver, point.to_vec());
        for instance in program
            .instance_results
            .iter()
            .filter(|instance| instance.source == driver)
        {
            let end = instance.round_offset + instance.point_arity;
            let mut point = point
                .get(instance.round_offset..end)
                .ok_or(Stage7KernelError::InvalidInputLength {
                    input: instance.symbol,
                    expected: end,
                    actual: point.len(),
                })?
                .to_vec();
            match instance.point_order {
                "as_is" => {}
                "reverse" => point.reverse(),
                _ => {
                    return Err(Stage7KernelError::InvalidProof {
                        driver,
                        reason: "unsupported point order",
                    });
                }
            }
            self.insert_point(instance.symbol, point);
        }
        for eval in program.evals.iter().filter(|eval| eval.source == driver) {
            let value = evals
                .iter()
                .find(|value| value.name == eval.name)
                .or_else(|| evals.get(eval.index))
                .ok_or(Stage7KernelError::MissingValue {
                    symbol: eval.symbol,
                })?
                .value;
            self.insert_scalar(eval.symbol, value);
            self.insert_scalar(eval.name, value);
        }
        let _ = self.evaluate_available_points(program)?;
        let _ = self.evaluate_available_field_exprs(program)?;
        self.verify_opening_equalities(program)?;
        Ok(())
    }

    fn evaluate_available_field_exprs(
        &mut self,
        program: &'static Stage7CpuProgramPlan,
    ) -> Result<usize, Stage7KernelError> {
        let mut inserted = 0usize;
        loop {
            let mut progress = 0usize;
            for expr in program.field_exprs {
                if self.try_scalar(expr.symbol).is_some() {
                    continue;
                }
                let Some(operands) = self.try_expr_operands(expr) else {
                    continue;
                };
                self.insert_scalar(expr.symbol, evaluate_stage7_field_expr(expr, &operands)?);
                progress += 1;
            }
            inserted += progress;
            if progress == 0 {
                return Ok(inserted);
            }
        }
    }

    fn evaluate_available_points(
        &mut self,
        program: &'static Stage7CpuProgramPlan,
    ) -> Result<usize, Stage7KernelError> {
        let mut inserted = 0usize;
        loop {
            let mut progress = 0usize;
            for slice in program.point_slices {
                if self.try_point(slice.symbol).is_some() {
                    continue;
                }
                let Some(input) = self.try_point(slice.input) else {
                    continue;
                };
                let end = slice.offset + slice.length;
                let point = input
                    .get(slice.offset..end)
                    .ok_or(Stage7KernelError::InvalidInputLength {
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
                let Some(point) = self.try_concat_point(concat) else {
                    continue;
                };
                require_operand_count(concat.symbol, concat.arity, point.len())?;
                self.insert_point(concat.symbol, point);
                progress += 1;
            }
            inserted += progress;
            if progress == 0 {
                return Ok(inserted);
            }
        }
    }

    fn verify_opening_equalities(
        &self,
        program: &'static Stage7CpuProgramPlan,
    ) -> Result<(), Stage7KernelError> {
        for equality in program.opening_equalities {
            match equality.mode {
                "point_and_eval" => {
                    if self.point(equality.lhs)? != self.point(equality.rhs)?
                        || self.scalar(equality.lhs)? != self.scalar(equality.rhs)?
                    {
                        return Err(Stage7KernelError::InvalidProof {
                            driver: equality.symbol,
                            reason: "opening claim equality failed",
                        });
                    }
                }
                _ => {
                    return Err(Stage7KernelError::InvalidProof {
                        driver: equality.symbol,
                        reason: "unsupported opening equality mode",
                    });
                }
            }
        }
        Ok(())
    }

    fn claim_value(
        &mut self,
        program: &'static Stage7CpuProgramPlan,
        claim: &Stage7SumcheckClaimPlan,
    ) -> Result<F, Stage7KernelError> {
        let _ = self.evaluate_available_field_exprs(program)?;
        self.scalar(claim.claim_value)
    }

    fn batch_claim_values(
        &mut self,
        program: &'static Stage7CpuProgramPlan,
        batch: &Stage7SumcheckBatchPlan,
    ) -> Result<Vec<F>, Stage7KernelError> {
        batch
            .claim_operands
            .iter()
            .map(|symbol| {
                let claim = program
                    .claim(symbol)
                    .ok_or(Stage7KernelError::MissingClaim {
                        batch: batch.symbol,
                        claim: symbol,
                    })?;
                self.claim_value(program, claim)
            })
            .collect()
    }

    fn insert_scalar(&mut self, symbol: &'static str, value: F) {
        if let Some((_, existing)) = self
            .scalars
            .iter_mut()
            .find(|(existing, _)| *existing == symbol)
        {
            *existing = value;
        } else {
            self.scalars.push((symbol, value));
        }
    }

    fn insert_point(&mut self, symbol: &'static str, point: Vec<F>) {
        if let Some((_, existing)) = self
            .points
            .iter_mut()
            .find(|(existing, _)| *existing == symbol)
        {
            *existing = point;
        } else {
            self.points.push((symbol, point));
        }
    }

    fn scalar(&self, symbol: &'static str) -> Result<F, Stage7KernelError> {
        self.try_scalar(symbol)
            .ok_or(Stage7KernelError::MissingValue { symbol })
    }

    fn try_scalar(&self, symbol: &str) -> Option<F> {
        self.scalars
            .iter()
            .find(|(existing, _)| *existing == symbol)
            .map(|(_, value)| *value)
    }

    fn point(&self, symbol: &'static str) -> Result<&[F], Stage7KernelError> {
        self.try_point(symbol)
            .ok_or(Stage7KernelError::MissingValue { symbol })
    }

    fn dynamic_point(&self, symbol: String) -> Result<&[F], Stage7KernelError> {
        self.try_point(&symbol)
            .ok_or(Stage7KernelError::MissingDynamicValue { symbol })
    }

    fn try_point(&self, symbol: &str) -> Option<&[F]> {
        self.points
            .iter()
            .find(|(existing, _)| *existing == symbol)
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

fn value_store_from_observations<F: Field>(
    program: &'static Stage7CpuProgramPlan,
    opening_inputs: &[Stage7OpeningInputValue<F>],
    challenge_vectors: &[Stage7ChallengeVector<F>],
    completed_sumchecks: &[Stage7SumcheckOutput<F>],
) -> Result<Stage7ValueStore<F>, Stage7KernelError> {
    let mut store = Stage7ValueStore::with_opening_inputs(opening_inputs);
    store.seed_constants(program);
    for challenge in challenge_vectors {
        let plan =
            find_squeeze(program, challenge.symbol).ok_or(Stage7KernelError::MissingValue {
                symbol: challenge.symbol,
            })?;
        store.observe_challenge_vector(plan, &challenge.values)?;
    }
    for output in completed_sumchecks {
        store.observe_sumcheck_output(program, output)?;
    }
    let _ = store.evaluate_available_points(program)?;
    let _ = store.evaluate_available_field_exprs(program)?;
    store.verify_opening_equalities(program)?;
    Ok(store)
}

fn prove_stage7_kernel<F, T>(
    context: Stage7KernelContext<'_>,
    inputs: &Stage7ProverInputs<'_, F>,
    store: Stage7ValueStore<F>,
    transcript: &mut T,
) -> Result<Stage7SumcheckOutput<F>, Stage7KernelError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    match context.abi_kind()? {
        Stage7KernelAbi::Batched => prove_batched_stage7(context, inputs, store, transcript),
        abi @ Stage7KernelAbi::HammingWeightClaimReduction => {
            Err(Stage7KernelError::KernelNotImplemented { abi: abi.name() })
        }
    }
}

fn verify_stage7_kernel<F, T>(
    context: Stage7KernelContext<'_>,
    store: Stage7ValueStore<F>,
    proof: &Stage7SumcheckOutput<F>,
    transcript: &mut T,
) -> Result<Stage7SumcheckOutput<F>, Stage7KernelError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    match context.abi_kind()? {
        Stage7KernelAbi::Batched => verify_batched_stage7(context, store, proof, transcript),
        abi @ Stage7KernelAbi::HammingWeightClaimReduction => {
            Err(Stage7KernelError::KernelNotImplemented { abi: abi.name() })
        }
    }
}

#[tracing::instrument(skip_all, name = "Stage7::prove_batched")]
fn prove_batched_stage7<F, T>(
    context: Stage7KernelContext<'_>,
    inputs: &Stage7ProverInputs<'_, F>,
    mut store: Stage7ValueStore<F>,
    transcript: &mut T,
) -> Result<Stage7SumcheckOutput<F>, Stage7KernelError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    let claims = context.batch_claims()?;
    let input_claims = store.batch_claim_values(context.program, context.batch)?;
    for claim in &input_claims {
        append_labeled_scalar(transcript, context.batch.claim_label, claim);
    }
    let batching_coeffs = transcript.challenge_vector(claims.len());
    let max_rounds = context.driver.num_rounds;
    let two_inv = F::from_u64(2)
        .inverse()
        .ok_or(Stage7KernelError::InvalidProof {
            driver: context.driver.symbol,
            reason: "field element 2 is not invertible",
        })?;
    let mut instances = Vec::with_capacity(claims.len());
    for (index, claim) in claims.iter().enumerate() {
        let offset = instance_round_offset(context.program, context.driver.symbol, claim.symbol)?;
        if offset + claim.num_rounds > max_rounds {
            return Err(Stage7KernelError::InvalidInputLength {
                input: claim.symbol,
                expected: max_rounds,
                actual: offset + claim.num_rounds,
            });
        }
        let relation = claim_relation(context.program, claim)?;
        let active_scale = F::one().mul_pow_2(max_rounds - offset - claim.num_rounds);
        let state =
            Stage7ProverInstanceState::new(context.program, claim, inputs, &store, active_scale)?;
        instances.push(Stage7BatchedInstance {
            claim,
            relation,
            offset,
            previous_claim: input_claims[index].mul_pow_2(max_rounds - claim.num_rounds),
            state,
        });
    }

    let mut point = Vec::with_capacity(max_rounds);
    let mut round_polynomials = Vec::with_capacity(max_rounds);
    let mut batched_claim = instances
        .iter()
        .zip(&batching_coeffs)
        .map(|(instance, &coefficient)| instance.previous_claim * coefficient)
        .sum::<F>();
    for round in 0..max_rounds {
        let mut individual_polys = Vec::with_capacity(instances.len());
        for instance in &mut instances {
            let poly = if instance.is_active(round) {
                instance
                    .state
                    .round_poly(instance.previous_claim, instance.relation)?
            } else {
                UnivariatePoly::new(vec![instance.previous_claim * two_inv])
            };
            individual_polys.push(poly);
        }
        let batched_poly = combine_univariate_polys(&individual_polys, &batching_coeffs);
        #[cfg(debug_assertions)]
        {
            if batched_poly.evaluate(F::zero()) + batched_poly.evaluate(F::one()) != batched_claim {
                return Err(Stage7KernelError::InvalidProof {
                    driver: context.driver.symbol,
                    reason: "batched round claim mismatch",
                });
            }
        }
        append_compressed_univariate_poly(transcript, context.driver.round_label, &batched_poly);
        let challenge = transcript.challenge();
        point.push(challenge);
        batched_claim = batched_poly.evaluate(challenge);
        for (instance, poly) in instances.iter_mut().zip(individual_polys) {
            instance.previous_claim = poly.evaluate(challenge);
            if instance.is_active(round) {
                instance.state.ingest_challenge(challenge);
            }
        }
        round_polynomials.push(batched_poly);
    }

    let mut evals = Vec::new();
    let mut expected = F::zero();
    for (instance, &coefficient) in instances.iter().zip(&batching_coeffs) {
        let relation_claim = instance.state.final_relation_eval(instance.relation)?;
        if instance.previous_claim != relation_claim {
            return Err(Stage7KernelError::InvalidProof {
                driver: instance.relation.symbol(),
                reason: "stage7 relation output claim mismatch",
            });
        }
        expected += coefficient * relation_claim;
        evals.extend(instance.state.final_evals(instance.relation)?);
    }
    if batched_claim != expected {
        return Err(Stage7KernelError::InvalidProof {
            driver: context.driver.symbol,
            reason: "batched output claim mismatch",
        });
    }
    store.observe_sumcheck_values(context.program, context.driver.symbol, &point, &evals)?;
    let opening_claims = append_opening_claims(context.program, &mut store, transcript, &evals)?;
    Ok(Stage7SumcheckOutput {
        driver: context.driver.symbol,
        point,
        evals,
        opening_claims,
        proof: SumcheckProof { round_polynomials },
    })
}

fn verify_batched_stage7<F, T>(
    context: Stage7KernelContext<'_>,
    mut store: Stage7ValueStore<F>,
    proof: &Stage7SumcheckOutput<F>,
    transcript: &mut T,
) -> Result<Stage7SumcheckOutput<F>, Stage7KernelError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    if proof.driver != context.driver.symbol {
        return Err(Stage7KernelError::InvalidProof {
            driver: context.driver.symbol,
            reason: "driver symbol mismatch",
        });
    }
    if proof.proof.round_polynomials.len() != context.driver.num_rounds {
        return Err(Stage7KernelError::InvalidProof {
            driver: context.driver.symbol,
            reason: "unexpected batched round count",
        });
    }

    let claims = context.batch_claims()?;
    let input_claims = store.batch_claim_values(context.program, context.batch)?;
    for claim in &input_claims {
        append_labeled_scalar(transcript, context.batch.claim_label, claim);
    }
    let batching_coeffs = transcript.challenge_vector(claims.len());
    let max_rounds = context.driver.num_rounds;
    let mut running_claim = input_claims
        .iter()
        .zip(claims.iter())
        .zip(&batching_coeffs)
        .map(|((claim, plan), &coefficient)| {
            claim.mul_pow_2(max_rounds - plan.num_rounds) * coefficient
        })
        .sum::<F>();
    let mut point = Vec::with_capacity(max_rounds);
    for poly in &proof.proof.round_polynomials {
        if polynomial_degree(poly) > context.driver.degree {
            return Err(Stage7KernelError::InvalidProof {
                driver: context.driver.symbol,
                reason: "batched polynomial exceeds degree bound",
            });
        }
        if poly.evaluate(F::zero()) + poly.evaluate(F::one()) != running_claim {
            return Err(Stage7KernelError::InvalidProof {
                driver: context.driver.symbol,
                reason: "batched round check failed",
            });
        }
        append_compressed_univariate_poly(transcript, context.driver.round_label, poly);
        let challenge = transcript.challenge();
        running_claim = poly.evaluate(challenge);
        point.push(challenge);
    }
    if !proof.point.is_empty() && proof.point != point {
        return Err(Stage7KernelError::InvalidProof {
            driver: context.driver.symbol,
            reason: "batched point mismatch",
        });
    }
    let expected =
        expected_batched_output_claim(context, &store, &proof.evals, &point, &batching_coeffs)?;
    if running_claim != expected {
        return Err(Stage7KernelError::InvalidProof {
            driver: context.driver.symbol,
            reason: "batched output claim mismatch",
        });
    }

    let output = Stage7SumcheckOutput {
        driver: context.driver.symbol,
        point,
        evals: proof.evals.clone(),
        opening_claims: Vec::new(),
        proof: proof.proof.clone(),
    };
    store.observe_sumcheck_output(context.program, &output)?;
    let opening_claims =
        append_opening_claims(context.program, &mut store, &mut *transcript, &output.evals)?;
    let output = Stage7SumcheckOutput {
        opening_claims,
        ..output
    };
    Ok(output)
}

fn expected_batched_output_claim<F: Field>(
    context: Stage7KernelContext<'_>,
    store: &Stage7ValueStore<F>,
    evals: &[Stage7NamedEval<F>],
    point: &[F],
    batching_coeffs: &[F],
) -> Result<F, Stage7KernelError> {
    let mut expected = F::zero();
    for (claim, &coefficient) in context.batch_claims()?.iter().zip(batching_coeffs) {
        let Some(instance) = context.program.instance_results.iter().find(|instance| {
            instance.claim == claim.symbol && instance.source == context.driver.symbol
        }) else {
            return Err(Stage7KernelError::MissingValue {
                symbol: claim.symbol,
            });
        };
        let local_point = point
            .get(instance.round_offset..instance.round_offset + instance.num_rounds)
            .ok_or(Stage7KernelError::InvalidInputLength {
                input: instance.symbol,
                expected: instance.round_offset + instance.num_rounds,
                actual: point.len(),
            })?;
        let value = match claim_relation(context.program, claim)? {
            Stage7Relation::HammingWeightClaimReduction => {
                expected_hamming_weight_claim_reduction(context.program, store, evals, local_point)?
            }
            Stage7Relation::Batched => {
                return Err(Stage7KernelError::InvalidProof {
                    driver: context.driver.symbol,
                    reason: "nested batched relation is unsupported",
                });
            }
        };
        expected += coefficient * value;
    }
    Ok(expected)
}

fn expected_hamming_weight_claim_reduction<F: Field>(
    program: &'static Stage7CpuProgramPlan,
    store: &Stage7ValueStore<F>,
    evals: &[Stage7NamedEval<F>],
    local_point: &[F],
) -> Result<F, Stage7KernelError> {
    let log_k_chunk = local_point.len();
    let booleanity_point = store.point("stage7.input.stage6.booleanity.InstructionRa_0")?;
    if booleanity_point.len() < log_k_chunk {
        return Err(Stage7KernelError::InvalidInputLength {
            input: "stage7.input.stage6.booleanity.InstructionRa_0",
            expected: log_k_chunk,
            actual: booleanity_point.len(),
        });
    }
    let r_addr_bool = &booleanity_point[..log_k_chunk];
    let rho_rev = reverse_slice(local_point);
    let eq_bool_eval = EqPolynomial::<F>::mle(&rho_rev, r_addr_bool);
    let gamma = store.scalar("stage7.hamming_weight_claim_reduction.gamma")?;
    let gamma_powers = gamma_powers(gamma, 3 * ra_eval_plans(program).len());

    let mut output_claim = F::zero();
    for (index, eval_plan) in ra_eval_plans(program).iter().enumerate() {
        let g_i = eval_by_name(evals, eval_plan.name)?;
        let virtual_point =
            store.dynamic_point(stage7_virtualization_input_symbol(eval_plan.oracle)?)?;
        if virtual_point.len() < log_k_chunk {
            return Err(Stage7KernelError::InvalidInputLength {
                input: "stage7.hamming_weight_claim_reduction.virtualization_point",
                expected: log_k_chunk,
                actual: virtual_point.len(),
            });
        }
        let eq_virt_eval = EqPolynomial::<F>::mle(&rho_rev, &virtual_point[..log_k_chunk]);
        output_claim += g_i
            * (gamma_powers[3 * index]
                + gamma_powers[3 * index + 1] * eq_bool_eval
                + gamma_powers[3 * index + 2] * eq_virt_eval);
    }
    Ok(output_claim)
}

struct Stage7BatchedInstance<'a, F: Field> {
    claim: &'a Stage7SumcheckClaimPlan,
    relation: Stage7Relation,
    offset: usize,
    previous_claim: F,
    state: Stage7ProverInstanceState<F>,
}

impl<F: Field> Stage7BatchedInstance<'_, F> {
    fn is_active(&self, round: usize) -> bool {
        round >= self.offset && round < self.offset + self.claim.num_rounds
    }
}

enum Stage7ProverInstanceState<F: Field> {
    HammingWeightClaimReduction(HammingWeightClaimReductionState<F>),
}

impl<F: Field> Stage7ProverInstanceState<F> {
    fn new(
        program: &'static Stage7CpuProgramPlan,
        claim: &Stage7SumcheckClaimPlan,
        inputs: &Stage7ProverInputs<'_, F>,
        store: &Stage7ValueStore<F>,
        active_scale: F,
    ) -> Result<Self, Stage7KernelError> {
        match claim_relation(program, claim)? {
            Stage7Relation::HammingWeightClaimReduction => {
                hamming_weight_claim_reduction_state(program, claim, inputs, store, active_scale)
                    .map(Self::HammingWeightClaimReduction)
            }
            relation @ Stage7Relation::Batched => Err(Stage7KernelError::KernelNotImplemented {
                abi: relation.symbol(),
            }),
        }
    }

    fn round_poly(
        &mut self,
        previous_claim: F,
        relation: Stage7Relation,
    ) -> Result<UnivariatePoly<F>, Stage7KernelError> {
        match self {
            Self::HammingWeightClaimReduction(state) => state.round_poly(previous_claim, relation),
        }
    }

    fn ingest_challenge(&mut self, challenge: F) {
        match self {
            Self::HammingWeightClaimReduction(state) => state.bind(challenge),
        }
    }

    fn final_relation_eval(&self, relation: Stage7Relation) -> Result<F, Stage7KernelError> {
        match self {
            Self::HammingWeightClaimReduction(state) => state.final_relation_eval(relation),
        }
    }

    fn final_evals(
        &self,
        relation: Stage7Relation,
    ) -> Result<Vec<Stage7NamedEval<F>>, Stage7KernelError> {
        match self {
            Self::HammingWeightClaimReduction(state) => state.final_evals(relation),
        }
    }
}

struct HammingWeightClaimReductionState<F: Field> {
    g: Vec<Vec<F>>,
    eq_bool: Vec<F>,
    eq_virt: Vec<Vec<F>>,
    gamma_powers: Vec<F>,
    outputs: Vec<Stage7RaOutputPlan>,
    active_scale: F,
}

#[derive(Clone, Copy)]
struct Stage7RaOutputPlan {
    name: &'static str,
    oracle: &'static str,
}

impl<F: Field> HammingWeightClaimReductionState<F> {
    fn round_poly(
        &mut self,
        previous_claim: F,
        relation: Stage7Relation,
    ) -> Result<UnivariatePoly<F>, Stage7KernelError> {
        if relation != Stage7Relation::HammingWeightClaimReduction {
            return Err(Stage7KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "wrong relation for hamming-weight claim-reduction state",
            });
        }
        let half_len = self
            .g
            .first()
            .ok_or(Stage7KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "missing G polynomial",
            })?
            .len()
            / 2;
        let mut evals = [F::zero(); 2];
        for row in 0..half_len {
            let eq_bool_evals = low_to_high_linear_evals(&self.eq_bool, row);
            for index in 0..self.g.len() {
                let g_evals = low_to_high_linear_evals(&self.g[index], row);
                let eq_virt_evals = low_to_high_linear_evals(&self.eq_virt[index], row);
                let gamma_hw = self.gamma_powers[3 * index];
                let gamma_bool = self.gamma_powers[3 * index + 1];
                let gamma_virt = self.gamma_powers[3 * index + 2];
                for eval_index in 0..2 {
                    evals[eval_index] += g_evals[eval_index]
                        * (gamma_hw
                            + gamma_bool * eq_bool_evals[eval_index]
                            + gamma_virt * eq_virt_evals[eval_index]);
                }
            }
        }
        for eval in &mut evals {
            *eval *= self.active_scale;
        }
        Ok(UnivariatePoly::from_evals_and_hint(previous_claim, &evals))
    }

    fn bind(&mut self, challenge: F) {
        let mut scratch = Vec::new();
        for g in &mut self.g {
            bind_dense_evals_reuse(g, &mut scratch, challenge);
        }
        bind_dense_evals_reuse(&mut self.eq_bool, &mut scratch, challenge);
        for eq in &mut self.eq_virt {
            bind_dense_evals_reuse(eq, &mut scratch, challenge);
        }
    }

    fn final_relation_eval(&self, relation: Stage7Relation) -> Result<F, Stage7KernelError> {
        if relation != Stage7Relation::HammingWeightClaimReduction {
            return Err(Stage7KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "wrong relation for hamming-weight claim-reduction state",
            });
        }
        let eq_bool = single_final_eval(&self.eq_bool, "stage7.eq_bool")?;
        let mut value = F::zero();
        for index in 0..self.g.len() {
            let g = single_final_eval(&self.g[index], "stage7.G")?;
            let eq_virt = single_final_eval(&self.eq_virt[index], "stage7.eq_virt")?;
            value += g
                * (self.gamma_powers[3 * index]
                    + self.gamma_powers[3 * index + 1] * eq_bool
                    + self.gamma_powers[3 * index + 2] * eq_virt);
        }
        Ok(value * self.active_scale)
    }

    fn final_evals(
        &self,
        relation: Stage7Relation,
    ) -> Result<Vec<Stage7NamedEval<F>>, Stage7KernelError> {
        if relation != Stage7Relation::HammingWeightClaimReduction {
            return Err(Stage7KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "wrong relation for hamming-weight claim-reduction state",
            });
        }
        if self.outputs.len() != self.g.len() {
            return Err(Stage7KernelError::PlanCountMismatch {
                artifact: "hamming-weight claim-reduction eval outputs",
                expected: self.g.len(),
                actual: self.outputs.len(),
            });
        }
        self.g
            .iter()
            .zip(&self.outputs)
            .map(|(g, output)| {
                Ok(Stage7NamedEval {
                    name: output.name,
                    oracle: output.oracle,
                    value: single_final_eval(g, output.name)?,
                })
            })
            .collect()
    }
}

fn hamming_weight_claim_reduction_state<F: Field>(
    program: &'static Stage7CpuProgramPlan,
    claim: &Stage7SumcheckClaimPlan,
    inputs: &Stage7ProverInputs<'_, F>,
    store: &Stage7ValueStore<F>,
    active_scale: F,
) -> Result<HammingWeightClaimReductionState<F>, Stage7KernelError> {
    let log_k_chunk = claim.num_rounds;
    let booleanity_point = store.point("stage7.input.stage6.booleanity.InstructionRa_0")?;
    if booleanity_point.len() < log_k_chunk {
        return Err(Stage7KernelError::InvalidInputLength {
            input: "stage7.input.stage6.booleanity.InstructionRa_0",
            expected: log_k_chunk,
            actual: booleanity_point.len(),
        });
    }
    let r_addr_bool = &booleanity_point[..log_k_chunk];
    let r_cycle = &booleanity_point[log_k_chunk..];
    let outputs = ra_eval_plans(program);
    let eq_cycle = EqPolynomial::<F>::evals(r_cycle, None);
    let g = if let Some(index_witness) = inputs.hamming_weight_claim_reduction_indices {
        let chunks = index_witness.ra_chunks_in_program_order();
        if chunks.len() != outputs.len() {
            return Err(Stage7KernelError::PlanCountMismatch {
                artifact: "stage7 RA index witness chunks",
                expected: outputs.len(),
                actual: chunks.len(),
            });
        }
        chunks
            .par_iter()
            .map(|chunk| pushforward_ra_indices(chunk.indices, log_k_chunk, &eq_cycle))
            .collect::<Result<Vec<_>, _>>()?
    } else {
        let witness =
            inputs
                .hamming_weight_claim_reduction
                .ok_or(Stage7KernelError::MissingKernelInput {
                    kernel: "jolt_stage7_hamming_weight_claim_reduction",
                    input: "hamming_weight_claim_reduction",
                })?;
        let chunks = witness.ra_chunks_in_program_order();
        if chunks.len() != outputs.len() {
            return Err(Stage7KernelError::PlanCountMismatch {
                artifact: "stage7 RA witness chunks",
                expected: outputs.len(),
                actual: chunks.len(),
            });
        }
        chunks
            .par_iter()
            .map(|chunk| pushforward_ra_chunk(chunk.evals, chunk.layout, log_k_chunk, &eq_cycle))
            .collect::<Result<Vec<_>, _>>()?
    };
    let eq_bool = EqPolynomial::<F>::evals(r_addr_bool, None);
    let mut eq_virt = Vec::with_capacity(outputs.len());
    for output in &outputs {
        let virtual_point =
            store.dynamic_point(stage7_virtualization_input_symbol(output.oracle)?)?;
        if virtual_point.len() < log_k_chunk {
            return Err(Stage7KernelError::InvalidInputLength {
                input: "stage7.hamming_weight_claim_reduction.virtualization_point",
                expected: log_k_chunk,
                actual: virtual_point.len(),
            });
        }
        eq_virt.push(EqPolynomial::<F>::evals(
            &virtual_point[..log_k_chunk],
            None,
        ));
    }
    let gamma = store.scalar("stage7.hamming_weight_claim_reduction.gamma")?;
    let gamma_powers = gamma_powers(gamma, 3 * outputs.len());
    Ok(HammingWeightClaimReductionState {
        g,
        eq_bool,
        eq_virt,
        gamma_powers,
        outputs,
        active_scale,
    })
}

#[derive(Clone, Copy)]
struct Stage7RaChunk<'a, F: Field> {
    evals: &'a [F],
    layout: Stage7RaChunkLayout,
}

#[derive(Clone, Copy)]
struct Stage7RaIndexChunk<'a> {
    indices: &'a [Option<u8>],
}

impl<'a, F: Field> Stage7HammingWeightClaimReductionWitness<'a, F> {
    fn ra_chunks_in_program_order(&self) -> Vec<Stage7RaChunk<'a, F>> {
        let mut chunks = Vec::with_capacity(
            self.instruction_ra.chunks.len()
                + self.bytecode_ra.chunks.len()
                + self.ram_ra.chunks.len(),
        );
        chunks.extend(
            self.instruction_ra
                .chunks
                .iter()
                .map(|chunk| Stage7RaChunk {
                    evals: chunk,
                    layout: self.instruction_ra.layout,
                }),
        );
        chunks.extend(self.bytecode_ra.chunks.iter().map(|chunk| Stage7RaChunk {
            evals: chunk,
            layout: self.bytecode_ra.layout,
        }));
        chunks.extend(self.ram_ra.chunks.iter().map(|chunk| Stage7RaChunk {
            evals: chunk,
            layout: self.ram_ra.layout,
        }));
        chunks
    }
}

impl<'a> Stage7HammingWeightClaimReductionIndexWitness<'a> {
    fn ra_chunks_in_program_order(&self) -> Vec<Stage7RaIndexChunk<'a>> {
        let mut chunks = Vec::with_capacity(
            self.instruction_ra.chunks.len()
                + self.bytecode_ra.chunks.len()
                + self.ram_ra.chunks.len(),
        );
        chunks.extend(
            self.instruction_ra
                .chunks
                .iter()
                .map(|indices| Stage7RaIndexChunk { indices }),
        );
        chunks.extend(
            self.bytecode_ra
                .chunks
                .iter()
                .map(|indices| Stage7RaIndexChunk { indices }),
        );
        chunks.extend(
            self.ram_ra
                .chunks
                .iter()
                .map(|indices| Stage7RaIndexChunk { indices }),
        );
        chunks
    }
}

fn pushforward_ra_chunk<F: Field>(
    chunk: &[F],
    layout: Stage7RaChunkLayout,
    log_k_chunk: usize,
    eq_cycle: &[F],
) -> Result<Vec<F>, Stage7KernelError> {
    let address_len = 1usize << log_k_chunk;
    let cycle_len = eq_cycle.len();
    let expected = address_len * cycle_len;
    if chunk.len() != expected {
        return Err(Stage7KernelError::InvalidInputLength {
            input: "stage7.ra_chunk",
            expected,
            actual: chunk.len(),
        });
    }
    let mut output = vec![F::zero(); address_len];
    match layout {
        Stage7RaChunkLayout::CycleMajor => {
            for (cycle, weight) in eq_cycle.iter().copied().enumerate().take(cycle_len) {
                let row_start = cycle * address_len;
                for address in 0..address_len {
                    output[address] += chunk[row_start + address] * weight;
                }
            }
        }
        Stage7RaChunkLayout::AddressMajor => {
            for (address, output_value) in output.iter_mut().enumerate().take(address_len) {
                let row_start = address * cycle_len;
                for (cycle, weight) in eq_cycle.iter().copied().enumerate().take(cycle_len) {
                    *output_value += chunk[row_start + cycle] * weight;
                }
            }
        }
    }
    Ok(output)
}

fn pushforward_ra_indices<F: Field>(
    indices: &[Option<u8>],
    log_k_chunk: usize,
    eq_cycle: &[F],
) -> Result<Vec<F>, Stage7KernelError> {
    let address_len = 1usize << log_k_chunk;
    if indices.len() != eq_cycle.len() {
        return Err(Stage7KernelError::InvalidInputLength {
            input: "stage7.ra_index_chunk",
            expected: eq_cycle.len(),
            actual: indices.len(),
        });
    }
    let mut output = vec![F::zero(); address_len];
    for (cycle, index) in indices.iter().enumerate() {
        if let Some(index) = index {
            let index = usize::from(*index);
            if index >= address_len {
                return Err(Stage7KernelError::InvalidInputLength {
                    input: "stage7.ra_index",
                    expected: address_len,
                    actual: index + 1,
                });
            }
            output[index] += eq_cycle[cycle];
        }
    }
    Ok(output)
}

fn low_to_high_linear_evals<F: Field>(evals: &[F], row: usize) -> [F; 2] {
    let low = evals[2 * row];
    let high = evals[2 * row + 1];
    [low, low + (high - low) * F::from_u64(2)]
}

fn single_final_eval<F: Field>(evals: &[F], symbol: &'static str) -> Result<F, Stage7KernelError> {
    if evals.len() == 1 {
        Ok(evals[0])
    } else {
        Err(Stage7KernelError::InvalidInputLength {
            input: symbol,
            expected: 1,
            actual: evals.len(),
        })
    }
}

fn gamma_powers<F: Field>(gamma: F, count: usize) -> Vec<F> {
    let mut powers = Vec::with_capacity(count);
    let mut power = F::one();
    for _ in 0..count {
        powers.push(power);
        power *= gamma;
    }
    powers
}

fn ra_eval_plans(program: &'static Stage7CpuProgramPlan) -> Vec<Stage7RaOutputPlan> {
    let mut evals = program
        .evals
        .iter()
        .filter(|eval| {
            eval.name
                .starts_with("stage7.hamming_weight_claim_reduction.eval.")
        })
        .collect::<Vec<_>>();
    evals.sort_by_key(|eval| eval.index);
    evals
        .into_iter()
        .map(|eval| Stage7RaOutputPlan {
            name: eval.name,
            oracle: eval.oracle,
        })
        .collect()
}

fn stage7_virtualization_input_symbol(oracle: &str) -> Result<String, Stage7KernelError> {
    if oracle.starts_with("InstructionRa_") {
        Ok(format!(
            "stage7.input.stage6.instruction_ra_virtual.{oracle}"
        ))
    } else if oracle.starts_with("BytecodeRa_") {
        Ok(format!("stage7.input.stage6.bytecode_read_raf.{oracle}"))
    } else if oracle.starts_with("RamRa_") {
        Ok(format!("stage7.input.stage6.ram_ra_virtual.{oracle}"))
    } else {
        Err(Stage7KernelError::InvalidProof {
            driver: "stage7.hamming_weight_claim_reduction.oracle",
            reason: "unknown RA oracle family",
        })
    }
}

fn eval_by_name<F: Field>(
    evals: &[Stage7NamedEval<F>],
    name: &'static str,
) -> Result<F, Stage7KernelError> {
    evals
        .iter()
        .find(|eval| eval.name == name)
        .map(|eval| eval.value)
        .ok_or(Stage7KernelError::MissingValue { symbol: name })
}

fn claim_relation(
    program: &'static Stage7CpuProgramPlan,
    claim: &Stage7SumcheckClaimPlan,
) -> Result<Stage7Relation, Stage7KernelError> {
    if let Some(relation) = claim.relation {
        return Stage7Relation::from_symbol(relation)
            .ok_or(Stage7KernelError::UnknownRelation { relation });
    }
    let kernel_symbol = claim.kernel.ok_or(Stage7KernelError::MissingKernel {
        driver: claim.symbol,
        kernel: "<missing>",
    })?;
    let kernel = find_kernel(program, kernel_symbol).ok_or(Stage7KernelError::MissingKernel {
        driver: claim.symbol,
        kernel: kernel_symbol,
    })?;
    kernel.relation_kind()
}

fn instance_round_offset(
    program: &'static Stage7CpuProgramPlan,
    driver: &'static str,
    claim: &'static str,
) -> Result<usize, Stage7KernelError> {
    program
        .instance_results
        .iter()
        .find(|instance| instance.source == driver && instance.claim == claim)
        .map(|instance| instance.round_offset)
        .ok_or(Stage7KernelError::MissingValue { symbol: claim })
}

fn evaluate_stage7_field_expr<F: Field>(
    expr: &Stage7FieldExprPlan,
    operands: &[F],
) -> Result<F, Stage7KernelError> {
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
                    Stage7KernelError::UnsupportedFieldExpr {
                        symbol: expr.symbol,
                        formula,
                    }
                })?;
                return Ok(pow_field(operands[0], exponent));
            }
            Err(Stage7KernelError::UnsupportedFieldExpr {
                symbol: expr.symbol,
                formula,
            })
        }
    }
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

fn single_operand<F: Field>(symbol: &'static str, operands: &[F]) -> Result<F, Stage7KernelError> {
    require_operand_count(symbol, 1, operands.len())?;
    Ok(operands[0])
}

fn require_operand_count(
    input: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), Stage7KernelError> {
    if expected == actual {
        Ok(())
    } else {
        Err(Stage7KernelError::InvalidInputLength {
            input,
            expected,
            actual,
        })
    }
}

fn combine_univariate_polys<F: Field>(
    polynomials: &[UnivariatePoly<F>],
    coefficients: &[F],
) -> UnivariatePoly<F> {
    let max_len = polynomials
        .iter()
        .map(|poly| poly.coefficients().len())
        .max()
        .unwrap_or(0);
    let mut combined = vec![F::zero(); max_len];
    for (poly, &coefficient) in polynomials.iter().zip(coefficients) {
        for (combined, &term) in combined.iter_mut().zip(poly.coefficients()) {
            *combined += term * coefficient;
        }
    }
    UnivariatePoly::new(combined)
}

fn polynomial_degree<F: Field>(poly: &UnivariatePoly<F>) -> usize {
    poly.coefficients()
        .iter()
        .rposition(|coefficient| *coefficient != F::zero())
        .unwrap_or(0)
}

fn append_compressed_univariate_poly<F, T>(
    transcript: &mut T,
    label: &'static str,
    poly: &UnivariatePoly<F>,
) where
    F: Field,
    T: Transcript<Challenge = F>,
{
    let compressed = poly.compress();
    transcript.append(&LabelWithCount(
        label.as_bytes(),
        compressed.coeffs_except_linear_term().len() as u64,
    ));
    for coefficient in compressed.coeffs_except_linear_term() {
        transcript.append(coefficient);
    }
}

fn append_labeled_scalar<F, T>(transcript: &mut T, label: &'static str, scalar: &F)
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    transcript.append(&Label(label.as_bytes()));
    transcript.append(scalar);
}

fn append_opening_claims<F, T>(
    program: &'static Stage7CpuProgramPlan,
    store: &mut Stage7ValueStore<F>,
    transcript: &mut T,
    evals: &[Stage7NamedEval<F>],
) -> Result<Vec<Stage7OpeningClaimValue<F>>, Stage7KernelError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    if program.opening_batches.is_empty() {
        for eval in evals {
            append_labeled_scalar(transcript, "opening_claim", &eval.value);
        }
        return Ok(Vec::new());
    }
    let _ = store.evaluate_available_points(program)?;
    let mut opening_claims = Vec::new();
    for batch in program.opening_batches {
        for symbol in batch.claim_operands {
            let claim =
                find_opening_claim(program, symbol).ok_or(Stage7KernelError::MissingClaim {
                    batch: batch.symbol,
                    claim: symbol,
                })?;
            let point = store.point(claim.point_source)?.to_vec();
            let value = store.scalar(claim.eval_source)?;
            append_labeled_scalar(transcript, "opening_claim", &value);
            opening_claims.push(Stage7OpeningClaimValue {
                symbol: claim.symbol,
                oracle: claim.oracle,
                domain: claim.domain,
                claim_kind: claim.claim_kind,
                point,
                eval: value,
            });
        }
    }
    Ok(opening_claims)
}

pub fn execute_stage7_program<F, T, E>(
    program: &'static Stage7CpuProgramPlan,
    mode: Stage7ExecutionMode,
    executor: &mut E,
    transcript: &mut T,
) -> Result<Stage7ExecutionArtifacts<F>, Stage7KernelError>
where
    F: Field,
    T: Transcript<Challenge = F>,
    E: Stage7KernelExecutor<F>,
{
    let mut artifacts = Stage7ExecutionArtifacts::default();
    for step in program.steps {
        match step.kind {
            "transcript_squeeze" => {
                let squeeze =
                    find_squeeze(program, step.symbol).ok_or(Stage7KernelError::MissingValue {
                        symbol: step.symbol,
                    })?;
                let values = transcript.challenge_vector(squeeze.count);
                executor.observe_challenge_vector(squeeze, &values)?;
                artifacts.challenge_vectors.push(Stage7ChallengeVector {
                    symbol: squeeze.symbol,
                    values,
                });
            }
            "transcript_absorb_bytes" => {
                let absorb = find_absorb_bytes(program, step.symbol).ok_or(
                    Stage7KernelError::MissingValue {
                        symbol: step.symbol,
                    },
                )?;
                absorb_stage7_bytes(absorb, transcript);
            }
            "sumcheck_driver" => {
                let driver =
                    find_driver(program, step.symbol).ok_or(Stage7KernelError::MissingDriver {
                        driver: step.symbol,
                    })?;
                let kernel_symbol = driver.kernel.ok_or(Stage7KernelError::MissingKernel {
                    driver: driver.symbol,
                    kernel: "<missing>",
                })?;
                let kernel = find_kernel(program, kernel_symbol).ok_or(
                    Stage7KernelError::MissingKernel {
                        driver: driver.symbol,
                        kernel: kernel_symbol,
                    },
                )?;
                let batch =
                    find_batch(program, driver.batch).ok_or(Stage7KernelError::MissingBatch {
                        driver: driver.symbol,
                        batch: driver.batch,
                    })?;
                let context = Stage7KernelContext {
                    mode,
                    program,
                    kernel,
                    batch,
                    driver,
                };
                let output = match mode {
                    Stage7ExecutionMode::Prover => executor.prove_sumcheck(context, transcript)?,
                    Stage7ExecutionMode::Verifier => {
                        executor.verify_sumcheck(context, transcript)?
                    }
                };
                executor.observe_sumcheck_output(&output)?;
                artifacts
                    .opening_claims
                    .extend(output.opening_claims.clone());
                artifacts.sumchecks.push(output);
            }
            _ => {
                return Err(Stage7KernelError::InvalidProgramStep {
                    symbol: step.symbol,
                    kind: step.kind,
                });
            }
        }
    }
    artifacts
        .opening_batches
        .extend(program.opening_batches.iter());
    Ok(artifacts)
}

fn absorb_stage7_bytes<T>(absorb: &'static Stage7TranscriptAbsorbBytesPlan, transcript: &mut T)
where
    T: Transcript,
{
    transcript.append(&LabelWithCount(
        absorb.label.as_bytes(),
        absorb.payload.len() as u64,
    ));
    transcript.append_bytes(absorb.payload.as_bytes());
}

fn find_squeeze(
    program: &'static Stage7CpuProgramPlan,
    symbol: &str,
) -> Option<&'static Stage7TranscriptSqueezePlan> {
    program
        .transcript_squeezes
        .iter()
        .find(|squeeze| squeeze.symbol == symbol)
}

fn find_absorb_bytes(
    program: &'static Stage7CpuProgramPlan,
    symbol: &str,
) -> Option<&'static Stage7TranscriptAbsorbBytesPlan> {
    program
        .transcript_absorb_bytes
        .iter()
        .find(|absorb| absorb.symbol == symbol)
}

fn find_driver(
    program: &'static Stage7CpuProgramPlan,
    symbol: &str,
) -> Option<&'static Stage7SumcheckDriverPlan> {
    program
        .drivers
        .iter()
        .find(|driver| driver.symbol == symbol)
}

fn find_kernel(
    program: &'static Stage7CpuProgramPlan,
    symbol: &str,
) -> Option<&'static Stage7KernelPlan> {
    program
        .kernels
        .iter()
        .find(|kernel| kernel.symbol == symbol)
}

fn find_batch(
    program: &'static Stage7CpuProgramPlan,
    symbol: &str,
) -> Option<&'static Stage7SumcheckBatchPlan> {
    program.batches.iter().find(|batch| batch.symbol == symbol)
}

fn find_opening_claim(
    program: &'static Stage7CpuProgramPlan,
    symbol: &str,
) -> Option<&'static Stage7OpeningClaimPlan> {
    program
        .opening_claims
        .iter()
        .find(|claim| claim.symbol == symbol)
}

fn reverse_slice<F: Field>(slice: &[F]) -> Vec<F> {
    slice.iter().rev().copied().collect()
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests use expect for assertion context")]
mod tests {
    use super::*;
    use jolt_field::Fr;
    use jolt_transcript::Blake2bTranscript;

    const PARAMS: Stage7Params = Stage7Params {
        field: "bn254_fr",
        pcs: "dory",
        transcript: "blake2b_transcript",
    };
    const STEPS: &[Stage7ProgramStepPlan] = &[Stage7ProgramStepPlan {
        kind: "sumcheck_driver",
        symbol: "stage7.sumcheck",
    }];
    const FIELD_CONSTANTS: &[Stage7FieldConstantPlan] = &[
        Stage7FieldConstantPlan {
            symbol: "stage7.field.one",
            field: "bn254_fr",
            value: 1,
        },
        Stage7FieldConstantPlan {
            symbol: "stage7.hamming_weight_claim_reduction.gamma",
            field: "bn254_fr",
            value: 2,
        },
    ];
    const CLAIM_INPUTS: &[&str] = &["stage7.input.claim"];
    const CLAIMS: &[Stage7SumcheckClaimPlan] = &[Stage7SumcheckClaimPlan {
        symbol: "stage7.hamming_weight_claim_reduction.input",
        stage: "stage7",
        domain: "jolt.stage7_hamming_weight_claim_reduction_domain",
        num_rounds: 1,
        degree: 2,
        claim: "stage7.hamming_weight_claim_reduction.weighted_stage6_claims",
        kernel: Some("jolt.cpu.stage7.hamming_weight_claim_reduction"),
        relation: None,
        claim_value: "stage7.input.claim",
        input_openings: CLAIM_INPUTS,
    }];
    const KERNELS: &[Stage7KernelPlan] = &[
        Stage7KernelPlan {
            symbol: "jolt.cpu.stage7.hamming_weight_claim_reduction",
            relation: "jolt.stage7.hamming_weight_claim_reduction",
            kind: "sumcheck",
            backend: "cpu",
            abi: "jolt_stage7_hamming_weight_claim_reduction",
        },
        Stage7KernelPlan {
            symbol: "jolt.cpu.stage7.batched",
            relation: "jolt.stage7.batched",
            kind: "sumcheck",
            backend: "cpu",
            abi: "jolt_stage7_batched",
        },
    ];
    const BATCHES: &[Stage7SumcheckBatchPlan] = &[Stage7SumcheckBatchPlan {
        symbol: "stage7.batch",
        stage: "stage7",
        proof_slot: "stage7.sumcheck",
        policy: "jolt_core_stage7_aligned",
        count: 1,
        ordered_claims: &["stage7.hamming_weight_claim_reduction.input"],
        claim_operands: &["stage7.hamming_weight_claim_reduction.input"],
        claim_label: "sumcheck_claim",
        round_label: "sumcheck_poly",
        round_schedule: &[1],
    }];
    const DRIVERS: &[Stage7SumcheckDriverPlan] = &[Stage7SumcheckDriverPlan {
        symbol: "stage7.sumcheck",
        stage: "stage7",
        proof_slot: "stage7.sumcheck",
        kernel: Some("jolt.cpu.stage7.batched"),
        relation: Some("jolt.stage7.batched"),
        batch: "stage7.batch",
        policy: "jolt_core_stage7_aligned",
        round_schedule: &[1],
        claim_label: "sumcheck_claim",
        round_label: "sumcheck_poly",
        num_rounds: 1,
        degree: 2,
    }];
    const INSTANCE_RESULTS: &[Stage7SumcheckInstanceResultPlan] =
        &[Stage7SumcheckInstanceResultPlan {
            symbol: "stage7.hamming_weight_claim_reduction.instance",
            source: "stage7.sumcheck",
            claim: "stage7.hamming_weight_claim_reduction.input",
            relation: "jolt.stage7.hamming_weight_claim_reduction",
            index: 0,
            point_arity: 1,
            num_rounds: 1,
            round_offset: 0,
            point_order: "reverse",
            degree: 2,
        }];
    const EVALS: &[Stage7SumcheckEvalPlan] = &[
        Stage7SumcheckEvalPlan {
            symbol: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_0",
            source: "stage7.sumcheck",
            name: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_0",
            index: 0,
            oracle: "InstructionRa_0",
        },
        Stage7SumcheckEvalPlan {
            symbol: "stage7.hamming_weight_claim_reduction.eval.RamRa_0",
            source: "stage7.sumcheck",
            name: "stage7.hamming_weight_claim_reduction.eval.RamRa_0",
            index: 1,
            oracle: "RamRa_0",
        },
    ];
    const OPENING_CLAIMS: &[Stage7OpeningClaimPlan] = &[
        Stage7OpeningClaimPlan {
            symbol: "stage7.hamming_weight_claim_reduction.opening.InstructionRa_0",
            oracle: "InstructionRa_0",
            domain: "jolt.main_witness_commit_domain",
            point_arity: 2,
            claim_kind: "committed",
            point_source: "stage7.hamming_weight_claim_reduction.point",
            eval_source: "stage7.hamming_weight_claim_reduction.eval.InstructionRa_0",
        },
        Stage7OpeningClaimPlan {
            symbol: "stage7.hamming_weight_claim_reduction.opening.RamRa_0",
            oracle: "RamRa_0",
            domain: "jolt.main_witness_commit_domain",
            point_arity: 2,
            claim_kind: "committed",
            point_source: "stage7.hamming_weight_claim_reduction.point",
            eval_source: "stage7.hamming_weight_claim_reduction.eval.RamRa_0",
        },
    ];
    const OPENING_BATCHES: &[Stage7OpeningBatchPlan] = &[Stage7OpeningBatchPlan {
        symbol: "stage7.openings",
        stage: "stage7",
        proof_slot: "stage7.openings",
        policy: "jolt_stage7_output_order",
        count: 2,
        ordered_claims: &[
            "stage7.hamming_weight_claim_reduction.opening.InstructionRa_0",
            "stage7.hamming_weight_claim_reduction.opening.RamRa_0",
        ],
        claim_operands: &[
            "stage7.hamming_weight_claim_reduction.opening.InstructionRa_0",
            "stage7.hamming_weight_claim_reduction.opening.RamRa_0",
        ],
    }];
    const POINT_SLICES: &[Stage7PointSlicePlan] = &[Stage7PointSlicePlan {
        symbol: "stage7.hamming_weight_claim_reduction.point.cycle",
        source: "stage7.input.stage6.booleanity.InstructionRa_0",
        offset: 1,
        length: 1,
        input: "stage7.input.stage6.booleanity.InstructionRa_0",
    }];
    const POINT_CONCAT_INPUTS: &[&str] = &[
        "stage7.hamming_weight_claim_reduction.instance",
        "stage7.hamming_weight_claim_reduction.point.cycle",
    ];
    const POINT_CONCATS: &[Stage7PointConcatPlan] = &[Stage7PointConcatPlan {
        symbol: "stage7.hamming_weight_claim_reduction.point",
        layout: "address_chunk_then_cycle",
        arity: 2,
        inputs: POINT_CONCAT_INPUTS,
    }];
    const PROGRAM: Stage7CpuProgramPlan = Stage7CpuProgramPlan {
        role: "prover",
        params: PARAMS,
        steps: STEPS,
        transcript_squeezes: &[],
        transcript_absorb_bytes: &[],
        opening_inputs: &[],
        field_constants: FIELD_CONSTANTS,
        field_exprs: &[],
        kernels: KERNELS,
        claims: CLAIMS,
        batches: BATCHES,
        drivers: DRIVERS,
        instance_results: INSTANCE_RESULTS,
        evals: EVALS,
        point_zeros: &[],
        point_slices: POINT_SLICES,
        point_concats: POINT_CONCATS,
        opening_claims: OPENING_CLAIMS,
        opening_equalities: &[],
        opening_batches: OPENING_BATCHES,
    };

    #[test]
    fn hamming_weight_claim_reduction_prover_and_replay_agree() {
        let r_bool = Fr::from_u64(5);
        let r_cycle = Fr::from_u64(3);
        let r_instr_virt = Fr::from_u64(7);
        let r_ram_virt = Fr::from_u64(11);
        let instruction_ra = vec![
            Fr::from_u64(1),
            Fr::from_u64(0),
            Fr::from_u64(0),
            Fr::from_u64(1),
        ];
        let ram_ra = vec![
            Fr::from_u64(0),
            Fr::from_u64(0),
            Fr::from_u64(1),
            Fr::from_u64(0),
        ];
        let bool_point = vec![r_bool, r_cycle];
        let instr_virt_point = vec![r_instr_virt, r_cycle];
        let ram_virt_point = vec![r_ram_virt, r_cycle];
        let instr_bool = eval_full_cycle_major(&instruction_ra, &bool_point);
        let ram_bool = eval_full_cycle_major(&ram_ra, &bool_point);
        let instr_virt = eval_full_cycle_major(&instruction_ra, &instr_virt_point);
        let ram_virt = eval_full_cycle_major(&ram_ra, &ram_virt_point);
        let ram_hw = r_cycle;
        let gamma = Fr::from_u64(2);
        let gamma_powers = gamma_powers(gamma, 6);
        let input_claim = gamma_powers[0] * Fr::from_u64(1)
            + gamma_powers[1] * instr_bool
            + gamma_powers[2] * instr_virt
            + gamma_powers[3] * ram_hw
            + gamma_powers[4] * ram_bool
            + gamma_powers[5] * ram_virt;
        let openings = vec![
            Stage7OpeningInputValue {
                symbol: "stage7.input.claim",
                point: Vec::new(),
                eval: input_claim,
            },
            Stage7OpeningInputValue {
                symbol: "stage7.input.stage6.hamming_booleanity.HammingWeight",
                point: vec![r_cycle],
                eval: ram_hw,
            },
            Stage7OpeningInputValue {
                symbol: "stage7.input.stage6.booleanity.InstructionRa_0",
                point: bool_point.clone(),
                eval: instr_bool,
            },
            Stage7OpeningInputValue {
                symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_0",
                point: instr_virt_point,
                eval: instr_virt,
            },
            Stage7OpeningInputValue {
                symbol: "stage7.input.stage6.booleanity.RamRa_0",
                point: bool_point,
                eval: ram_bool,
            },
            Stage7OpeningInputValue {
                symbol: "stage7.input.stage6.ram_ra_virtual.RamRa_0",
                point: ram_virt_point,
                eval: ram_virt,
            },
        ];
        let instruction_chunks: Vec<&[Fr]> = vec![&instruction_ra];
        let ram_chunks: Vec<&[Fr]> = vec![&ram_ra];
        let inputs = Stage7ProverInputs::new(&openings).with_hamming_weight_claim_reduction(
            Stage7HammingWeightClaimReductionWitness {
                instruction_ra: Stage7RaChunks {
                    chunks: &instruction_chunks,
                    layout: Stage7RaChunkLayout::CycleMajor,
                },
                bytecode_ra: Stage7RaChunks {
                    chunks: &[],
                    layout: Stage7RaChunkLayout::CycleMajor,
                },
                ram_ra: Stage7RaChunks {
                    chunks: &ram_chunks,
                    layout: Stage7RaChunkLayout::CycleMajor,
                },
            },
        );
        let mut prover = Stage7ProverKernelExecutor::new(inputs);
        let mut prover_transcript = Blake2bTranscript::<Fr>::new(b"stage7-test");
        let artifacts = execute_stage7_program(
            &PROGRAM,
            Stage7ExecutionMode::Prover,
            &mut prover,
            &mut prover_transcript,
        )
        .expect("stage7 prover succeeds");
        assert_eq!(artifacts.sumchecks.len(), 1);
        assert_eq!(artifacts.sumchecks[0].evals.len(), 2);

        let proof = Stage7Proof {
            sumchecks: artifacts.sumchecks.clone(),
        };
        let mut verifier = Stage7ProofCarryingKernelExecutor::new(&proof, &openings);
        let mut verifier_transcript = Blake2bTranscript::<Fr>::new(b"stage7-test");
        let verified = execute_stage7_program(
            &PROGRAM,
            Stage7ExecutionMode::Verifier,
            &mut verifier,
            &mut verifier_transcript,
        )
        .expect("stage7 replay succeeds");
        assert_eq!(verified.sumchecks[0].point, artifacts.sumchecks[0].point);
        assert_eq!(verifier_transcript.state(), prover_transcript.state());
    }

    #[test]
    fn hamming_weight_claim_reduction_index_witness_matches_dense_witness() {
        let r_bool = Fr::from_u64(5);
        let r_cycle = Fr::from_u64(3);
        let r_instr_virt = Fr::from_u64(7);
        let r_ram_virt = Fr::from_u64(11);
        let instruction_ra = vec![
            Fr::from_u64(1),
            Fr::from_u64(0),
            Fr::from_u64(0),
            Fr::from_u64(1),
        ];
        let ram_ra = vec![
            Fr::from_u64(0),
            Fr::from_u64(0),
            Fr::from_u64(1),
            Fr::from_u64(0),
        ];
        let bool_point = vec![r_bool, r_cycle];
        let instr_virt_point = vec![r_instr_virt, r_cycle];
        let ram_virt_point = vec![r_ram_virt, r_cycle];
        let instr_bool = eval_full_cycle_major(&instruction_ra, &bool_point);
        let ram_bool = eval_full_cycle_major(&ram_ra, &bool_point);
        let instr_virt = eval_full_cycle_major(&instruction_ra, &instr_virt_point);
        let ram_virt = eval_full_cycle_major(&ram_ra, &ram_virt_point);
        let ram_hw = r_cycle;
        let gamma = Fr::from_u64(2);
        let gamma_powers = gamma_powers(gamma, 6);
        let input_claim = gamma_powers[0] * Fr::from_u64(1)
            + gamma_powers[1] * instr_bool
            + gamma_powers[2] * instr_virt
            + gamma_powers[3] * ram_hw
            + gamma_powers[4] * ram_bool
            + gamma_powers[5] * ram_virt;
        let openings = vec![
            Stage7OpeningInputValue {
                symbol: "stage7.input.claim",
                point: Vec::new(),
                eval: input_claim,
            },
            Stage7OpeningInputValue {
                symbol: "stage7.input.stage6.hamming_booleanity.HammingWeight",
                point: vec![r_cycle],
                eval: ram_hw,
            },
            Stage7OpeningInputValue {
                symbol: "stage7.input.stage6.booleanity.InstructionRa_0",
                point: bool_point.clone(),
                eval: instr_bool,
            },
            Stage7OpeningInputValue {
                symbol: "stage7.input.stage6.instruction_ra_virtual.InstructionRa_0",
                point: instr_virt_point,
                eval: instr_virt,
            },
            Stage7OpeningInputValue {
                symbol: "stage7.input.stage6.booleanity.RamRa_0",
                point: bool_point,
                eval: ram_bool,
            },
            Stage7OpeningInputValue {
                symbol: "stage7.input.stage6.ram_ra_virtual.RamRa_0",
                point: ram_virt_point,
                eval: ram_virt,
            },
        ];
        let instruction_indices = vec![Some(0), Some(1)];
        let ram_indices = vec![None, Some(0)];
        let instruction_chunks: Vec<&[Option<u8>]> = vec![&instruction_indices];
        let ram_chunks: Vec<&[Option<u8>]> = vec![&ram_indices];
        let inputs = Stage7ProverInputs::new(&openings)
            .with_hamming_weight_claim_reduction_indices(
                Stage7HammingWeightClaimReductionIndexWitness {
                    instruction_ra: Stage7RaIndexChunks {
                        chunks: &instruction_chunks,
                    },
                    bytecode_ra: Stage7RaIndexChunks { chunks: &[] },
                    ram_ra: Stage7RaIndexChunks {
                        chunks: &ram_chunks,
                    },
                },
            );
        let mut prover = Stage7ProverKernelExecutor::new(inputs);
        let mut prover_transcript = Blake2bTranscript::<Fr>::new(b"stage7-test");
        let artifacts = execute_stage7_program(
            &PROGRAM,
            Stage7ExecutionMode::Prover,
            &mut prover,
            &mut prover_transcript,
        )
        .expect("stage7 index prover succeeds");
        assert_eq!(artifacts.sumchecks.len(), 1);

        let proof = Stage7Proof {
            sumchecks: artifacts.sumchecks.clone(),
        };
        let mut verifier = Stage7ProofCarryingKernelExecutor::new(&proof, &openings);
        let mut verifier_transcript = Blake2bTranscript::<Fr>::new(b"stage7-test");
        let verified = execute_stage7_program(
            &PROGRAM,
            Stage7ExecutionMode::Verifier,
            &mut verifier,
            &mut verifier_transcript,
        )
        .expect("stage7 index replay succeeds");
        assert_eq!(verified.sumchecks[0].point, artifacts.sumchecks[0].point);
        assert_eq!(verifier_transcript.state(), prover_transcript.state());
    }

    fn eval_full_cycle_major(evals: &[Fr], point: &[Fr]) -> Fr {
        let address = EqPolynomial::<Fr>::evals(&point[..1], None);
        let cycle = EqPolynomial::<Fr>::evals(&point[1..], None);
        let mut value = Fr::from_u64(0);
        for cycle_index in 0..2 {
            for address_index in 0..2 {
                value += evals[cycle_index * 2 + address_index]
                    * cycle[cycle_index]
                    * address[address_index];
            }
        }
        value
    }
}
