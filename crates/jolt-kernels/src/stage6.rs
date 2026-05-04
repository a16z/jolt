//! Stage 6 coarse-kernel ABI used by Bolt-generated Jolt prover code.

use std::error::Error;
use std::fmt::{self, Display, Formatter};

use crate::dense::{bind_dense_evals_reuse, DENSE_BIND_PAR_THRESHOLD};
use jolt_field::Field;
use jolt_poly::{EqPolynomial, UnivariatePoly};
use jolt_transcript::{Label, LabelWithCount, Transcript};
use rayon::prelude::*;

pub use crate::stage4::{
    Stage4ChallengeVector as Stage6ChallengeVector,
    Stage4ExecutionArtifacts as Stage6ExecutionArtifacts,
    Stage4ExecutionMode as Stage6ExecutionMode, Stage4FieldConstantPlan as Stage6FieldConstantPlan,
    Stage4FieldExprPlan as Stage6FieldExprPlan, Stage4NamedEval as Stage6NamedEval,
    Stage4OpeningBatchPlan as Stage6OpeningBatchPlan,
    Stage4OpeningClaimEqualityPlan as Stage6OpeningClaimEqualityPlan,
    Stage4OpeningClaimPlan as Stage6OpeningClaimPlan,
    Stage4OpeningInputPlan as Stage6OpeningInputPlan,
    Stage4OpeningInputValue as Stage6OpeningInputValue, Stage4Params as Stage6Params,
    Stage4PointConcatPlan as Stage6PointConcatPlan, Stage4PointSlicePlan as Stage6PointSlicePlan,
    Stage4ProgramStepPlan as Stage6ProgramStepPlan, Stage4Proof as Stage6Proof,
    Stage4SumcheckBatchPlan as Stage6SumcheckBatchPlan,
    Stage4SumcheckClaimPlan as Stage6SumcheckClaimPlan,
    Stage4SumcheckDriverPlan as Stage6SumcheckDriverPlan,
    Stage4SumcheckEvalPlan as Stage6SumcheckEvalPlan,
    Stage4SumcheckInstanceResultPlan as Stage6SumcheckInstanceResultPlan,
    Stage4SumcheckOutput as Stage6SumcheckOutput,
    Stage4TranscriptAbsorbBytesPlan as Stage6TranscriptAbsorbBytesPlan,
    Stage4TranscriptSqueezePlan as Stage6TranscriptSqueezePlan,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage6KernelPlan {
    pub symbol: &'static str,
    pub relation: &'static str,
    pub kind: &'static str,
    pub backend: &'static str,
    pub abi: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage6PointZeroPlan {
    pub symbol: &'static str,
    pub field: &'static str,
    pub arity: usize,
}

impl Stage6KernelPlan {
    pub fn relation_kind(&self) -> Result<Stage6Relation, Stage6KernelError> {
        Stage6Relation::from_symbol(self.relation).ok_or(Stage6KernelError::UnknownRelation {
            relation: self.relation,
        })
    }

    pub fn abi_kind(&self) -> Result<Stage6KernelAbi, Stage6KernelError> {
        Stage6KernelAbi::from_name(self.abi)
            .ok_or(Stage6KernelError::UnknownKernelAbi { abi: self.abi })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage6CpuProgramPlan {
    pub role: &'static str,
    pub params: Stage6Params,
    pub steps: &'static [Stage6ProgramStepPlan],
    pub transcript_squeezes: &'static [Stage6TranscriptSqueezePlan],
    pub transcript_absorb_bytes: &'static [Stage6TranscriptAbsorbBytesPlan],
    pub opening_inputs: &'static [Stage6OpeningInputPlan],
    pub field_constants: &'static [Stage6FieldConstantPlan],
    pub field_exprs: &'static [Stage6FieldExprPlan],
    pub kernels: &'static [Stage6KernelPlan],
    pub claims: &'static [Stage6SumcheckClaimPlan],
    pub batches: &'static [Stage6SumcheckBatchPlan],
    pub drivers: &'static [Stage6SumcheckDriverPlan],
    pub instance_results: &'static [Stage6SumcheckInstanceResultPlan],
    pub evals: &'static [Stage6SumcheckEvalPlan],
    pub point_zeros: &'static [Stage6PointZeroPlan],
    pub point_slices: &'static [Stage6PointSlicePlan],
    pub point_concats: &'static [Stage6PointConcatPlan],
    pub opening_claims: &'static [Stage6OpeningClaimPlan],
    pub opening_equalities: &'static [Stage6OpeningClaimEqualityPlan],
    pub opening_batches: &'static [Stage6OpeningBatchPlan],
}

impl Stage6CpuProgramPlan {
    pub fn claim(&self, symbol: &str) -> Option<&Stage6SumcheckClaimPlan> {
        self.claims.iter().find(|claim| claim.symbol == symbol)
    }

    pub fn instance_results_for_driver(
        &self,
        driver: &'static str,
    ) -> impl Iterator<Item = &Stage6SumcheckInstanceResultPlan> {
        self.instance_results
            .iter()
            .filter(move |instance| instance.source == driver)
    }

    pub fn evals_for_driver(
        &self,
        driver: &'static str,
    ) -> impl Iterator<Item = &Stage6SumcheckEvalPlan> {
        self.evals.iter().filter(move |eval| eval.source == driver)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage6Relation {
    BytecodeReadRaf,
    Booleanity,
    HammingBooleanity,
    RamRaVirtual,
    InstructionRaVirtual,
    IncClaimReduction,
    Batched,
}

impl Stage6Relation {
    pub fn from_symbol(symbol: &str) -> Option<Self> {
        match symbol {
            "jolt.stage6.bytecode_read_raf" => Some(Self::BytecodeReadRaf),
            "jolt.stage6.booleanity" => Some(Self::Booleanity),
            "jolt.stage6.hamming_booleanity" => Some(Self::HammingBooleanity),
            "jolt.stage6.ram_ra_virtual" => Some(Self::RamRaVirtual),
            "jolt.stage6.instruction_ra_virtual" => Some(Self::InstructionRaVirtual),
            "jolt.stage6.inc_claim_reduction" => Some(Self::IncClaimReduction),
            "jolt.stage6.batched" => Some(Self::Batched),
            _ => None,
        }
    }

    pub fn symbol(self) -> &'static str {
        match self {
            Self::BytecodeReadRaf => "jolt.stage6.bytecode_read_raf",
            Self::Booleanity => "jolt.stage6.booleanity",
            Self::HammingBooleanity => "jolt.stage6.hamming_booleanity",
            Self::RamRaVirtual => "jolt.stage6.ram_ra_virtual",
            Self::InstructionRaVirtual => "jolt.stage6.instruction_ra_virtual",
            Self::IncClaimReduction => "jolt.stage6.inc_claim_reduction",
            Self::Batched => "jolt.stage6.batched",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage6KernelAbi {
    BytecodeReadRaf,
    Booleanity,
    HammingBooleanity,
    RamRaVirtual,
    InstructionRaVirtual,
    IncClaimReduction,
    Batched,
}

impl Stage6KernelAbi {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "jolt_stage6_bytecode_read_raf" => Some(Self::BytecodeReadRaf),
            "jolt_stage6_booleanity" => Some(Self::Booleanity),
            "jolt_stage6_hamming_booleanity" => Some(Self::HammingBooleanity),
            "jolt_stage6_ram_ra_virtual" => Some(Self::RamRaVirtual),
            "jolt_stage6_instruction_ra_virtual" => Some(Self::InstructionRaVirtual),
            "jolt_stage6_inc_claim_reduction" => Some(Self::IncClaimReduction),
            "jolt_stage6_batched" => Some(Self::Batched),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::BytecodeReadRaf => "jolt_stage6_bytecode_read_raf",
            Self::Booleanity => "jolt_stage6_booleanity",
            Self::HammingBooleanity => "jolt_stage6_hamming_booleanity",
            Self::RamRaVirtual => "jolt_stage6_ram_ra_virtual",
            Self::InstructionRaVirtual => "jolt_stage6_instruction_ra_virtual",
            Self::IncClaimReduction => "jolt_stage6_inc_claim_reduction",
            Self::Batched => "jolt_stage6_batched",
        }
    }
}

const BYTECODE_READ_RAF_STAGE_COUNT: usize = 5;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage6KernelError {
    MissingClaim {
        batch: &'static str,
        claim: &'static str,
    },
    MissingValue {
        symbol: &'static str,
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
        expected: Stage6ExecutionMode,
        actual: Stage6ExecutionMode,
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

impl Display for Stage6KernelError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingClaim { batch, claim } => {
                write!(
                    formatter,
                    "stage6 batch @{batch} references missing claim @{claim}"
                )
            }
            Self::MissingValue { symbol } => {
                write!(formatter, "stage6 value @{symbol} is not available")
            }
            Self::MissingDriver { driver } => {
                write!(formatter, "stage6 driver @{driver} is not available")
            }
            Self::MissingKernel { driver, kernel } => {
                write!(
                    formatter,
                    "stage6 driver @{driver} references missing kernel @{kernel}"
                )
            }
            Self::MissingBatch { driver, batch } => {
                write!(
                    formatter,
                    "stage6 driver @{driver} references missing batch @{batch}"
                )
            }
            Self::UnknownRelation { relation } => {
                write!(formatter, "unknown stage6 relation `{relation}`")
            }
            Self::UnknownKernelAbi { abi } => {
                write!(formatter, "unknown stage6 kernel ABI `{abi}`")
            }
            Self::PlanCountMismatch {
                artifact,
                expected,
                actual,
            } => {
                write!(
                    formatter,
                    "stage6 {artifact} plan count mismatch: expected {expected}, got {actual}"
                )
            }
            Self::InvalidInputLength {
                input,
                expected,
                actual,
            } => {
                write!(
                    formatter,
                    "stage6 input `{input}` has length {actual}, expected {expected}"
                )
            }
            Self::UnsupportedFieldExpr { symbol, formula } => {
                write!(
                    formatter,
                    "stage6 field expr @{symbol} uses unsupported formula `{formula}`"
                )
            }
            Self::KernelNotImplemented { abi } => {
                write!(formatter, "stage6 kernel ABI `{abi}` is not implemented")
            }
            Self::WrongExecutorMode {
                driver,
                expected,
                actual,
            } => {
                write!(
                    formatter,
                    "stage6 driver @{driver} expected {expected:?} executor, got {actual:?}"
                )
            }
            Self::MissingProof { driver } => {
                write!(formatter, "stage6 proof for driver @{driver} is missing")
            }
            Self::MissingKernelInput { kernel, input } => {
                write!(
                    formatter,
                    "stage6 kernel `{kernel}` is missing required input `{input}`"
                )
            }
            Self::InvalidProgramStep { symbol, kind } => {
                write!(
                    formatter,
                    "stage6 program step @{symbol} has invalid kind `{kind}`"
                )
            }
            Self::InvalidProof { driver, reason } => {
                write!(
                    formatter,
                    "stage6 proof for driver @{driver} is invalid: {reason}"
                )
            }
        }
    }
}

impl Error for Stage6KernelError {}

#[derive(Clone, Copy)]
pub struct Stage6BooleanityWitness<'a, F: Field> {
    pub chunks: &'a [&'a [F]],
}

#[derive(Clone, Copy, Debug)]
pub struct Stage6BytecodeEntry<F: Field> {
    pub address: F,
    pub imm: F,
    pub circuit_flags: [bool; 14],
    pub rd: Option<usize>,
    pub rs1: Option<usize>,
    pub rs2: Option<usize>,
    pub lookup_table: Option<usize>,
    pub is_interleaved: bool,
    pub is_branch: bool,
    pub left_is_rs1: bool,
    pub left_is_pc: bool,
    pub right_is_rs2: bool,
    pub right_is_imm: bool,
    pub is_noop: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct Stage6BytecodeReadRafData<'a, F: Field> {
    pub entries: &'a [Stage6BytecodeEntry<F>],
    pub entry_bytecode_index: usize,
    pub num_lookup_tables: usize,
}

#[derive(Clone, Copy)]
pub struct Stage6BytecodeReadRafWitness<'a, F: Field> {
    pub data: Stage6BytecodeReadRafData<'a, F>,
    pub bytecode_ra_chunks: &'a [&'a [F]],
}

#[derive(Clone, Copy)]
pub struct Stage6HammingBooleanityWitness<'a, F: Field> {
    pub hamming_weight: &'a [F],
}

#[derive(Clone, Copy)]
pub struct Stage6IncClaimReductionWitness<'a, F: Field> {
    pub ram_inc: &'a [F],
    pub rd_inc: &'a [F],
}

#[derive(Clone, Copy)]
pub struct Stage6RamRaVirtualWitness<'a, F: Field> {
    pub ram_ra_chunks: &'a [&'a [F]],
}

#[derive(Clone, Copy)]
pub struct Stage6InstructionRaVirtualWitness<'a, F: Field> {
    pub instruction_ra_chunks: &'a [&'a [F]],
    pub virtual_count: usize,
}

#[derive(Clone, Copy)]
pub struct Stage6ProverInputs<'a, F: Field> {
    pub opening_inputs: &'a [Stage6OpeningInputValue<F>],
    pub bytecode_read_raf: Option<Stage6BytecodeReadRafWitness<'a, F>>,
    pub booleanity: Option<Stage6BooleanityWitness<'a, F>>,
    pub hamming_booleanity: Option<Stage6HammingBooleanityWitness<'a, F>>,
    pub inc_claim_reduction: Option<Stage6IncClaimReductionWitness<'a, F>>,
    pub ram_ra_virtual: Option<Stage6RamRaVirtualWitness<'a, F>>,
    pub instruction_ra_virtual: Option<Stage6InstructionRaVirtualWitness<'a, F>>,
}

impl<'a, F: Field> Stage6ProverInputs<'a, F> {
    pub fn new(opening_inputs: &'a [Stage6OpeningInputValue<F>]) -> Self {
        Self {
            opening_inputs,
            bytecode_read_raf: None,
            booleanity: None,
            hamming_booleanity: None,
            inc_claim_reduction: None,
            ram_ra_virtual: None,
            instruction_ra_virtual: None,
        }
    }

    pub fn empty() -> Self {
        Self {
            opening_inputs: &[],
            bytecode_read_raf: None,
            booleanity: None,
            hamming_booleanity: None,
            inc_claim_reduction: None,
            ram_ra_virtual: None,
            instruction_ra_virtual: None,
        }
    }

    pub fn with_hamming_booleanity(
        mut self,
        hamming_booleanity: Stage6HammingBooleanityWitness<'a, F>,
    ) -> Self {
        self.hamming_booleanity = Some(hamming_booleanity);
        self
    }

    pub fn with_booleanity(mut self, booleanity: Stage6BooleanityWitness<'a, F>) -> Self {
        self.booleanity = Some(booleanity);
        self
    }

    pub fn with_bytecode_read_raf(
        mut self,
        bytecode_read_raf: Stage6BytecodeReadRafWitness<'a, F>,
    ) -> Self {
        self.bytecode_read_raf = Some(bytecode_read_raf);
        self
    }

    pub fn with_inc_claim_reduction(
        mut self,
        inc_claim_reduction: Stage6IncClaimReductionWitness<'a, F>,
    ) -> Self {
        self.inc_claim_reduction = Some(inc_claim_reduction);
        self
    }

    pub fn with_ram_ra_virtual(mut self, ram_ra_virtual: Stage6RamRaVirtualWitness<'a, F>) -> Self {
        self.ram_ra_virtual = Some(ram_ra_virtual);
        self
    }

    pub fn with_instruction_ra_virtual(
        mut self,
        instruction_ra_virtual: Stage6InstructionRaVirtualWitness<'a, F>,
    ) -> Self {
        self.instruction_ra_virtual = Some(instruction_ra_virtual);
        self
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Stage6KernelContext<'a> {
    pub mode: Stage6ExecutionMode,
    pub program: &'static Stage6CpuProgramPlan,
    pub kernel: &'a Stage6KernelPlan,
    pub batch: &'a Stage6SumcheckBatchPlan,
    pub driver: &'a Stage6SumcheckDriverPlan,
}

impl Stage6KernelContext<'_> {
    pub fn relation_kind(&self) -> Result<Stage6Relation, Stage6KernelError> {
        self.kernel.relation_kind()
    }

    pub fn abi_kind(&self) -> Result<Stage6KernelAbi, Stage6KernelError> {
        self.kernel.abi_kind()
    }

    pub fn batch_claims(&self) -> Result<Vec<&'static Stage6SumcheckClaimPlan>, Stage6KernelError> {
        self.batch
            .claim_operands
            .iter()
            .map(|symbol| {
                self.program
                    .claim(symbol)
                    .ok_or(Stage6KernelError::MissingClaim {
                        batch: self.batch.symbol,
                        claim: symbol,
                    })
            })
            .collect()
    }
}

pub trait Stage6KernelExecutor<F: Field> {
    fn observe_challenge_vector(
        &mut self,
        _plan: &'static Stage6TranscriptSqueezePlan,
        _values: &[F],
    ) -> Result<(), Stage6KernelError> {
        Ok(())
    }

    fn observe_sumcheck_output(
        &mut self,
        _output: &Stage6SumcheckOutput<F>,
    ) -> Result<(), Stage6KernelError> {
        Ok(())
    }

    fn prove_sumcheck<T>(
        &mut self,
        context: Stage6KernelContext<'_>,
        transcript: &mut T,
    ) -> Result<Stage6SumcheckOutput<F>, Stage6KernelError>
    where
        T: Transcript<Challenge = F>;

    fn verify_sumcheck<T>(
        &mut self,
        context: Stage6KernelContext<'_>,
        transcript: &mut T,
    ) -> Result<Stage6SumcheckOutput<F>, Stage6KernelError>
    where
        T: Transcript<Challenge = F>;
}

#[derive(Clone, Debug, Default)]
pub struct UnsupportedStage6KernelExecutor;

impl<F: Field> Stage6KernelExecutor<F> for UnsupportedStage6KernelExecutor {
    fn prove_sumcheck<T>(
        &mut self,
        context: Stage6KernelContext<'_>,
        _transcript: &mut T,
    ) -> Result<Stage6SumcheckOutput<F>, Stage6KernelError>
    where
        T: Transcript<Challenge = F>,
    {
        let abi = context.abi_kind()?;
        let _ = context.relation_kind()?;
        Err(Stage6KernelError::KernelNotImplemented { abi: abi.name() })
    }

    fn verify_sumcheck<T>(
        &mut self,
        context: Stage6KernelContext<'_>,
        _transcript: &mut T,
    ) -> Result<Stage6SumcheckOutput<F>, Stage6KernelError>
    where
        T: Transcript<Challenge = F>,
    {
        let abi = context.abi_kind()?;
        let _ = context.relation_kind()?;
        Err(Stage6KernelError::KernelNotImplemented { abi: abi.name() })
    }
}

#[derive(Clone)]
pub struct Stage6ProverKernelExecutor<'a, F: Field> {
    pub inputs: Stage6ProverInputs<'a, F>,
    challenge_vectors: Vec<Stage6ChallengeVector<F>>,
    completed_sumchecks: Vec<Stage6SumcheckOutput<F>>,
}

impl<'a, F: Field> Stage6ProverKernelExecutor<'a, F> {
    pub fn new(inputs: Stage6ProverInputs<'a, F>) -> Self {
        Self {
            inputs,
            challenge_vectors: Vec::new(),
            completed_sumchecks: Vec::new(),
        }
    }

    fn value_store(
        &self,
        program: &'static Stage6CpuProgramPlan,
    ) -> Result<Stage6ValueStore<F>, Stage6KernelError> {
        value_store_from_observations(
            program,
            self.inputs.opening_inputs,
            &self.challenge_vectors,
            &self.completed_sumchecks,
        )
    }
}

impl<F: Field> Stage6KernelExecutor<F> for Stage6ProverKernelExecutor<'_, F> {
    fn observe_challenge_vector(
        &mut self,
        plan: &'static Stage6TranscriptSqueezePlan,
        values: &[F],
    ) -> Result<(), Stage6KernelError> {
        self.challenge_vectors.push(Stage6ChallengeVector {
            symbol: plan.symbol,
            values: values.to_vec(),
        });
        Ok(())
    }

    fn observe_sumcheck_output(
        &mut self,
        output: &Stage6SumcheckOutput<F>,
    ) -> Result<(), Stage6KernelError> {
        self.completed_sumchecks.push(output.clone());
        Ok(())
    }

    fn prove_sumcheck<T>(
        &mut self,
        context: Stage6KernelContext<'_>,
        transcript: &mut T,
    ) -> Result<Stage6SumcheckOutput<F>, Stage6KernelError>
    where
        T: Transcript<Challenge = F>,
    {
        prove_stage6_kernel(
            context,
            &self.inputs,
            self.value_store(context.program)?,
            transcript,
        )
    }

    fn verify_sumcheck<T>(
        &mut self,
        context: Stage6KernelContext<'_>,
        _transcript: &mut T,
    ) -> Result<Stage6SumcheckOutput<F>, Stage6KernelError>
    where
        T: Transcript<Challenge = F>,
    {
        Err(Stage6KernelError::WrongExecutorMode {
            driver: context.driver.symbol,
            expected: Stage6ExecutionMode::Prover,
            actual: Stage6ExecutionMode::Verifier,
        })
    }
}

#[derive(Clone)]
pub struct Stage6ProofCarryingKernelExecutor<'a, F: Field> {
    pub proof: &'a Stage6Proof<F>,
    pub opening_inputs: &'a [Stage6OpeningInputValue<F>],
    pub bytecode_read_raf: Option<Stage6BytecodeReadRafData<'a, F>>,
    pub cursor: usize,
    challenge_vectors: Vec<Stage6ChallengeVector<F>>,
    completed_sumchecks: Vec<Stage6SumcheckOutput<F>>,
}

impl<'a, F: Field> Stage6ProofCarryingKernelExecutor<'a, F> {
    pub fn new(
        proof: &'a Stage6Proof<F>,
        opening_inputs: &'a [Stage6OpeningInputValue<F>],
    ) -> Self {
        Self {
            proof,
            opening_inputs,
            bytecode_read_raf: None,
            cursor: 0,
            challenge_vectors: Vec::new(),
            completed_sumchecks: Vec::new(),
        }
    }

    pub fn with_bytecode_read_raf_data(
        mut self,
        bytecode_read_raf: Stage6BytecodeReadRafData<'a, F>,
    ) -> Self {
        self.bytecode_read_raf = Some(bytecode_read_raf);
        self
    }

    fn value_store(
        &self,
        program: &'static Stage6CpuProgramPlan,
    ) -> Result<Stage6ValueStore<F>, Stage6KernelError> {
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
    ) -> Result<&'a Stage6SumcheckOutput<F>, Stage6KernelError> {
        let proof = self
            .proof
            .sumchecks
            .get(self.cursor)
            .ok_or(Stage6KernelError::MissingProof { driver })?;
        self.cursor += 1;
        Ok(proof)
    }
}

impl<F: Field> Stage6KernelExecutor<F> for Stage6ProofCarryingKernelExecutor<'_, F> {
    fn observe_challenge_vector(
        &mut self,
        plan: &'static Stage6TranscriptSqueezePlan,
        values: &[F],
    ) -> Result<(), Stage6KernelError> {
        self.challenge_vectors.push(Stage6ChallengeVector {
            symbol: plan.symbol,
            values: values.to_vec(),
        });
        Ok(())
    }

    fn observe_sumcheck_output(
        &mut self,
        output: &Stage6SumcheckOutput<F>,
    ) -> Result<(), Stage6KernelError> {
        self.completed_sumchecks.push(output.clone());
        Ok(())
    }

    fn prove_sumcheck<T>(
        &mut self,
        context: Stage6KernelContext<'_>,
        transcript: &mut T,
    ) -> Result<Stage6SumcheckOutput<F>, Stage6KernelError>
    where
        T: Transcript<Challenge = F>,
    {
        let proof = self.next_proof(context.driver.symbol)?;
        verify_stage6_kernel(
            context,
            self.value_store(context.program)?,
            proof,
            self.bytecode_read_raf,
            transcript,
        )
    }

    fn verify_sumcheck<T>(
        &mut self,
        context: Stage6KernelContext<'_>,
        transcript: &mut T,
    ) -> Result<Stage6SumcheckOutput<F>, Stage6KernelError>
    where
        T: Transcript<Challenge = F>,
    {
        let proof = self.next_proof(context.driver.symbol)?;
        verify_stage6_kernel(
            context,
            self.value_store(context.program)?,
            proof,
            self.bytecode_read_raf,
            transcript,
        )
    }
}

#[derive(Clone, Debug, Default)]
struct Stage6ValueStore<F: Field> {
    scalars: Vec<(&'static str, F)>,
    points: Vec<(&'static str, Vec<F>)>,
}

impl<F: Field> Stage6ValueStore<F> {
    fn with_opening_inputs(inputs: &[Stage6OpeningInputValue<F>]) -> Self {
        let mut store = Self::default();
        for input in inputs {
            store.insert_scalar(input.symbol, input.eval);
            store.insert_point(input.symbol, input.point.clone());
        }
        store
    }

    fn seed_constants(&mut self, program: &'static Stage6CpuProgramPlan) {
        for constant in program.field_constants {
            self.insert_scalar(constant.symbol, F::from_u64(constant.value as u64));
        }
        for zero in program.point_zeros {
            self.insert_point(zero.symbol, vec![F::from_u64(0); zero.arity]);
        }
    }

    fn observe_challenge_vector(
        &mut self,
        plan: &'static Stage6TranscriptSqueezePlan,
        values: &[F],
    ) -> Result<(), Stage6KernelError> {
        self.insert_point(plan.symbol, values.to_vec());
        if matches!(plan.kind, "challenge_scalar" | "scalar") {
            require_operand_count(plan.symbol, 1, values.len())?;
            self.insert_scalar(plan.symbol, values[0]);
        }
        Ok(())
    }

    fn observe_sumcheck_output(
        &mut self,
        program: &'static Stage6CpuProgramPlan,
        output: &Stage6SumcheckOutput<F>,
    ) -> Result<(), Stage6KernelError> {
        self.observe_sumcheck_values(program, output.driver, &output.point, &output.evals)
    }

    fn observe_sumcheck_values(
        &mut self,
        program: &'static Stage6CpuProgramPlan,
        driver: &'static str,
        point: &[F],
        evals: &[Stage6NamedEval<F>],
    ) -> Result<(), Stage6KernelError> {
        self.insert_point(driver, point.to_vec());
        for instance in program
            .instance_results
            .iter()
            .filter(|instance| instance.source == driver)
        {
            let end = instance.round_offset + instance.point_arity;
            let mut point = point
                .get(instance.round_offset..end)
                .ok_or(Stage6KernelError::InvalidInputLength {
                    input: instance.symbol,
                    expected: end,
                    actual: point.len(),
                })?
                .to_vec();
            match instance.point_order {
                "as_is" | "stage6_booleanity" => {}
                "reverse" => point.reverse(),
                "bytecode_read_raf" => point = normalize_bytecode_read_raf_point(program, &point)?,
                "instruction_read_raf" => point = normalize_instruction_read_raf_point(&point)?,
                _ => {
                    return Err(Stage6KernelError::InvalidProof {
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
                .ok_or(Stage6KernelError::MissingValue {
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
        program: &'static Stage6CpuProgramPlan,
    ) -> Result<usize, Stage6KernelError> {
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
                self.insert_scalar(expr.symbol, evaluate_stage6_field_expr(expr, &operands)?);
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
        program: &'static Stage6CpuProgramPlan,
    ) -> Result<usize, Stage6KernelError> {
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
                    .ok_or(Stage6KernelError::InvalidInputLength {
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
        program: &'static Stage6CpuProgramPlan,
    ) -> Result<(), Stage6KernelError> {
        for equality in program.opening_equalities {
            match equality.mode {
                "point_and_eval" => {
                    if self.point(equality.lhs)? != self.point(equality.rhs)?
                        || self.scalar(equality.lhs)? != self.scalar(equality.rhs)?
                    {
                        return Err(Stage6KernelError::InvalidProof {
                            driver: equality.symbol,
                            reason: "opening claim equality failed",
                        });
                    }
                }
                _ => {
                    return Err(Stage6KernelError::InvalidProof {
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
        program: &'static Stage6CpuProgramPlan,
        claim: &Stage6SumcheckClaimPlan,
    ) -> Result<F, Stage6KernelError> {
        let _ = self.evaluate_available_field_exprs(program)?;
        self.scalar(claim.claim_value)
    }

    fn batch_claim_values(
        &mut self,
        program: &'static Stage6CpuProgramPlan,
        batch: &Stage6SumcheckBatchPlan,
    ) -> Result<Vec<F>, Stage6KernelError> {
        batch
            .claim_operands
            .iter()
            .map(|symbol| {
                let claim = program
                    .claim(symbol)
                    .ok_or(Stage6KernelError::MissingClaim {
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

    fn scalar(&self, symbol: &'static str) -> Result<F, Stage6KernelError> {
        self.try_scalar(symbol)
            .ok_or(Stage6KernelError::MissingValue { symbol })
    }

    fn try_scalar(&self, symbol: &str) -> Option<F> {
        self.scalars
            .iter()
            .find(|(existing, _)| *existing == symbol)
            .map(|(_, value)| *value)
    }

    fn point(&self, symbol: &'static str) -> Result<&[F], Stage6KernelError> {
        self.try_point(symbol)
            .ok_or(Stage6KernelError::MissingValue { symbol })
    }

    fn try_point(&self, symbol: &str) -> Option<&[F]> {
        self.points
            .iter()
            .find(|(existing, _)| *existing == symbol)
            .map(|(_, point)| point.as_slice())
    }

    fn try_expr_operands(&self, expr: &Stage6FieldExprPlan) -> Option<Vec<F>> {
        expr.operands
            .iter()
            .map(|operand| self.try_scalar(operand))
            .collect()
    }

    fn try_concat_point(&self, concat: &Stage6PointConcatPlan) -> Option<Vec<F>> {
        let mut point = Vec::with_capacity(concat.arity);
        for input in concat.inputs {
            point.extend_from_slice(self.try_point(input)?);
        }
        Some(point)
    }
}

fn value_store_from_observations<F: Field>(
    program: &'static Stage6CpuProgramPlan,
    opening_inputs: &[Stage6OpeningInputValue<F>],
    challenge_vectors: &[Stage6ChallengeVector<F>],
    completed_sumchecks: &[Stage6SumcheckOutput<F>],
) -> Result<Stage6ValueStore<F>, Stage6KernelError> {
    let mut store = Stage6ValueStore::with_opening_inputs(opening_inputs);
    store.seed_constants(program);
    for challenge in challenge_vectors {
        let plan =
            find_squeeze(program, challenge.symbol).ok_or(Stage6KernelError::MissingValue {
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

fn prove_stage6_kernel<F, T>(
    context: Stage6KernelContext<'_>,
    inputs: &Stage6ProverInputs<'_, F>,
    store: Stage6ValueStore<F>,
    transcript: &mut T,
) -> Result<Stage6SumcheckOutput<F>, Stage6KernelError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    match context.abi_kind()? {
        Stage6KernelAbi::Batched => prove_batched_stage6(context, inputs, store, transcript),
        abi => Err(Stage6KernelError::KernelNotImplemented { abi: abi.name() }),
    }
}

fn verify_stage6_kernel<F, T>(
    context: Stage6KernelContext<'_>,
    store: Stage6ValueStore<F>,
    proof: &Stage6SumcheckOutput<F>,
    bytecode_read_raf: Option<Stage6BytecodeReadRafData<'_, F>>,
    transcript: &mut T,
) -> Result<Stage6SumcheckOutput<F>, Stage6KernelError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    match context.abi_kind()? {
        Stage6KernelAbi::Batched => {
            verify_batched_stage6(context, store, proof, bytecode_read_raf, transcript)
        }
        abi => Err(Stage6KernelError::KernelNotImplemented { abi: abi.name() }),
    }
}

#[tracing::instrument(skip_all, name = "Stage6::prove_batched")]
fn prove_batched_stage6<F, T>(
    context: Stage6KernelContext<'_>,
    inputs: &Stage6ProverInputs<'_, F>,
    mut store: Stage6ValueStore<F>,
    transcript: &mut T,
) -> Result<Stage6SumcheckOutput<F>, Stage6KernelError>
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
    let timing_enabled = std::env::var_os("JOLT_STAGE6_KERNEL_TIMINGS").is_some();
    let two_inv = F::from_u64(2)
        .inverse()
        .ok_or(Stage6KernelError::InvalidProof {
            driver: context.driver.symbol,
            reason: "field element 2 is not invertible",
        })?;
    let mut instances = Vec::with_capacity(claims.len());
    let mut timing_stats = Vec::with_capacity(claims.len());
    for (index, claim) in claims.iter().enumerate() {
        let offset = instance_round_offset(context.program, context.driver.symbol, claim.symbol)?;
        if offset + claim.num_rounds > max_rounds {
            return Err(Stage6KernelError::InvalidInputLength {
                input: claim.symbol,
                expected: max_rounds,
                actual: offset + claim.num_rounds,
            });
        }
        let relation = claim_relation(context.program, claim)?;
        let active_scale = F::one().mul_pow_2(max_rounds - offset - claim.num_rounds);
        let init_start = timing_enabled.then(std::time::Instant::now);
        let state =
            Stage6ProverInstanceState::new(context.program, claim, inputs, &store, active_scale)?;
        let init_nanos = init_start.map_or(0, |start| start.elapsed().as_nanos());
        instances.push(Stage6BatchedInstance {
            claim,
            relation,
            offset,
            previous_claim: input_claims[index].mul_pow_2(max_rounds - claim.num_rounds),
            state,
        });
        timing_stats.push((relation, init_nanos, 0u128, 0u128));
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
        for (index, instance) in instances.iter_mut().enumerate() {
            let poly = if instance.is_active(round) {
                let round_start = timing_enabled.then(std::time::Instant::now);
                let poly = instance
                    .state
                    .round_poly(instance.previous_claim, instance.relation)?;
                if let Some(start) = round_start {
                    timing_stats[index].2 += start.elapsed().as_nanos();
                }
                poly
            } else {
                UnivariatePoly::new(vec![instance.previous_claim * two_inv])
            };
            individual_polys.push(poly);
        }
        let batched_poly = combine_univariate_polys(&individual_polys, &batching_coeffs);
        if batched_poly.evaluate(F::zero()) + batched_poly.evaluate(F::one()) != batched_claim {
            return Err(Stage6KernelError::InvalidProof {
                driver: context.driver.symbol,
                reason: "batched round claim mismatch",
            });
        }
        append_compressed_univariate_poly(transcript, context.driver.round_label, &batched_poly);
        let challenge = transcript.challenge();
        point.push(challenge);
        batched_claim = batched_poly.evaluate(challenge);
        for (index, (instance, poly)) in instances.iter_mut().zip(individual_polys).enumerate() {
            instance.previous_claim = poly.evaluate(challenge);
            if instance.is_active(round) {
                let bind_start = timing_enabled.then(std::time::Instant::now);
                instance.state.ingest_challenge(challenge);
                if let Some(start) = bind_start {
                    timing_stats[index].3 += start.elapsed().as_nanos();
                }
            }
        }
        round_polynomials.push(batched_poly);
    }

    let mut evals = Vec::new();
    let mut expected = F::zero();
    for (instance, &coefficient) in instances.iter().zip(&batching_coeffs) {
        let relation_claim = instance.state.final_relation_eval(instance.relation)?;
        if instance.previous_claim != relation_claim {
            return Err(Stage6KernelError::InvalidProof {
                driver: instance.relation.symbol(),
                reason: "stage6 relation output claim mismatch",
            });
        }
        expected += coefficient * relation_claim;
        evals.extend(instance.state.final_evals(instance.relation)?);
    }
    if batched_claim != expected {
        return Err(Stage6KernelError::InvalidProof {
            driver: context.driver.symbol,
            reason: "batched output claim mismatch",
        });
    }
    store.observe_sumcheck_values(context.program, context.driver.symbol, &point, &evals)?;
    append_opening_claims(context.program, &mut store, transcript, &evals)?;
    if timing_enabled {
        for (relation, init_nanos, round_nanos, bind_nanos) in timing_stats {
            eprintln!(
                "[stage6 timings] relation={} init_ms={:.3} round_ms={:.3} bind_ms={:.3}",
                relation.symbol(),
                init_nanos as f64 / 1_000_000.0,
                round_nanos as f64 / 1_000_000.0,
                bind_nanos as f64 / 1_000_000.0,
            );
        }
    }
    Ok(Stage6SumcheckOutput {
        driver: context.driver.symbol,
        point,
        evals,
        proof: jolt_sumcheck::SumcheckProof { round_polynomials },
    })
}

fn verify_batched_stage6<F, T>(
    context: Stage6KernelContext<'_>,
    mut store: Stage6ValueStore<F>,
    proof: &Stage6SumcheckOutput<F>,
    bytecode_read_raf: Option<Stage6BytecodeReadRafData<'_, F>>,
    transcript: &mut T,
) -> Result<Stage6SumcheckOutput<F>, Stage6KernelError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    if proof.driver != context.driver.symbol {
        return Err(Stage6KernelError::InvalidProof {
            driver: context.driver.symbol,
            reason: "driver symbol mismatch",
        });
    }
    if proof.proof.round_polynomials.len() != context.driver.num_rounds {
        return Err(Stage6KernelError::InvalidProof {
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
            return Err(Stage6KernelError::InvalidProof {
                driver: context.driver.symbol,
                reason: "batched polynomial exceeds degree bound",
            });
        }
        if poly.evaluate(F::zero()) + poly.evaluate(F::one()) != running_claim {
            return Err(Stage6KernelError::InvalidProof {
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
        return Err(Stage6KernelError::InvalidProof {
            driver: context.driver.symbol,
            reason: "batched point mismatch",
        });
    }
    if let Some(expected) = expected_batched_output_claim_if_supported(
        context,
        &store,
        &proof.evals,
        &point,
        &batching_coeffs,
        bytecode_read_raf,
    )? {
        if running_claim != expected {
            return Err(Stage6KernelError::InvalidProof {
                driver: context.driver.symbol,
                reason: "batched output claim mismatch",
            });
        }
    }

    let output = Stage6SumcheckOutput {
        driver: context.driver.symbol,
        point,
        evals: proof.evals.clone(),
        proof: proof.proof.clone(),
    };
    store.observe_sumcheck_output(context.program, &output)?;
    append_opening_claims(context.program, &mut store, transcript, &output.evals)?;
    Ok(output)
}

fn expected_batched_output_claim_if_supported<F: Field>(
    context: Stage6KernelContext<'_>,
    store: &Stage6ValueStore<F>,
    evals: &[Stage6NamedEval<F>],
    point: &[F],
    batching_coeffs: &[F],
    bytecode_read_raf: Option<Stage6BytecodeReadRafData<'_, F>>,
) -> Result<Option<F>, Stage6KernelError> {
    let mut expected = F::zero();
    for (claim, &coefficient) in context.batch_claims()?.iter().zip(batching_coeffs) {
        let Some(instance) = context.program.instance_results.iter().find(|instance| {
            instance.claim == claim.symbol && instance.source == context.driver.symbol
        }) else {
            return Ok(None);
        };
        let local_point = point
            .get(instance.round_offset..instance.round_offset + instance.num_rounds)
            .ok_or(Stage6KernelError::InvalidInputLength {
                input: instance.symbol,
                expected: instance.round_offset + instance.num_rounds,
                actual: point.len(),
            })?;
        let value = match claim_relation(context.program, claim)? {
            Stage6Relation::HammingBooleanity => {
                expected_hamming_booleanity(store, evals, local_point)?
            }
            Stage6Relation::IncClaimReduction => {
                expected_inc_claim_reduction(store, evals, local_point)?
            }
            Stage6Relation::RamRaVirtual => expected_ram_ra_virtual(store, evals, local_point)?,
            Stage6Relation::InstructionRaVirtual => {
                expected_instruction_ra_virtual(context.program, store, evals, local_point)?
            }
            Stage6Relation::Booleanity => {
                expected_booleanity(context.program, store, evals, local_point)?
            }
            Stage6Relation::BytecodeReadRaf => {
                let Some(data) = bytecode_read_raf else {
                    return Ok(None);
                };
                expected_bytecode_read_raf(context.program, data, store, evals, local_point)?
            }
            Stage6Relation::Batched => return Ok(None),
        };
        expected += coefficient * value;
    }
    Ok(Some(expected))
}

fn expected_bytecode_read_raf<F: Field>(
    program: &'static Stage6CpuProgramPlan,
    data: Stage6BytecodeReadRafData<'_, F>,
    store: &Stage6ValueStore<F>,
    evals: &[Stage6NamedEval<F>],
    local_point: &[F],
) -> Result<F, Stage6KernelError> {
    let log_t = stage6_trace_rounds(program)?;
    let opening_point = normalize_bytecode_read_raf_point(program, local_point)?;
    let log_k = opening_point.len() - log_t;
    let (r_address_prime, r_cycle_prime) = opening_point.split_at(log_k);

    let gamma = store.scalar("stage6.bytecode_read_raf.gamma")?;
    let gamma_powers = bytecode_gamma_powers(gamma);
    let int_eval = identity_polynomial_eval(r_address_prime);
    let stage_value_evals = bytecode_stage_value_evals(data, store, r_address_prime, log_t)?;
    let stage_cycle_points = bytecode_stage_cycle_points(store, log_t)?;
    let int_contrib = [
        gamma_powers[5] * int_eval,
        F::zero(),
        gamma_powers[4] * int_eval,
        F::zero(),
        F::zero(),
    ];

    let mut val = F::zero();
    for index in 0..stage_value_evals.len() {
        val += (stage_value_evals[index] + int_contrib[index])
            * EqPolynomial::<F>::mle(&stage_cycle_points[index], r_cycle_prime)
            * gamma_powers[index];
    }

    let entry_bits = index_bits(data.entry_bytecode_index, log_k)?;
    let zero_cycle = vec![F::zero(); r_cycle_prime.len()];
    let entry_contrib = gamma_powers[7]
        * EqPolynomial::<F>::mle(&entry_bits, r_address_prime)
        * EqPolynomial::<F>::mle(&zero_cycle, r_cycle_prime);
    let bytecode_ra =
        indexed_evals_by_prefix_any(evals, "stage6.bytecode_read_raf.eval.BytecodeRa_")?
            .into_iter()
            .product::<F>();
    Ok((val + entry_contrib) * bytecode_ra)
}

fn expected_booleanity<F: Field>(
    program: &'static Stage6CpuProgramPlan,
    store: &Stage6ValueStore<F>,
    evals: &[Stage6NamedEval<F>],
    local_point: &[F],
) -> Result<F, Stage6KernelError> {
    let log_t = stage6_trace_rounds(program)?;
    let log_k_chunk =
        local_point
            .len()
            .checked_sub(log_t)
            .ok_or(Stage6KernelError::InvalidInputLength {
                input: "stage6.booleanity.point",
                expected: log_t,
                actual: local_point.len(),
            })?;
    let stage5_point = store.point("stage6.input.stage5.instruction_read_raf.InstructionRa_0")?;
    let stage5_address_len =
        stage5_point
            .len()
            .checked_sub(log_t)
            .ok_or(Stage6KernelError::InvalidInputLength {
                input: "stage6.input.stage5.instruction_read_raf.InstructionRa_0",
                expected: log_t,
                actual: stage5_point.len(),
            })?;
    if stage5_address_len < log_k_chunk {
        return Err(Stage6KernelError::InvalidInputLength {
            input: "stage6.input.stage5.instruction_read_raf.InstructionRa_0",
            expected: log_k_chunk + log_t,
            actual: stage5_point.len(),
        });
    }

    let mut stage5_addr = stage5_point[..stage5_address_len].to_vec();
    stage5_addr.reverse();
    let mut combined_r = stage5_addr[stage5_address_len - log_k_chunk..].to_vec();
    combined_r.extend(stage5_point[stage5_address_len..].iter().rev().copied());
    require_operand_count(
        "stage6.booleanity.combined_point",
        local_point.len(),
        combined_r.len(),
    )?;
    let eq_eval = EqPolynomial::<F>::mle(local_point, &combined_r);

    let gamma = store.scalar("stage6.booleanity.gamma")?;
    let gamma_sq = gamma.square();
    let mut gamma_power = F::one();
    let mut booleanity = F::zero();
    for ra in booleanity_evals(evals)? {
        booleanity += gamma_power * (ra.square() - ra);
        gamma_power *= gamma_sq;
    }
    Ok(eq_eval * booleanity)
}

fn expected_hamming_booleanity<F: Field>(
    store: &Stage6ValueStore<F>,
    evals: &[Stage6NamedEval<F>],
    local_point: &[F],
) -> Result<F, Stage6KernelError> {
    let hamming = eval_by_name(evals, "stage6.hamming_booleanity.eval.HammingWeight")?;
    let lookup_output_point = reverse_slice(store.point("stage6.input.stage1.LookupOutput")?);
    require_operand_count(
        "stage6.input.stage1.LookupOutput",
        local_point.len(),
        lookup_output_point.len(),
    )?;
    let eq_eval = EqPolynomial::<F>::mle(local_point, &lookup_output_point);
    Ok((hamming.square() - hamming) * eq_eval)
}

fn expected_inc_claim_reduction<F: Field>(
    store: &Stage6ValueStore<F>,
    evals: &[Stage6NamedEval<F>],
    local_point: &[F],
) -> Result<F, Stage6KernelError> {
    let r_cycle_reduced = reverse_slice(local_point);
    let ram_inc_stage2 = suffix_point(
        store.point("stage6.input.stage2.ram_read_write.RamInc")?,
        r_cycle_reduced.len(),
        "stage6.input.stage2.ram_read_write.RamInc",
    )?;
    let ram_inc_stage4 = suffix_point(
        store.point("stage6.input.stage4.ram_val_check.RamInc")?,
        r_cycle_reduced.len(),
        "stage6.input.stage4.ram_val_check.RamInc",
    )?;
    let rd_inc_stage4 = suffix_point(
        store.point("stage6.input.stage4.registers_read_write.RdInc")?,
        r_cycle_reduced.len(),
        "stage6.input.stage4.registers_read_write.RdInc",
    )?;
    let rd_inc_stage5 = suffix_point(
        store.point("stage6.input.stage5.registers_val_evaluation.RdInc")?,
        r_cycle_reduced.len(),
        "stage6.input.stage5.registers_val_evaluation.RdInc",
    )?;
    let gamma = store.scalar("stage6.inc_claim_reduction.gamma")?;
    let eq_ram_combined = EqPolynomial::<F>::mle(ram_inc_stage2, &r_cycle_reduced)
        + gamma * EqPolynomial::<F>::mle(ram_inc_stage4, &r_cycle_reduced);
    let eq_rd_combined = EqPolynomial::<F>::mle(rd_inc_stage4, &r_cycle_reduced)
        + gamma * EqPolynomial::<F>::mle(rd_inc_stage5, &r_cycle_reduced);
    let ram_inc = eval_by_name(evals, "stage6.inc_claim_reduction.eval.RamInc")?;
    let rd_inc = eval_by_name(evals, "stage6.inc_claim_reduction.eval.RdInc")?;
    Ok(ram_inc * eq_ram_combined + gamma.square() * rd_inc * eq_rd_combined)
}

fn expected_ram_ra_virtual<F: Field>(
    store: &Stage6ValueStore<F>,
    evals: &[Stage6NamedEval<F>],
    local_point: &[F],
) -> Result<F, Stage6KernelError> {
    let r_cycle_reduced = reverse_slice(local_point);
    let r_cycle = suffix_point(
        store.point("stage6.input.stage5.ram_ra_claim_reduction.RamRa")?,
        r_cycle_reduced.len(),
        "stage6.input.stage5.ram_ra_claim_reduction.RamRa",
    )?;
    let eq_eval = EqPolynomial::<F>::mle(r_cycle, &r_cycle_reduced);
    let ram_ra = indexed_evals_by_prefix_any(evals, "stage6.ram_ra_virtual.eval.RamRa_")?
        .into_iter()
        .product::<F>();
    Ok(eq_eval * ram_ra)
}

fn expected_instruction_ra_virtual<F: Field>(
    program: &'static Stage6CpuProgramPlan,
    store: &Stage6ValueStore<F>,
    evals: &[Stage6NamedEval<F>],
    local_point: &[F],
) -> Result<F, Stage6KernelError> {
    let r_cycle_reduced = reverse_slice(local_point);
    let r_cycle = suffix_point(
        store.point("stage6.input.stage5.instruction_read_raf.InstructionRa_0")?,
        r_cycle_reduced.len(),
        "stage6.input.stage5.instruction_read_raf.InstructionRa_0",
    )?;
    let eq_eval = EqPolynomial::<F>::mle(r_cycle, &r_cycle_reduced);
    let committed_ra =
        indexed_evals_by_prefix_any(evals, "stage6.instruction_ra_virtual.eval.InstructionRa_")?;
    let virtual_count = program
        .opening_inputs
        .iter()
        .filter(|input| {
            input
                .symbol
                .starts_with("stage6.input.stage5.instruction_read_raf.InstructionRa_")
        })
        .count();
    if virtual_count == 0 || committed_ra.len() % virtual_count != 0 {
        return Err(Stage6KernelError::InvalidInputLength {
            input: "stage6.instruction_ra_virtual.eval.InstructionRa_",
            expected: virtual_count,
            actual: committed_ra.len(),
        });
    }
    let committed_per_virtual = committed_ra.len() / virtual_count;
    let gamma = store.scalar("stage6.instruction_ra_virtual.gamma")?;
    let mut gamma_power = F::one();
    let mut value = F::zero();
    for chunk in committed_ra.chunks(committed_per_virtual) {
        value += gamma_power * chunk.iter().copied().product::<F>();
        gamma_power *= gamma;
    }
    Ok(eq_eval * value)
}

struct Stage6BatchedInstance<'a, F: Field> {
    claim: &'a Stage6SumcheckClaimPlan,
    relation: Stage6Relation,
    offset: usize,
    previous_claim: F,
    state: Stage6ProverInstanceState<F>,
}

impl<F: Field> Stage6BatchedInstance<'_, F> {
    fn is_active(&self, round: usize) -> bool {
        round >= self.offset && round < self.offset + self.claim.num_rounds
    }
}

enum Stage6ProverInstanceState<F: Field> {
    Booleanity(BooleanityStage6State<F>),
    BytecodeReadRaf(BytecodeReadRafStage6State<F>),
    Dense(DenseStage6State<F>),
}

impl<F: Field> Stage6ProverInstanceState<F> {
    fn new(
        program: &'static Stage6CpuProgramPlan,
        claim: &Stage6SumcheckClaimPlan,
        inputs: &Stage6ProverInputs<'_, F>,
        store: &Stage6ValueStore<F>,
        active_scale: F,
    ) -> Result<Self, Stage6KernelError> {
        match claim_relation(program, claim)? {
            Stage6Relation::BytecodeReadRaf => {
                bytecode_read_raf_state(program, claim, inputs, store, active_scale)
            }
            Stage6Relation::Booleanity => {
                booleanity_state(program, claim, inputs, store, active_scale).map(Self::Booleanity)
            }
            Stage6Relation::HammingBooleanity => {
                hamming_booleanity_state(program, claim, inputs, store, active_scale)
                    .map(Self::Dense)
            }
            Stage6Relation::IncClaimReduction => {
                inc_claim_reduction_state(program, claim, inputs, store, active_scale)
                    .map(Self::Dense)
            }
            Stage6Relation::RamRaVirtual => {
                ram_ra_virtual_state(program, claim, inputs, store, active_scale).map(Self::Dense)
            }
            Stage6Relation::InstructionRaVirtual => {
                instruction_ra_virtual_state(program, claim, inputs, store, active_scale)
                    .map(Self::Dense)
            }
            relation @ Stage6Relation::Batched => Err(Stage6KernelError::KernelNotImplemented {
                abi: relation.symbol(),
            }),
        }
    }

    fn round_poly(
        &mut self,
        previous_claim: F,
        relation: Stage6Relation,
    ) -> Result<UnivariatePoly<F>, Stage6KernelError> {
        match self {
            Self::Booleanity(state) => state.round_poly(previous_claim, relation),
            Self::BytecodeReadRaf(state) => state.round_poly(previous_claim, relation),
            Self::Dense(state) => state.round_poly(previous_claim, relation),
        }
    }

    fn ingest_challenge(&mut self, challenge: F) {
        match self {
            Self::Booleanity(state) => state.bind(challenge),
            Self::BytecodeReadRaf(state) => state.bind(challenge),
            Self::Dense(state) => state.bind(challenge),
        }
    }

    fn final_relation_eval(&self, relation: Stage6Relation) -> Result<F, Stage6KernelError> {
        match self {
            Self::Booleanity(state) => state.final_relation_eval(relation),
            Self::BytecodeReadRaf(state) => state.final_relation_eval(relation),
            Self::Dense(state) => state.final_relation_eval(relation),
        }
    }

    fn final_evals(
        &self,
        relation: Stage6Relation,
    ) -> Result<Vec<Stage6NamedEval<F>>, Stage6KernelError> {
        match self {
            Self::Booleanity(state) => state.final_evals(relation),
            Self::BytecodeReadRaf(state) => state.final_evals(relation),
            Self::Dense(state) => state.final_evals(relation),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BytecodeReadRafPhase {
    Address,
    Cycle,
}

struct BytecodeReadRafStage6State<F: Field> {
    log_k: usize,
    log_t: usize,
    chunk_lens: Vec<usize>,
    bytecode_ra_chunks: Vec<Vec<F>>,
    stage_factors: [Vec<F>; BYTECODE_READ_RAF_STAGE_COUNT],
    stage_values: [Vec<F>; BYTECODE_READ_RAF_STAGE_COUNT],
    entry_trace: Vec<F>,
    entry_expected: Vec<F>,
    address_challenges: Vec<F>,
    cycle_factors: Vec<Vec<F>>,
    cycle_eqs: [Vec<F>; BYTECODE_READ_RAF_STAGE_COUNT],
    cycle_entry_eq: Vec<F>,
    bound_stage_values: Option<[F; BYTECODE_READ_RAF_STAGE_COUNT]>,
    bound_entry_expected: Option<F>,
    outputs: Vec<FactorOutput>,
    gamma_powers: [F; 8],
    active_scale: F,
    degree_bound: usize,
    phase: BytecodeReadRafPhase,
}

impl<F: Field> BytecodeReadRafStage6State<F> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        data: Stage6BytecodeReadRafData<'_, F>,
        bytecode_ra_chunks: &[&[F]],
        bytecode_cycle_indices: Vec<usize>,
        chunk_lens: Vec<usize>,
        store: &Stage6ValueStore<F>,
        log_k: usize,
        log_t: usize,
        active_scale: F,
        degree_bound: usize,
        outputs: Vec<FactorOutput>,
    ) -> Result<Self, Stage6KernelError> {
        if degree_bound < 2 || degree_bound < bytecode_ra_chunks.len() + 1 {
            return Err(Stage6KernelError::InvalidProof {
                driver: Stage6Relation::BytecodeReadRaf.symbol(),
                reason: "bytecode read RAF degree bound is too small",
            });
        }
        let expected_entries =
            1usize
                .checked_shl(log_k as u32)
                .ok_or(Stage6KernelError::InvalidInputLength {
                    input: "stage6.bytecode_read_raf.entries",
                    expected: usize::BITS as usize,
                    actual: log_k,
                })?;
        require_operand_count(
            "stage6.bytecode_read_raf.entries",
            expected_entries,
            data.entries.len(),
        )?;
        require_operand_count(
            "stage6.bytecode_read_raf.trace",
            1usize << log_t,
            bytecode_cycle_indices.len(),
        )?;
        if data.entry_bytecode_index >= expected_entries {
            return Err(Stage6KernelError::InvalidInputLength {
                input: "stage6.bytecode_read_raf.entry_bytecode_index",
                expected: expected_entries,
                actual: data.entry_bytecode_index + 1,
            });
        }
        if bytecode_cycle_indices
            .iter()
            .any(|&index| index >= expected_entries)
        {
            return Err(Stage6KernelError::InvalidInputLength {
                input: "stage6.bytecode_read_raf.BytecodeRa",
                expected: expected_entries,
                actual: expected_entries + 1,
            });
        }

        let gamma = store.scalar("stage6.bytecode_read_raf.gamma")?;
        let gamma_powers = bytecode_gamma_powers(gamma);
        let stage_cycle_points = bytecode_stage_cycle_points(store, log_t)?;
        let cycle_eqs = stage_cycle_points.each_ref().map(|point| {
            let eq = EqPolynomial::<F>::evals(point, None);
            debug_assert_eq!(eq.len(), 1usize << log_t);
            eq
        });

        let stage1_gamma = store.scalar("stage6.bytecode_read_raf.stage1_gamma")?;
        let stage2_gamma = store.scalar("stage6.bytecode_read_raf.stage2_gamma")?;
        let stage3_gamma = store.scalar("stage6.bytecode_read_raf.stage3_gamma")?;
        let stage4_gamma = store.scalar("stage6.bytecode_read_raf.stage4_gamma")?;
        let stage5_gamma = store.scalar("stage6.bytecode_read_raf.stage5_gamma")?;
        let stage1_gamma_powers = field_powers(stage1_gamma, 16);
        let stage2_gamma_powers = field_powers(stage2_gamma, 4);
        let stage3_gamma_powers = field_powers(stage3_gamma, 9);
        let stage4_gamma_powers = field_powers(stage4_gamma, 3);
        let stage5_gamma_powers = field_powers(stage5_gamma, data.num_lookup_tables + 2);
        let stage4_register_point =
            register_prefix_point(store, "stage6.input.stage4.Rs1Ra", log_t)?;
        let stage5_register_point = register_prefix_point(
            store,
            "stage6.input.stage5.registers_val_evaluation.RdWa",
            log_t,
        )?;

        let mut stage_values: [Vec<F>; BYTECODE_READ_RAF_STAGE_COUNT] =
            std::array::from_fn(|_| vec![F::zero(); expected_entries]);
        for (index, entry) in data.entries.iter().enumerate() {
            let mut values = bytecode_entry_stage_values(
                entry,
                data.num_lookup_tables,
                stage4_register_point,
                stage5_register_point,
                &stage1_gamma_powers,
                &stage2_gamma_powers,
                &stage3_gamma_powers,
                &stage4_gamma_powers,
                &stage5_gamma_powers,
            )?;
            let int_eval = F::from_u64(index as u64);
            values[0] += gamma_powers[5] * int_eval;
            values[2] += gamma_powers[4] * int_eval;
            for stage in 0..BYTECODE_READ_RAF_STAGE_COUNT {
                stage_values[stage][index] = values[stage];
            }
        }

        let mut stage_factors: [Vec<F>; BYTECODE_READ_RAF_STAGE_COUNT] =
            std::array::from_fn(|_| vec![F::zero(); expected_entries]);
        for (cycle, &bytecode_index) in bytecode_cycle_indices.iter().enumerate() {
            for stage in 0..BYTECODE_READ_RAF_STAGE_COUNT {
                stage_factors[stage][bytecode_index] += cycle_eqs[stage][cycle];
            }
        }

        let mut entry_trace = vec![F::zero(); expected_entries];
        entry_trace[bytecode_cycle_indices[0]] = F::one();
        let mut entry_expected = vec![F::zero(); expected_entries];
        entry_expected[data.entry_bytecode_index] = F::one();

        let mut cycle_entry_eq = vec![F::zero(); 1usize << log_t];
        cycle_entry_eq[0] = F::one();

        Ok(Self {
            log_k,
            log_t,
            chunk_lens,
            bytecode_ra_chunks: bytecode_ra_chunks
                .iter()
                .map(|chunk| (*chunk).to_vec())
                .collect(),
            stage_factors,
            stage_values,
            entry_trace,
            entry_expected,
            address_challenges: Vec::with_capacity(log_k),
            cycle_factors: Vec::new(),
            cycle_eqs,
            cycle_entry_eq,
            bound_stage_values: None,
            bound_entry_expected: None,
            outputs,
            gamma_powers,
            active_scale,
            degree_bound,
            phase: BytecodeReadRafPhase::Address,
        })
    }

    fn round_poly(
        &self,
        previous_claim: F,
        relation: Stage6Relation,
    ) -> Result<UnivariatePoly<F>, Stage6KernelError> {
        let poly = match self.phase {
            BytecodeReadRafPhase::Address => self.address_round_poly(relation)?,
            BytecodeReadRafPhase::Cycle => self.cycle_round_poly(relation)?,
        };
        if poly.evaluate(F::zero()) + poly.evaluate(F::one()) != previous_claim {
            return Err(Stage6KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "stage6 relation input claim mismatch",
            });
        }
        Ok(poly)
    }

    fn address_round_poly(
        &self,
        relation: Stage6Relation,
    ) -> Result<UnivariatePoly<F>, Stage6KernelError> {
        let first_len = self.stage_values[0].len();
        if first_len == 0 || !first_len.is_power_of_two() {
            return Err(Stage6KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "bytecode read RAF address phase has invalid length",
            });
        }
        let eval_count = self.degree_bound + 1;
        let mut evals = if first_len / 2 >= DENSE_BIND_PAR_THRESHOLD {
            (0..first_len / 2)
                .into_par_iter()
                .fold(
                    || vec![F::zero(); eval_count],
                    |mut row_evals, row| {
                        self.accumulate_address_row(row, &mut row_evals);
                        row_evals
                    },
                )
                .reduce(
                    || vec![F::zero(); eval_count],
                    |mut left, right| {
                        for (left, right) in left.iter_mut().zip(right) {
                            *left += right;
                        }
                        left
                    },
                )
        } else {
            let mut evals = vec![F::zero(); eval_count];
            for row in 0..first_len / 2 {
                self.accumulate_address_row(row, &mut evals);
            }
            evals
        };
        for eval in &mut evals {
            *eval *= self.active_scale;
        }
        Ok(UnivariatePoly::from_evals(&evals))
    }

    fn accumulate_address_row(&self, row: usize, evals: &mut [F]) {
        for (point_index, eval) in evals.iter_mut().enumerate() {
            let point = F::from_u64(point_index as u64);
            let mut value = F::zero();
            for stage in 0..BYTECODE_READ_RAF_STAGE_COUNT {
                let trace_eval = pair_linear_eval(&self.stage_factors[stage], row, point);
                let value_eval = pair_linear_eval(&self.stage_values[stage], row, point);
                value += self.gamma_powers[stage] * trace_eval * value_eval;
            }
            let entry_trace = pair_linear_eval(&self.entry_trace, row, point);
            let entry_expected = pair_linear_eval(&self.entry_expected, row, point);
            value += self.gamma_powers[7] * entry_trace * entry_expected;
            *eval += value;
        }
    }

    fn cycle_round_poly(
        &self,
        relation: Stage6Relation,
    ) -> Result<UnivariatePoly<F>, Stage6KernelError> {
        let Some(bound_stage_values) = self.bound_stage_values else {
            return Err(Stage6KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "bytecode read RAF cycle phase missing bound values",
            });
        };
        let Some(bound_entry_expected) = self.bound_entry_expected else {
            return Err(Stage6KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "bytecode read RAF cycle phase missing entry value",
            });
        };
        let first_len = self.cycle_factors.first().map_or(0, Vec::len);
        if first_len == 0 || !first_len.is_power_of_two() {
            return Err(Stage6KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "bytecode read RAF cycle phase has invalid length",
            });
        }
        let eval_count = self.degree_bound + 1;
        let mut evals = if first_len / 2 >= DENSE_BIND_PAR_THRESHOLD {
            (0..first_len / 2)
                .into_par_iter()
                .fold(
                    || vec![F::zero(); eval_count],
                    |mut row_evals, row| {
                        self.accumulate_cycle_row(
                            row,
                            bound_stage_values,
                            bound_entry_expected,
                            &mut row_evals,
                        );
                        row_evals
                    },
                )
                .reduce(
                    || vec![F::zero(); eval_count],
                    |mut left, right| {
                        for (left, right) in left.iter_mut().zip(right) {
                            *left += right;
                        }
                        left
                    },
                )
        } else {
            let mut evals = vec![F::zero(); eval_count];
            for row in 0..first_len / 2 {
                self.accumulate_cycle_row(
                    row,
                    bound_stage_values,
                    bound_entry_expected,
                    &mut evals,
                );
            }
            evals
        };
        for eval in &mut evals {
            *eval *= self.active_scale;
        }
        Ok(UnivariatePoly::from_evals(&evals))
    }

    fn accumulate_cycle_row(
        &self,
        row: usize,
        bound_stage_values: [F; BYTECODE_READ_RAF_STAGE_COUNT],
        bound_entry_expected: F,
        evals: &mut [F],
    ) {
        for (point_index, eval) in evals.iter_mut().enumerate() {
            let point = F::from_u64(point_index as u64);
            let mut ra_product = F::one();
            for factor in &self.cycle_factors {
                ra_product *= pair_linear_eval(factor, row, point);
            }
            let mut weighted_value = F::zero();
            for stage in 0..BYTECODE_READ_RAF_STAGE_COUNT {
                weighted_value += self.gamma_powers[stage]
                    * bound_stage_values[stage]
                    * pair_linear_eval(&self.cycle_eqs[stage], row, point);
            }
            weighted_value += self.gamma_powers[7]
                * bound_entry_expected
                * pair_linear_eval(&self.cycle_entry_eq, row, point);
            *eval += ra_product * weighted_value;
        }
    }

    fn bind(&mut self, challenge: F) {
        match self.phase {
            BytecodeReadRafPhase::Address => self.bind_address(challenge),
            BytecodeReadRafPhase::Cycle => self.bind_cycle(challenge),
        }
    }

    fn bind_address(&mut self, challenge: F) {
        for stage in 0..BYTECODE_READ_RAF_STAGE_COUNT {
            bind_dense_evals_reuse(&mut self.stage_factors[stage], &mut Vec::new(), challenge);
            bind_dense_evals_reuse(&mut self.stage_values[stage], &mut Vec::new(), challenge);
        }
        bind_dense_evals_reuse(&mut self.entry_trace, &mut Vec::new(), challenge);
        bind_dense_evals_reuse(&mut self.entry_expected, &mut Vec::new(), challenge);
        self.address_challenges.push(challenge);
        if self.address_challenges.len() == self.log_k {
            self.init_cycle_phase();
        }
    }

    fn init_cycle_phase(&mut self) {
        let bound_stage_values = std::array::from_fn(|stage| {
            self.stage_values[stage]
                .first()
                .copied()
                .unwrap_or(F::zero())
        });
        let bound_entry_expected = self.entry_expected.first().copied().unwrap_or(F::zero());
        let mut address_point = self.address_challenges.clone();
        address_point.reverse();

        self.cycle_factors = self
            .bytecode_ra_chunks
            .iter()
            .zip(&self.chunk_lens)
            .scan(0usize, |offset, (chunk, &chunk_len)| {
                let start = *offset;
                *offset += chunk_len;
                Some((chunk, start, chunk_len))
            })
            .map(|(chunk, offset, chunk_len)| {
                let eq_chunk =
                    EqPolynomial::<F>::evals(&address_point[offset..offset + chunk_len], None);
                let trace_len = 1usize << self.log_t;
                (0..trace_len)
                    .map(|cycle| {
                        eq_chunk
                            .iter()
                            .enumerate()
                            .map(|(chunk_value, &eq)| chunk[chunk_value * trace_len + cycle] * eq)
                            .sum()
                    })
                    .collect()
            })
            .collect();

        self.bound_stage_values = Some(bound_stage_values);
        self.bound_entry_expected = Some(bound_entry_expected);
        self.stage_factors = std::array::from_fn(|_| Vec::new());
        self.stage_values = std::array::from_fn(|_| Vec::new());
        self.entry_trace.clear();
        self.entry_expected.clear();
        self.phase = BytecodeReadRafPhase::Cycle;
    }

    fn bind_cycle(&mut self, challenge: F) {
        for factor in &mut self.cycle_factors {
            bind_dense_evals_reuse(factor, &mut Vec::new(), challenge);
        }
        for eq in &mut self.cycle_eqs {
            bind_dense_evals_reuse(eq, &mut Vec::new(), challenge);
        }
        bind_dense_evals_reuse(&mut self.cycle_entry_eq, &mut Vec::new(), challenge);
    }

    fn final_relation_eval(&self, relation: Stage6Relation) -> Result<F, Stage6KernelError> {
        let Some(bound_stage_values) = self.bound_stage_values else {
            return Err(Stage6KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "bytecode read RAF final eval missing bound values",
            });
        };
        let Some(bound_entry_expected) = self.bound_entry_expected else {
            return Err(Stage6KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "bytecode read RAF final eval missing entry value",
            });
        };
        let mut ra_product = F::one();
        for factor in &self.cycle_factors {
            ra_product *= factor
                .first()
                .copied()
                .ok_or(Stage6KernelError::InvalidProof {
                    driver: relation.symbol(),
                    reason: "bytecode read RAF final eval missing RA factor",
                })?;
        }
        let mut weighted_value = F::zero();
        for stage in 0..BYTECODE_READ_RAF_STAGE_COUNT {
            weighted_value += self.gamma_powers[stage]
                * bound_stage_values[stage]
                * self.cycle_eqs[stage].first().copied().ok_or(
                    Stage6KernelError::InvalidProof {
                        driver: relation.symbol(),
                        reason: "bytecode read RAF final eval missing cycle eq",
                    },
                )?;
        }
        weighted_value += self.gamma_powers[7]
            * bound_entry_expected
            * self
                .cycle_entry_eq
                .first()
                .copied()
                .ok_or(Stage6KernelError::InvalidProof {
                    driver: relation.symbol(),
                    reason: "bytecode read RAF final eval missing entry eq",
                })?;
        Ok(ra_product * weighted_value)
    }

    fn final_evals(
        &self,
        relation: Stage6Relation,
    ) -> Result<Vec<Stage6NamedEval<F>>, Stage6KernelError> {
        self.outputs
            .iter()
            .map(|output| {
                let factor =
                    output
                        .factor
                        .checked_sub(1)
                        .ok_or(Stage6KernelError::InvalidProof {
                            driver: relation.symbol(),
                            reason: "bytecode read RAF output factor underflow",
                        })?;
                let value = self
                    .cycle_factors
                    .get(factor)
                    .and_then(|values| values.first())
                    .copied()
                    .ok_or(Stage6KernelError::InvalidProof {
                        driver: relation.symbol(),
                        reason: "bytecode read RAF final eval missing output factor",
                    })?;
                Ok(named_eval(output.name, output.oracle, value))
            })
            .collect()
    }
}

#[inline]
fn pair_linear_eval<F: Field>(values: &[F], row: usize, point: F) -> F {
    let low = values[row << 1];
    let high = values[(row << 1) + 1];
    low + (high - low) * point
}

struct BooleanityStage6State<F: Field> {
    eq: Vec<F>,
    eq_scratch: Vec<F>,
    chunks: Vec<Vec<F>>,
    chunk_scratch: Vec<Vec<F>>,
    gamma_powers: Vec<F>,
    outputs: Vec<FactorOutput>,
    active_scale: F,
    degree_bound: usize,
}

impl<F: Field> BooleanityStage6State<F> {
    fn new(
        eq: Vec<F>,
        chunks: Vec<Vec<F>>,
        gamma_powers: Vec<F>,
        outputs: Vec<FactorOutput>,
        active_scale: F,
        degree_bound: usize,
    ) -> Result<Self, Stage6KernelError> {
        if degree_bound < 3 {
            return Err(Stage6KernelError::InvalidProof {
                driver: Stage6Relation::Booleanity.symbol(),
                reason: "booleanity degree bound is too small",
            });
        }
        require_operand_count("stage6.booleanity.gamma", chunks.len(), gamma_powers.len())?;
        if chunks.iter().any(|chunk| chunk.len() != eq.len()) {
            return Err(Stage6KernelError::InvalidProof {
                driver: Stage6Relation::Booleanity.symbol(),
                reason: "booleanity chunks have inconsistent lengths",
            });
        }
        let chunk_scratch = (0..chunks.len()).map(|_| Vec::new()).collect();
        Ok(Self {
            eq,
            eq_scratch: Vec::new(),
            chunks,
            chunk_scratch,
            gamma_powers,
            outputs,
            active_scale,
            degree_bound,
        })
    }

    fn round_poly(
        &self,
        previous_claim: F,
        relation: Stage6Relation,
    ) -> Result<UnivariatePoly<F>, Stage6KernelError> {
        let first_len = self.eq.len();
        if first_len == 0 || !first_len.is_power_of_two() {
            return Err(Stage6KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "booleanity factor has invalid length",
            });
        }
        let mut evals = if self.degree_bound == 3 {
            self.round_evals_degree3(first_len / 2)
        } else {
            self.round_evals_generic(first_len / 2)
        };
        for eval in &mut evals {
            *eval *= self.active_scale;
        }
        let poly = UnivariatePoly::from_evals(&evals);
        if poly.evaluate(F::zero()) + poly.evaluate(F::one()) != previous_claim {
            return Err(Stage6KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "stage6 relation input claim mismatch",
            });
        }
        Ok(poly)
    }

    fn round_evals_degree3(&self, half_len: usize) -> Vec<F> {
        let evals = if half_len >= DENSE_BIND_PAR_THRESHOLD {
            (0..half_len)
                .into_par_iter()
                .fold(
                    || [F::zero(); 4],
                    |mut row_evals, row| {
                        self.accumulate_row_degree3(row, &mut row_evals);
                        row_evals
                    },
                )
                .reduce(
                    || [F::zero(); 4],
                    |left, right| {
                        [
                            left[0] + right[0],
                            left[1] + right[1],
                            left[2] + right[2],
                            left[3] + right[3],
                        ]
                    },
                )
        } else {
            let mut evals = [F::zero(); 4];
            for row in 0..half_len {
                self.accumulate_row_degree3(row, &mut evals);
            }
            evals
        };
        evals.to_vec()
    }

    fn accumulate_row_degree3(&self, row: usize, evals: &mut [F; 4]) {
        let eq_low = self.eq[row << 1];
        let eq_high = self.eq[(row << 1) + 1];
        let delta_eq = eq_high - eq_low;
        let eq_at_0 = eq_low;
        let eq_at_1 = eq_high;
        let eq_at_2 = eq_low + delta_eq.mul_u64(2);
        let eq_at_3 = eq_low + delta_eq.mul_u64(3);
        for (chunk_index, chunk) in self.chunks.iter().enumerate() {
            let ra_low = chunk[row << 1];
            let ra_high = chunk[(row << 1) + 1];
            if ra_low == F::zero() && ra_high == F::zero() {
                continue;
            }
            let delta_ra = ra_high - ra_low;
            let gamma_power = self.gamma_powers[chunk_index];
            let ra_at_0 = ra_low;
            let ra_at_1 = ra_high;
            let ra_at_2 = ra_low + delta_ra.mul_u64(2);
            let ra_at_3 = ra_low + delta_ra.mul_u64(3);
            evals[0] += gamma_power * eq_at_0 * (ra_at_0.square() - ra_at_0);
            evals[1] += gamma_power * eq_at_1 * (ra_at_1.square() - ra_at_1);
            evals[2] += gamma_power * eq_at_2 * (ra_at_2.square() - ra_at_2);
            evals[3] += gamma_power * eq_at_3 * (ra_at_3.square() - ra_at_3);
        }
    }

    fn round_evals_generic(&self, half_len: usize) -> Vec<F> {
        let eval_count = self.degree_bound + 1;
        if half_len >= DENSE_BIND_PAR_THRESHOLD {
            (0..half_len)
                .into_par_iter()
                .fold(
                    || vec![F::zero(); eval_count],
                    |mut row_evals, row| {
                        self.accumulate_row_generic(row, &mut row_evals);
                        row_evals
                    },
                )
                .reduce(
                    || vec![F::zero(); eval_count],
                    |mut left, right| {
                        for (left, right) in left.iter_mut().zip(right) {
                            *left += right;
                        }
                        left
                    },
                )
        } else {
            let mut evals = vec![F::zero(); eval_count];
            for row in 0..half_len {
                self.accumulate_row_generic(row, &mut evals);
            }
            evals
        }
    }

    fn accumulate_row_generic(&self, row: usize, evals: &mut [F]) {
        let eq_low = self.eq[row << 1];
        let eq_high = self.eq[(row << 1) + 1];
        let delta_eq = eq_high - eq_low;
        for (chunk_index, chunk) in self.chunks.iter().enumerate() {
            let ra_low = chunk[row << 1];
            let ra_high = chunk[(row << 1) + 1];
            if ra_low == F::zero() && ra_high == F::zero() {
                continue;
            }
            let delta_ra = ra_high - ra_low;
            let gamma_power = self.gamma_powers[chunk_index];
            for (point_index, eval) in evals.iter_mut().enumerate() {
                let point = F::from_u64(point_index as u64);
                let eq_at_point = eq_low + delta_eq * point;
                let ra_at_point = ra_low + delta_ra * point;
                *eval += gamma_power * eq_at_point * (ra_at_point.square() - ra_at_point);
            }
        }
    }

    fn bind(&mut self, challenge: F) {
        bind_dense_evals_reuse(&mut self.eq, &mut self.eq_scratch, challenge);
        if self.eq.len() >= DENSE_BIND_PAR_THRESHOLD {
            self.chunks
                .par_iter_mut()
                .zip(self.chunk_scratch.par_iter_mut())
                .for_each(|(chunk, scratch)| {
                    bind_dense_evals_reuse(chunk, scratch, challenge);
                });
        } else {
            for (chunk, scratch) in self.chunks.iter_mut().zip(&mut self.chunk_scratch) {
                bind_dense_evals_reuse(chunk, scratch, challenge);
            }
        }
    }

    fn factor_eval(&self, index: usize, relation: Stage6Relation) -> Result<F, Stage6KernelError> {
        self.chunks
            .get(index)
            .and_then(|values| values.first())
            .copied()
            .ok_or(Stage6KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "empty booleanity factor",
            })
    }

    fn final_relation_eval(&self, relation: Stage6Relation) -> Result<F, Stage6KernelError> {
        let eq = self
            .eq
            .first()
            .copied()
            .ok_or(Stage6KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "empty booleanity eq factor",
            })?;
        let mut booleanity = F::zero();
        for (index, &gamma_power) in self.gamma_powers.iter().enumerate() {
            let ra = self.factor_eval(index, relation)?;
            booleanity += gamma_power * (ra.square() - ra);
        }
        Ok(eq * booleanity)
    }

    fn final_evals(
        &self,
        relation: Stage6Relation,
    ) -> Result<Vec<Stage6NamedEval<F>>, Stage6KernelError> {
        self.outputs
            .iter()
            .map(|output| {
                let factor =
                    output
                        .factor
                        .checked_sub(1)
                        .ok_or(Stage6KernelError::InvalidProof {
                            driver: relation.symbol(),
                            reason: "booleanity output factor underflow",
                        })?;
                Ok(named_eval(
                    output.name,
                    output.oracle,
                    self.factor_eval(factor, relation)?,
                ))
            })
            .collect()
    }
}

struct DenseStage6State<F: Field> {
    factors: Vec<Vec<F>>,
    factor_scratch: Vec<Vec<F>>,
    terms: Vec<DenseTerm<F>>,
    outputs: Vec<FactorOutput>,
    active_scale: F,
    degree_bound: usize,
}

#[derive(Clone)]
struct DenseTerm<F: Field> {
    coefficient: F,
    factors: Vec<usize>,
}

#[derive(Clone, Copy)]
struct FactorOutput {
    name: &'static str,
    oracle: &'static str,
    factor: usize,
}

impl<F: Field> DenseStage6State<F> {
    fn new(
        factors: Vec<Vec<F>>,
        terms: Vec<DenseTerm<F>>,
        outputs: Vec<FactorOutput>,
        active_scale: F,
        degree_bound: usize,
    ) -> Self {
        let factor_scratch = (0..factors.len()).map(|_| Vec::new()).collect();
        Self {
            factors,
            factor_scratch,
            terms,
            outputs,
            active_scale,
            degree_bound,
        }
    }

    fn round_poly(
        &self,
        previous_claim: F,
        relation: Stage6Relation,
    ) -> Result<UnivariatePoly<F>, Stage6KernelError> {
        let first_len = self.factors.first().map_or(0, Vec::len);
        if first_len == 0 || !first_len.is_power_of_two() {
            return Err(Stage6KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "stage6 dense factor has invalid length",
            });
        }
        if self.factors.iter().any(|factor| factor.len() != first_len) {
            return Err(Stage6KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "stage6 dense factors have inconsistent lengths",
            });
        }
        let poly = round_poly_from_dense_terms(
            &self.factors,
            &self.terms,
            self.active_scale,
            self.degree_bound,
            relation,
        )?;
        if poly.evaluate(F::zero()) + poly.evaluate(F::one()) != previous_claim {
            return Err(Stage6KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "stage6 relation input claim mismatch",
            });
        }
        Ok(poly)
    }

    fn bind(&mut self, challenge: F) {
        if self.factors.first().map_or(0, Vec::len) / 2 >= DENSE_BIND_PAR_THRESHOLD {
            self.factors
                .par_iter_mut()
                .zip(self.factor_scratch.par_iter_mut())
                .for_each(|(factor, scratch)| {
                    bind_dense_evals_reuse(factor, scratch, challenge);
                });
        } else {
            for (factor, scratch) in self.factors.iter_mut().zip(&mut self.factor_scratch) {
                bind_dense_evals_reuse(factor, scratch, challenge);
            }
        }
    }

    fn factor_eval(&self, index: usize, relation: Stage6Relation) -> Result<F, Stage6KernelError> {
        self.factors
            .get(index)
            .and_then(|values| values.first())
            .copied()
            .ok_or(Stage6KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "empty stage6 factor",
            })
    }

    fn final_relation_eval(&self, relation: Stage6Relation) -> Result<F, Stage6KernelError> {
        let mut result = F::zero();
        for term in &self.terms {
            let mut value = term.coefficient;
            for &factor in &term.factors {
                value *= self.factor_eval(factor, relation)?;
            }
            result += value;
        }
        Ok(result)
    }

    fn final_evals(
        &self,
        relation: Stage6Relation,
    ) -> Result<Vec<Stage6NamedEval<F>>, Stage6KernelError> {
        self.outputs
            .iter()
            .map(|output| {
                Ok(named_eval(
                    output.name,
                    output.oracle,
                    self.factor_eval(output.factor, relation)?,
                ))
            })
            .collect()
    }
}

fn bytecode_read_raf_state<F: Field>(
    program: &'static Stage6CpuProgramPlan,
    claim: &Stage6SumcheckClaimPlan,
    inputs: &Stage6ProverInputs<'_, F>,
    store: &Stage6ValueStore<F>,
    active_scale: F,
) -> Result<Stage6ProverInstanceState<F>, Stage6KernelError> {
    let witness = inputs
        .bytecode_read_raf
        .ok_or(Stage6KernelError::MissingKernelInput {
            kernel: "jolt_stage6_batched",
            input: "bytecode_read_raf",
        })?;
    if witness.bytecode_ra_chunks.is_empty() {
        return Err(Stage6KernelError::InvalidInputLength {
            input: "stage6.bytecode_read_raf.BytecodeRa",
            expected: 1,
            actual: 0,
        });
    }

    let log_t = stage6_trace_rounds(program)?;
    let log_k =
        claim
            .num_rounds
            .checked_sub(log_t)
            .ok_or(Stage6KernelError::InvalidInputLength {
                input: "stage6.bytecode_read_raf.input",
                expected: log_t,
                actual: claim.num_rounds,
            })?;
    let domain_len = 1usize.checked_shl(claim.num_rounds as u32).ok_or(
        Stage6KernelError::InvalidInputLength {
            input: "stage6.bytecode_read_raf.domain",
            expected: usize::BITS as usize,
            actual: claim.num_rounds,
        },
    )?;
    let expected_entries =
        1usize
            .checked_shl(log_k as u32)
            .ok_or(Stage6KernelError::InvalidInputLength {
                input: "stage6.bytecode_read_raf.entries",
                expected: usize::BITS as usize,
                actual: log_k,
            })?;
    require_operand_count(
        "stage6.bytecode_read_raf.entries",
        expected_entries,
        witness.data.entries.len(),
    )?;
    if witness.data.entry_bytecode_index >= expected_entries {
        return Err(Stage6KernelError::InvalidInputLength {
            input: "stage6.bytecode_read_raf.entry_bytecode_index",
            expected: expected_entries,
            actual: witness.data.entry_bytecode_index + 1,
        });
    }

    let mut chunk_lens = Vec::with_capacity(witness.bytecode_ra_chunks.len());
    for chunk in witness.bytecode_ra_chunks {
        let rounds = log2_exact(chunk.len(), "stage6.bytecode_read_raf.BytecodeRa")?;
        let chunk_len = rounds
            .checked_sub(log_t)
            .ok_or(Stage6KernelError::InvalidInputLength {
                input: "stage6.bytecode_read_raf.BytecodeRa",
                expected: log_t,
                actual: rounds,
            })?;
        chunk_lens.push(chunk_len);
    }
    let covered_address_len = chunk_lens.iter().sum::<usize>();
    require_operand_count(
        "stage6.bytecode_read_raf.address_chunks",
        log_k,
        covered_address_len,
    )?;

    if let Some(bytecode_cycle_indices) =
        bytecode_cycle_indices_from_one_hot(witness.bytecode_ra_chunks, &chunk_lens, log_t)
    {
        let outputs = bytecode_read_raf_output_plans(program, witness.bytecode_ra_chunks.len())?;
        return BytecodeReadRafStage6State::new(
            witness.data,
            witness.bytecode_ra_chunks,
            bytecode_cycle_indices,
            chunk_lens,
            store,
            log_k,
            log_t,
            active_scale,
            claim.degree,
            outputs,
        )
        .map(Stage6ProverInstanceState::BytecodeReadRaf);
    }

    bytecode_read_raf_dense_state(
        program,
        claim,
        witness,
        store,
        active_scale,
        log_k,
        log_t,
        domain_len,
        chunk_lens,
    )
    .map(Stage6ProverInstanceState::Dense)
}

#[allow(clippy::too_many_arguments)]
fn bytecode_read_raf_dense_state<F: Field>(
    program: &'static Stage6CpuProgramPlan,
    claim: &Stage6SumcheckClaimPlan,
    witness: Stage6BytecodeReadRafWitness<'_, F>,
    store: &Stage6ValueStore<F>,
    active_scale: F,
    log_k: usize,
    log_t: usize,
    domain_len: usize,
    chunk_lens: Vec<usize>,
) -> Result<DenseStage6State<F>, Stage6KernelError> {
    let mut factors = Vec::with_capacity(witness.bytecode_ra_chunks.len() + 1);
    factors.push(bytecode_weighted_value_factor(
        witness.data,
        store,
        log_k,
        log_t,
        domain_len,
    )?);
    factors.extend(expanded_bytecode_ra_factors(
        witness.bytecode_ra_chunks,
        &chunk_lens,
        log_k,
        log_t,
        domain_len,
    )?);
    let term_factors = (0..factors.len()).collect::<Vec<_>>();
    let outputs = bytecode_read_raf_output_plans(program, witness.bytecode_ra_chunks.len())?;

    Ok(DenseStage6State::new(
        factors,
        vec![DenseTerm {
            coefficient: F::one(),
            factors: term_factors,
        }],
        outputs,
        active_scale,
        claim.degree,
    ))
}

fn booleanity_state<F: Field>(
    program: &'static Stage6CpuProgramPlan,
    claim: &Stage6SumcheckClaimPlan,
    inputs: &Stage6ProverInputs<'_, F>,
    store: &Stage6ValueStore<F>,
    active_scale: F,
) -> Result<BooleanityStage6State<F>, Stage6KernelError> {
    let witness = inputs
        .booleanity
        .ok_or(Stage6KernelError::MissingKernelInput {
            kernel: "jolt_stage6_batched",
            input: "booleanity",
        })?;
    if witness.chunks.is_empty() {
        return Err(Stage6KernelError::InvalidInputLength {
            input: "stage6.booleanity.Ra",
            expected: 1,
            actual: 0,
        });
    }
    let domain_len = witness.chunks[0].len();
    let booleanity_rounds = log2_exact(domain_len, "stage6.booleanity.trace_len")?;
    require_operand_count(
        "stage6.booleanity.input",
        booleanity_rounds,
        claim.num_rounds,
    )?;
    for chunk in witness.chunks {
        require_operand_count("stage6.booleanity.Ra", domain_len, chunk.len())?;
    }

    let log_t = stage6_trace_rounds(program)?;
    let log_k_chunk =
        booleanity_rounds
            .checked_sub(log_t)
            .ok_or(Stage6KernelError::InvalidInputLength {
                input: "stage6.booleanity.trace_len",
                expected: log_t,
                actual: booleanity_rounds,
            })?;
    let stage5_point = store.point("stage6.input.stage5.instruction_read_raf.InstructionRa_0")?;
    let stage5_address_len =
        stage5_point
            .len()
            .checked_sub(log_t)
            .ok_or(Stage6KernelError::InvalidInputLength {
                input: "stage6.input.stage5.instruction_read_raf.InstructionRa_0",
                expected: log_t,
                actual: stage5_point.len(),
            })?;
    if stage5_address_len < log_k_chunk {
        return Err(Stage6KernelError::InvalidInputLength {
            input: "stage6.input.stage5.instruction_read_raf.InstructionRa_0",
            expected: log_k_chunk + log_t,
            actual: stage5_point.len(),
        });
    }

    let mut stage5_addr = stage5_point[..stage5_address_len].to_vec();
    stage5_addr.reverse();
    let mut combined_r = stage5_addr[stage5_address_len - log_k_chunk..].to_vec();
    combined_r.extend(stage5_point[stage5_address_len..].iter().rev().copied());
    require_operand_count(
        "stage6.booleanity.combined_point",
        booleanity_rounds,
        combined_r.len(),
    )?;

    let eq_point = reverse_slice(&combined_r);
    let eq = EqPolynomial::<F>::evals(&eq_point, None);
    require_operand_count("stage6.booleanity.eq", domain_len, eq.len())?;

    let gamma = store.scalar("stage6.booleanity.gamma")?;
    let gamma_sq = gamma.square();
    let mut gamma_power = F::one();
    let mut gamma_powers = Vec::with_capacity(witness.chunks.len());
    for _ in 0..witness.chunks.len() {
        gamma_powers.push(gamma_power);
        gamma_power *= gamma_sq;
    }

    BooleanityStage6State::new(
        eq,
        witness
            .chunks
            .iter()
            .map(|chunk| (*chunk).to_vec())
            .collect(),
        gamma_powers,
        booleanity_output_plans(program, witness.chunks.len())?,
        active_scale,
        claim.degree,
    )
}

fn hamming_booleanity_state<F: Field>(
    program: &'static Stage6CpuProgramPlan,
    claim: &Stage6SumcheckClaimPlan,
    inputs: &Stage6ProverInputs<'_, F>,
    store: &Stage6ValueStore<F>,
    active_scale: F,
) -> Result<DenseStage6State<F>, Stage6KernelError> {
    let witness = inputs
        .hamming_booleanity
        .ok_or(Stage6KernelError::MissingKernelInput {
            kernel: "jolt_stage6_batched",
            input: "hamming_booleanity",
        })?;
    let trace_rounds = log2_exact(
        witness.hamming_weight.len(),
        "stage6.hamming_booleanity.trace_len",
    )?;
    require_operand_count(
        "stage6.hamming_booleanity.input",
        trace_rounds,
        claim.num_rounds,
    )?;
    let lookup_output_point = store.point("stage6.input.stage1.LookupOutput")?.to_vec();
    require_operand_count(
        "stage6.input.stage1.LookupOutput",
        trace_rounds,
        lookup_output_point.len(),
    )?;
    let eq_lookup_output = EqPolynomial::<F>::evals(&lookup_output_point, None);
    require_operand_count(
        "stage6.hamming_booleanity.eq",
        witness.hamming_weight.len(),
        eq_lookup_output.len(),
    )?;
    let output = program
        .evals
        .iter()
        .find(|eval| eval.name == "stage6.hamming_booleanity.eval.HammingWeight")
        .map(|eval| FactorOutput {
            name: eval.name,
            oracle: eval.oracle,
            factor: 1,
        })
        .ok_or(Stage6KernelError::MissingValue {
            symbol: "stage6.hamming_booleanity.eval.HammingWeight",
        })?;

    Ok(DenseStage6State::new(
        vec![eq_lookup_output, witness.hamming_weight.to_vec()],
        vec![
            DenseTerm {
                coefficient: F::one(),
                factors: vec![0, 1, 1],
            },
            DenseTerm {
                coefficient: -F::one(),
                factors: vec![0, 1],
            },
        ],
        vec![output],
        active_scale,
        claim.degree,
    ))
}

fn inc_claim_reduction_state<F: Field>(
    program: &'static Stage6CpuProgramPlan,
    claim: &Stage6SumcheckClaimPlan,
    inputs: &Stage6ProverInputs<'_, F>,
    store: &Stage6ValueStore<F>,
    active_scale: F,
) -> Result<DenseStage6State<F>, Stage6KernelError> {
    let witness = inputs
        .inc_claim_reduction
        .ok_or(Stage6KernelError::MissingKernelInput {
            kernel: "jolt_stage6_batched",
            input: "inc_claim_reduction",
        })?;
    let trace_rounds = log2_exact(
        witness.ram_inc.len(),
        "stage6.inc_claim_reduction.trace_len",
    )?;
    require_operand_count(
        "stage6.inc_claim_reduction.RdInc",
        witness.ram_inc.len(),
        witness.rd_inc.len(),
    )?;
    require_operand_count(
        "stage6.inc_claim_reduction.input",
        trace_rounds,
        claim.num_rounds,
    )?;

    let ram_inc_stage2 = suffix_point(
        store.point("stage6.input.stage2.ram_read_write.RamInc")?,
        trace_rounds,
        "stage6.input.stage2.ram_read_write.RamInc",
    )?;
    let ram_inc_stage4 = suffix_point(
        store.point("stage6.input.stage4.ram_val_check.RamInc")?,
        trace_rounds,
        "stage6.input.stage4.ram_val_check.RamInc",
    )?;
    let rd_inc_stage4 = suffix_point(
        store.point("stage6.input.stage4.registers_read_write.RdInc")?,
        trace_rounds,
        "stage6.input.stage4.registers_read_write.RdInc",
    )?;
    let rd_inc_stage5 = suffix_point(
        store.point("stage6.input.stage5.registers_val_evaluation.RdInc")?,
        trace_rounds,
        "stage6.input.stage5.registers_val_evaluation.RdInc",
    )?;
    let gamma = store.scalar("stage6.inc_claim_reduction.gamma")?;
    let gamma2 = gamma.square();

    let mut eq_ram_combined = EqPolynomial::<F>::evals(ram_inc_stage2, None);
    let eq_ram_stage4 = EqPolynomial::<F>::evals(ram_inc_stage4, None);
    let mut eq_rd_combined = EqPolynomial::<F>::evals(rd_inc_stage4, None);
    let eq_rd_stage5 = EqPolynomial::<F>::evals(rd_inc_stage5, None);
    require_operand_count(
        "stage6.inc_claim_reduction.eq_ram",
        witness.ram_inc.len(),
        eq_ram_combined.len(),
    )?;
    require_operand_count(
        "stage6.inc_claim_reduction.eq_rd",
        witness.rd_inc.len(),
        eq_rd_combined.len(),
    )?;
    for (combined, stage4) in eq_ram_combined.iter_mut().zip(eq_ram_stage4) {
        *combined += gamma * stage4;
    }
    for (combined, stage5) in eq_rd_combined.iter_mut().zip(eq_rd_stage5) {
        *combined += gamma * stage5;
    }

    Ok(DenseStage6State::new(
        vec![
            eq_ram_combined,
            witness.ram_inc.to_vec(),
            eq_rd_combined,
            witness.rd_inc.to_vec(),
        ],
        vec![
            DenseTerm {
                coefficient: F::one(),
                factors: vec![0, 1],
            },
            DenseTerm {
                coefficient: gamma2,
                factors: vec![2, 3],
            },
        ],
        vec![
            factor_output_by_name(program, "stage6.inc_claim_reduction.eval.RamInc", 1)?,
            factor_output_by_name(program, "stage6.inc_claim_reduction.eval.RdInc", 3)?,
        ],
        active_scale,
        claim.degree,
    ))
}

fn ram_ra_virtual_state<F: Field>(
    program: &'static Stage6CpuProgramPlan,
    claim: &Stage6SumcheckClaimPlan,
    inputs: &Stage6ProverInputs<'_, F>,
    store: &Stage6ValueStore<F>,
    active_scale: F,
) -> Result<DenseStage6State<F>, Stage6KernelError> {
    let witness = inputs
        .ram_ra_virtual
        .ok_or(Stage6KernelError::MissingKernelInput {
            kernel: "jolt_stage6_batched",
            input: "ram_ra_virtual",
        })?;
    if witness.ram_ra_chunks.is_empty() {
        return Err(Stage6KernelError::InvalidInputLength {
            input: "stage6.ram_ra_virtual.RamRa",
            expected: 1,
            actual: 0,
        });
    }
    let trace_len = witness.ram_ra_chunks[0].len();
    let trace_rounds = log2_exact(trace_len, "stage6.ram_ra_virtual.trace_len")?;
    require_operand_count(
        "stage6.ram_ra_virtual.input",
        trace_rounds,
        claim.num_rounds,
    )?;
    for chunk in witness.ram_ra_chunks {
        require_operand_count("stage6.ram_ra_virtual.RamRa", trace_len, chunk.len())?;
    }

    let input_point = store.point("stage6.input.stage5.ram_ra_claim_reduction.RamRa")?;
    let r_cycle = suffix_point(
        input_point,
        trace_rounds,
        "stage6.input.stage5.ram_ra_claim_reduction.RamRa",
    )?;
    let eq_cycle = EqPolynomial::<F>::evals(r_cycle, None);
    require_operand_count("stage6.ram_ra_virtual.eq", trace_len, eq_cycle.len())?;

    let mut factors = Vec::with_capacity(witness.ram_ra_chunks.len() + 1);
    factors.push(eq_cycle);
    factors.extend(witness.ram_ra_chunks.iter().map(|chunk| (*chunk).to_vec()));
    let term_factors = (0..factors.len()).collect::<Vec<_>>();
    let outputs = ram_ra_virtual_output_plans(program, witness.ram_ra_chunks.len())?;

    Ok(DenseStage6State::new(
        factors,
        vec![DenseTerm {
            coefficient: F::one(),
            factors: term_factors,
        }],
        outputs,
        active_scale,
        claim.degree,
    ))
}

fn instruction_ra_virtual_state<F: Field>(
    program: &'static Stage6CpuProgramPlan,
    claim: &Stage6SumcheckClaimPlan,
    inputs: &Stage6ProverInputs<'_, F>,
    store: &Stage6ValueStore<F>,
    active_scale: F,
) -> Result<DenseStage6State<F>, Stage6KernelError> {
    let witness = inputs
        .instruction_ra_virtual
        .ok_or(Stage6KernelError::MissingKernelInput {
            kernel: "jolt_stage6_batched",
            input: "instruction_ra_virtual",
        })?;
    if witness.instruction_ra_chunks.is_empty()
        || witness.virtual_count == 0
        || witness.instruction_ra_chunks.len() % witness.virtual_count != 0
    {
        return Err(Stage6KernelError::InvalidInputLength {
            input: "stage6.instruction_ra_virtual.InstructionRa",
            expected: witness.virtual_count,
            actual: witness.instruction_ra_chunks.len(),
        });
    }
    let trace_len = witness.instruction_ra_chunks[0].len();
    let trace_rounds = log2_exact(trace_len, "stage6.instruction_ra_virtual.trace_len")?;
    require_operand_count(
        "stage6.instruction_ra_virtual.input",
        trace_rounds,
        claim.num_rounds,
    )?;
    for chunk in witness.instruction_ra_chunks {
        require_operand_count(
            "stage6.instruction_ra_virtual.InstructionRa",
            trace_len,
            chunk.len(),
        )?;
    }

    let input_point = store.point("stage6.input.stage5.instruction_read_raf.InstructionRa_0")?;
    let r_cycle = suffix_point(
        input_point,
        trace_rounds,
        "stage6.input.stage5.instruction_read_raf.InstructionRa_0",
    )?;
    let eq_cycle = EqPolynomial::<F>::evals(r_cycle, None);
    require_operand_count(
        "stage6.instruction_ra_virtual.eq",
        trace_len,
        eq_cycle.len(),
    )?;

    let mut factors = Vec::with_capacity(witness.instruction_ra_chunks.len() + 1);
    factors.push(eq_cycle);
    factors.extend(
        witness
            .instruction_ra_chunks
            .iter()
            .map(|chunk| (*chunk).to_vec()),
    );

    let chunks_per_virtual = witness.instruction_ra_chunks.len() / witness.virtual_count;
    let gamma = store.scalar("stage6.instruction_ra_virtual.gamma")?;
    let mut gamma_power = F::one();
    let mut terms = Vec::with_capacity(witness.virtual_count);
    for virtual_index in 0..witness.virtual_count {
        let start = 1 + virtual_index * chunks_per_virtual;
        let end = start + chunks_per_virtual;
        let mut factors = Vec::with_capacity(chunks_per_virtual + 1);
        factors.push(0);
        factors.extend(start..end);
        terms.push(DenseTerm {
            coefficient: gamma_power,
            factors,
        });
        gamma_power *= gamma;
    }
    let outputs =
        instruction_ra_virtual_output_plans(program, witness.instruction_ra_chunks.len())?;

    Ok(DenseStage6State::new(
        factors,
        terms,
        outputs,
        active_scale,
        claim.degree,
    ))
}

fn evaluate_stage6_field_expr<F: Field>(
    expr: &Stage6FieldExprPlan,
    operands: &[F],
) -> Result<F, Stage6KernelError> {
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
                    Stage6KernelError::UnsupportedFieldExpr {
                        symbol: expr.symbol,
                        formula,
                    }
                })?;
                return Ok(pow_field(operands[0], exponent));
            }
            Err(Stage6KernelError::UnsupportedFieldExpr {
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

fn single_operand<F: Field>(symbol: &'static str, operands: &[F]) -> Result<F, Stage6KernelError> {
    require_operand_count(symbol, 1, operands.len())?;
    Ok(operands[0])
}

fn require_operand_count(
    input: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), Stage6KernelError> {
    if expected == actual {
        Ok(())
    } else {
        Err(Stage6KernelError::InvalidInputLength {
            input,
            expected,
            actual,
        })
    }
}

fn append_opening_claims<F, T>(
    program: &'static Stage6CpuProgramPlan,
    store: &mut Stage6ValueStore<F>,
    transcript: &mut T,
    evals: &[Stage6NamedEval<F>],
) -> Result<(), Stage6KernelError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    if program.opening_batches.is_empty() {
        for eval in evals {
            append_labeled_scalar(transcript, "opening_claim", &eval.value);
        }
        return Ok(());
    }
    let _ = store.evaluate_available_points(program)?;
    let mut seen = program
        .opening_inputs
        .iter()
        .filter_map(|input| {
            store
                .try_point(input.symbol)
                .map(|point| (input.claim_kind, input.oracle, point.to_vec()))
        })
        .collect::<Vec<_>>();
    for batch in program.opening_batches {
        for symbol in batch.claim_operands {
            let claim =
                find_opening_claim(program, symbol).ok_or(Stage6KernelError::MissingClaim {
                    batch: batch.symbol,
                    claim: symbol,
                })?;
            let point = store.point(claim.point_source)?.to_vec();
            if seen.iter().any(|(kind, oracle, seen_point)| {
                *kind == claim.claim_kind && *oracle == claim.oracle && seen_point == &point
            }) {
                continue;
            }
            let value = store.scalar(claim.eval_source)?;
            append_labeled_scalar(transcript, "opening_claim", &value);
            seen.push((claim.claim_kind, claim.oracle, point));
        }
    }
    Ok(())
}

fn find_opening_claim<'a>(
    program: &'a Stage6CpuProgramPlan,
    symbol: &str,
) -> Option<&'a Stage6OpeningClaimPlan> {
    program
        .opening_claims
        .iter()
        .find(|claim| claim.symbol == symbol)
}

fn stage6_trace_rounds(program: &'static Stage6CpuProgramPlan) -> Result<usize, Stage6KernelError> {
    program
        .instance_results
        .iter()
        .find(|instance| instance.relation == "jolt.stage6.hamming_booleanity")
        .map(|instance| instance.num_rounds)
        .ok_or(Stage6KernelError::MissingValue {
            symbol: "stage6.hamming_booleanity.instance",
        })
}

fn bytecode_gamma_powers<F: Field>(gamma: F) -> [F; 8] {
    let mut powers = [F::one(); 8];
    for index in 1..powers.len() {
        powers[index] = powers[index - 1] * gamma;
    }
    powers
}

fn bytecode_stage_cycle_points<F: Field>(
    store: &Stage6ValueStore<F>,
    log_t: usize,
) -> Result<[Vec<F>; 5], Stage6KernelError> {
    Ok([
        suffix_point(
            store.point("stage6.input.stage1.Imm")?,
            log_t,
            "stage6.input.stage1.Imm",
        )?
        .to_vec(),
        suffix_point(
            store.point("stage6.input.stage2.OpFlagJump")?,
            log_t,
            "stage6.input.stage2.OpFlagJump",
        )?
        .to_vec(),
        suffix_point(
            store.point("stage6.input.stage3.spartan_shift.UnexpandedPC")?,
            log_t,
            "stage6.input.stage3.spartan_shift.UnexpandedPC",
        )?
        .to_vec(),
        suffix_point(
            store.point("stage6.input.stage4.Rs1Ra")?,
            log_t,
            "stage6.input.stage4.Rs1Ra",
        )?
        .to_vec(),
        suffix_point(
            store.point("stage6.input.stage5.registers_val_evaluation.RdWa")?,
            log_t,
            "stage6.input.stage5.registers_val_evaluation.RdWa",
        )?
        .to_vec(),
    ])
}

fn bytecode_stage_value_evals<F: Field>(
    data: Stage6BytecodeReadRafData<'_, F>,
    store: &Stage6ValueStore<F>,
    r_address: &[F],
    log_t: usize,
) -> Result<[F; 5], Stage6KernelError> {
    let expected_len = 1usize.checked_shl(r_address.len() as u32).ok_or(
        Stage6KernelError::InvalidInputLength {
            input: "stage6.bytecode_read_raf.entries",
            expected: usize::BITS as usize,
            actual: r_address.len(),
        },
    )?;
    require_operand_count(
        "stage6.bytecode_read_raf.entries",
        expected_len,
        data.entries.len(),
    )?;
    if data.entry_bytecode_index >= expected_len {
        return Err(Stage6KernelError::InvalidInputLength {
            input: "stage6.bytecode_read_raf.entry_bytecode_index",
            expected: expected_len,
            actual: data.entry_bytecode_index + 1,
        });
    }

    let stage1_gamma = store.scalar("stage6.bytecode_read_raf.stage1_gamma")?;
    let stage2_gamma = store.scalar("stage6.bytecode_read_raf.stage2_gamma")?;
    let stage3_gamma = store.scalar("stage6.bytecode_read_raf.stage3_gamma")?;
    let stage4_gamma = store.scalar("stage6.bytecode_read_raf.stage4_gamma")?;
    let stage5_gamma = store.scalar("stage6.bytecode_read_raf.stage5_gamma")?;
    let stage1_gamma_powers = field_powers(stage1_gamma, 16);
    let stage2_gamma_powers = field_powers(stage2_gamma, 4);
    let stage3_gamma_powers = field_powers(stage3_gamma, 9);
    let stage4_gamma_powers = field_powers(stage4_gamma, 3);
    let stage5_gamma_powers = field_powers(stage5_gamma, data.num_lookup_tables + 2);

    let stage4_register_point = register_prefix_point(store, "stage6.input.stage4.Rs1Ra", log_t)?;
    let stage5_register_point = register_prefix_point(
        store,
        "stage6.input.stage5.registers_val_evaluation.RdWa",
        log_t,
    )?;

    let mut evals = [F::zero(); 5];
    for (index, entry) in data.entries.iter().enumerate() {
        let eq = indexed_boolean_eq(index, r_address)?;
        let values = bytecode_entry_stage_values(
            entry,
            data.num_lookup_tables,
            stage4_register_point,
            stage5_register_point,
            &stage1_gamma_powers,
            &stage2_gamma_powers,
            &stage3_gamma_powers,
            &stage4_gamma_powers,
            &stage5_gamma_powers,
        )?;
        for stage in 0..evals.len() {
            evals[stage] += eq * values[stage];
        }
    }
    Ok(evals)
}

fn bytecode_weighted_value_factor<F: Field>(
    data: Stage6BytecodeReadRafData<'_, F>,
    store: &Stage6ValueStore<F>,
    log_k: usize,
    log_t: usize,
    domain_len: usize,
) -> Result<Vec<F>, Stage6KernelError> {
    let gamma = store.scalar("stage6.bytecode_read_raf.gamma")?;
    let gamma_powers = bytecode_gamma_powers(gamma);
    let stage_cycle_points = bytecode_stage_cycle_points(store, log_t)?;

    let stage1_gamma = store.scalar("stage6.bytecode_read_raf.stage1_gamma")?;
    let stage2_gamma = store.scalar("stage6.bytecode_read_raf.stage2_gamma")?;
    let stage3_gamma = store.scalar("stage6.bytecode_read_raf.stage3_gamma")?;
    let stage4_gamma = store.scalar("stage6.bytecode_read_raf.stage4_gamma")?;
    let stage5_gamma = store.scalar("stage6.bytecode_read_raf.stage5_gamma")?;
    let stage1_gamma_powers = field_powers(stage1_gamma, 16);
    let stage2_gamma_powers = field_powers(stage2_gamma, 4);
    let stage3_gamma_powers = field_powers(stage3_gamma, 9);
    let stage4_gamma_powers = field_powers(stage4_gamma, 3);
    let stage5_gamma_powers = field_powers(stage5_gamma, data.num_lookup_tables + 2);
    let stage4_register_point = register_prefix_point(store, "stage6.input.stage4.Rs1Ra", log_t)?;
    let stage5_register_point = register_prefix_point(
        store,
        "stage6.input.stage5.registers_val_evaluation.RdWa",
        log_t,
    )?;
    let stage_values = data
        .entries
        .iter()
        .map(|entry| {
            bytecode_entry_stage_values(
                entry,
                data.num_lookup_tables,
                stage4_register_point,
                stage5_register_point,
                &stage1_gamma_powers,
                &stage2_gamma_powers,
                &stage3_gamma_powers,
                &stage4_gamma_powers,
                &stage5_gamma_powers,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    (0..domain_len)
        .map(|row| {
            let (address_bits, cycle_bits) = normalized_bytecode_row_bits::<F>(row, log_k, log_t)?;
            let address_index = bits_to_index(&address_bits);
            let int_eval = identity_polynomial_eval(&address_bits);
            let int_contrib = [
                gamma_powers[5] * int_eval,
                F::zero(),
                gamma_powers[4] * int_eval,
                F::zero(),
                F::zero(),
            ];
            let mut value = F::zero();
            for stage in 0..stage_values[address_index].len() {
                value += (stage_values[address_index][stage] + int_contrib[stage])
                    * EqPolynomial::<F>::mle(&stage_cycle_points[stage], &cycle_bits)
                    * gamma_powers[stage];
            }
            if address_index == data.entry_bytecode_index
                && cycle_bits.iter().all(|bit| *bit == F::zero())
            {
                value += gamma_powers[7];
            }
            Ok(value)
        })
        .collect()
}

fn expanded_bytecode_ra_factors<F: Field>(
    chunks: &[&[F]],
    chunk_lens: &[usize],
    log_k: usize,
    log_t: usize,
    domain_len: usize,
) -> Result<Vec<Vec<F>>, Stage6KernelError> {
    let mut factors = Vec::with_capacity(chunks.len());
    let mut offset = 0usize;
    for (chunk, &chunk_len) in chunks.iter().zip(chunk_lens) {
        let factor = (0..domain_len)
            .map(|row| {
                let (address_bits, cycle_bits) =
                    normalized_bytecode_row_bits::<F>(row, log_k, log_t)?;
                let mut chunk_bits = address_bits[offset..offset + chunk_len].to_vec();
                chunk_bits.extend(cycle_bits);
                let index = bits_to_index(&chunk_bits);
                chunk
                    .get(index)
                    .copied()
                    .ok_or(Stage6KernelError::InvalidInputLength {
                        input: "stage6.bytecode_read_raf.BytecodeRa",
                        expected: index + 1,
                        actual: chunk.len(),
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        factors.push(factor);
        offset += chunk_len;
    }
    Ok(factors)
}

fn bytecode_cycle_indices_from_one_hot<F: Field>(
    chunks: &[&[F]],
    chunk_lens: &[usize],
    log_t: usize,
) -> Option<Vec<usize>> {
    let trace_len = 1usize.checked_shl(log_t as u32)?;
    let mut indices = vec![0usize; trace_len];
    let mut remaining_address_bits = chunk_lens.iter().sum::<usize>();
    for (chunk, &chunk_len) in chunks.iter().zip(chunk_lens) {
        remaining_address_bits = remaining_address_bits.checked_sub(chunk_len)?;
        let chunk_domain = 1usize.checked_shl(chunk_len as u32)?;
        if chunk.len() != chunk_domain.checked_mul(trace_len)? {
            return None;
        }
        for cycle in 0..trace_len {
            let mut selected = None;
            for chunk_value in 0..chunk_domain {
                let value = chunk[chunk_value * trace_len + cycle];
                if value == F::one() {
                    if selected.replace(chunk_value).is_some() {
                        return None;
                    }
                } else if value != F::zero() {
                    return None;
                }
            }
            let selected = selected?;
            indices[cycle] |= selected << remaining_address_bits;
        }
    }
    Some(indices)
}

fn normalized_bytecode_row_bits<F: Field>(
    row: usize,
    log_k: usize,
    log_t: usize,
) -> Result<(Vec<F>, Vec<F>), Stage6KernelError> {
    let mut raw_bits = index_bits(row, log_k + log_t)?;
    raw_bits.reverse();
    let mut cycle_bits = raw_bits.split_off(log_k);
    raw_bits.reverse();
    cycle_bits.reverse();
    Ok((raw_bits, cycle_bits))
}

fn bytecode_entry_stage_values<F: Field>(
    entry: &Stage6BytecodeEntry<F>,
    num_lookup_tables: usize,
    stage4_register_point: &[F],
    stage5_register_point: &[F],
    stage1_gamma_powers: &[F],
    stage2_gamma_powers: &[F],
    stage3_gamma_powers: &[F],
    stage4_gamma_powers: &[F],
    stage5_gamma_powers: &[F],
) -> Result<[F; 5], Stage6KernelError> {
    let mut stage1 = entry.address + entry.imm * stage1_gamma_powers[1];
    for (flag, gamma) in entry
        .circuit_flags
        .iter()
        .zip(stage1_gamma_powers.iter().skip(2))
    {
        if *flag {
            stage1 += *gamma;
        }
    }

    let mut stage2 = F::zero();
    if entry.circuit_flags[5] {
        stage2 += stage2_gamma_powers[0];
    }
    if entry.is_branch {
        stage2 += stage2_gamma_powers[1];
    }
    if entry.circuit_flags[6] {
        stage2 += stage2_gamma_powers[2];
    }
    if entry.circuit_flags[7] {
        stage2 += stage2_gamma_powers[3];
    }

    let mut stage3 = entry.imm + entry.address * stage3_gamma_powers[1];
    if entry.left_is_rs1 {
        stage3 += stage3_gamma_powers[2];
    }
    if entry.left_is_pc {
        stage3 += stage3_gamma_powers[3];
    }
    if entry.right_is_rs2 {
        stage3 += stage3_gamma_powers[4];
    }
    if entry.right_is_imm {
        stage3 += stage3_gamma_powers[5];
    }
    if entry.is_noop {
        stage3 += stage3_gamma_powers[6];
    }
    if entry.circuit_flags[7] {
        stage3 += stage3_gamma_powers[7];
    }
    if entry.circuit_flags[12] {
        stage3 += stage3_gamma_powers[8];
    }

    let stage4 = register_eq(entry.rd, stage4_register_point, "stage6.bytecode.entry.rd")?
        * stage4_gamma_powers[0]
        + register_eq(
            entry.rs1,
            stage4_register_point,
            "stage6.bytecode.entry.rs1",
        )? * stage4_gamma_powers[1]
        + register_eq(
            entry.rs2,
            stage4_register_point,
            "stage6.bytecode.entry.rs2",
        )? * stage4_gamma_powers[2];

    let mut stage5 = register_eq(entry.rd, stage5_register_point, "stage6.bytecode.entry.rd")?
        * stage5_gamma_powers[0];
    if !entry.is_interleaved {
        stage5 += stage5_gamma_powers[1];
    }
    if let Some(table) = entry.lookup_table {
        if table >= num_lookup_tables {
            return Err(Stage6KernelError::InvalidInputLength {
                input: "stage6.bytecode.entry.lookup_table",
                expected: num_lookup_tables,
                actual: table + 1,
            });
        }
        stage5 += stage5_gamma_powers[2 + table];
    }

    Ok([stage1, stage2, stage3, stage4, stage5])
}

fn register_eq<F: Field>(
    index: Option<usize>,
    point: &[F],
    input: &'static str,
) -> Result<F, Stage6KernelError> {
    let Some(index) = index else {
        return Ok(F::zero());
    };
    let register_count =
        1usize
            .checked_shl(point.len() as u32)
            .ok_or(Stage6KernelError::InvalidInputLength {
                input,
                expected: usize::BITS as usize,
                actual: point.len(),
            })?;
    if index >= register_count {
        return Err(Stage6KernelError::InvalidInputLength {
            input,
            expected: register_count,
            actual: index + 1,
        });
    }
    indexed_boolean_eq(index, point)
}

fn indexed_boolean_eq<F: Field>(index: usize, point: &[F]) -> Result<F, Stage6KernelError> {
    let bits = index_bits(index, point.len())?;
    Ok(EqPolynomial::<F>::mle(&bits, point))
}

fn index_bits<F: Field>(index: usize, len: usize) -> Result<Vec<F>, Stage6KernelError> {
    if len >= usize::BITS as usize {
        return Err(Stage6KernelError::InvalidInputLength {
            input: "stage6.index_bits",
            expected: usize::BITS as usize - 1,
            actual: len,
        });
    }
    let limit = 1usize << len;
    if index >= limit {
        return Err(Stage6KernelError::InvalidInputLength {
            input: "stage6.index_bits.index",
            expected: limit,
            actual: index + 1,
        });
    }
    Ok((0..len)
        .map(|bit| F::from_u64(((index >> (len - 1 - bit)) & 1) as u64))
        .collect())
}

fn bits_to_index<F: Field>(bits: &[F]) -> usize {
    bits.iter().fold(0usize, |index, bit| {
        let bit = if *bit == F::zero() { 0 } else { 1 };
        (index << 1) | bit
    })
}

fn field_powers<F: Field>(base: F, count: usize) -> Vec<F> {
    let mut powers = Vec::with_capacity(count);
    let mut power = F::one();
    for _ in 0..count {
        powers.push(power);
        power *= base;
    }
    powers
}

fn identity_polynomial_eval<F: Field>(point: &[F]) -> F {
    point
        .iter()
        .enumerate()
        .map(|(index, value)| value.mul_pow_2(point.len() - 1 - index))
        .sum()
}

fn register_prefix_point<'a, F: Field>(
    store: &'a Stage6ValueStore<F>,
    symbol: &'static str,
    log_t: usize,
) -> Result<&'a [F], Stage6KernelError> {
    let point = store.point(symbol)?;
    let register_len =
        point
            .len()
            .checked_sub(log_t)
            .ok_or(Stage6KernelError::InvalidInputLength {
                input: symbol,
                expected: log_t,
                actual: point.len(),
            })?;
    prefix_point(point, register_len, symbol)
}

fn normalize_bytecode_read_raf_point<F: Field>(
    program: &'static Stage6CpuProgramPlan,
    point: &[F],
) -> Result<Vec<F>, Stage6KernelError> {
    let log_t = stage6_trace_rounds(program)?;
    let log_k = point
        .len()
        .checked_sub(log_t)
        .ok_or(Stage6KernelError::InvalidInputLength {
            input: "stage6.bytecode_read_raf.point",
            expected: log_t,
            actual: point.len(),
        })?;
    let mut normalized = point.to_vec();
    normalized[..log_k].reverse();
    normalized[log_k..].reverse();
    Ok(normalized)
}

fn normalize_instruction_read_raf_point<F: Field>(
    point: &[F],
) -> Result<Vec<F>, Stage6KernelError> {
    const LOG_K: usize = 128;
    if point.len() < LOG_K {
        return Err(Stage6KernelError::InvalidInputLength {
            input: "stage6.instruction_read_raf.point",
            expected: LOG_K,
            actual: point.len(),
        });
    }
    let mut normalized = point.to_vec();
    normalized[LOG_K..].reverse();
    Ok(normalized)
}

fn reverse_slice<F: Field>(values: &[F]) -> Vec<F> {
    values.iter().rev().copied().collect()
}

fn prefix_point<'a, F: Field>(
    point: &'a [F],
    length: usize,
    input: &'static str,
) -> Result<&'a [F], Stage6KernelError> {
    point
        .get(..length)
        .filter(|prefix| prefix.len() == length)
        .ok_or(Stage6KernelError::InvalidInputLength {
            input,
            expected: length,
            actual: point.len(),
        })
}

fn suffix_point<'a, F: Field>(
    point: &'a [F],
    length: usize,
    input: &'static str,
) -> Result<&'a [F], Stage6KernelError> {
    point
        .get(point.len().saturating_sub(length)..)
        .filter(|suffix| suffix.len() == length)
        .ok_or(Stage6KernelError::InvalidInputLength {
            input,
            expected: length,
            actual: point.len(),
        })
}

fn log2_exact(value: usize, input: &'static str) -> Result<usize, Stage6KernelError> {
    if value != 0 && value.is_power_of_two() {
        Ok(value.trailing_zeros() as usize)
    } else {
        Err(Stage6KernelError::InvalidInputLength {
            input,
            expected: value.next_power_of_two(),
            actual: value,
        })
    }
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

fn named_eval<F: Field>(name: &'static str, oracle: &'static str, value: F) -> Stage6NamedEval<F> {
    Stage6NamedEval {
        name,
        oracle,
        value,
    }
}

fn factor_output_by_name(
    program: &'static Stage6CpuProgramPlan,
    name: &'static str,
    factor: usize,
) -> Result<FactorOutput, Stage6KernelError> {
    program
        .evals
        .iter()
        .find(|eval| eval.name == name)
        .map(|eval| FactorOutput {
            name: eval.name,
            oracle: eval.oracle,
            factor,
        })
        .ok_or(Stage6KernelError::MissingValue { symbol: name })
}

fn ram_ra_virtual_output_plans(
    program: &'static Stage6CpuProgramPlan,
    chunk_count: usize,
) -> Result<Vec<FactorOutput>, Stage6KernelError> {
    indexed_output_plans_by_prefix(program, "stage6.ram_ra_virtual.eval.RamRa_", chunk_count, 1)
}

fn bytecode_read_raf_output_plans(
    program: &'static Stage6CpuProgramPlan,
    chunk_count: usize,
) -> Result<Vec<FactorOutput>, Stage6KernelError> {
    indexed_output_plans_by_prefix(
        program,
        "stage6.bytecode_read_raf.eval.BytecodeRa_",
        chunk_count,
        1,
    )
}

fn booleanity_output_plans(
    program: &'static Stage6CpuProgramPlan,
    chunk_count: usize,
) -> Result<Vec<FactorOutput>, Stage6KernelError> {
    let mut evals = program
        .evals
        .iter()
        .filter(|eval| {
            eval.name
                .starts_with("stage6.booleanity.eval.InstructionRa_")
                || eval.name.starts_with("stage6.booleanity.eval.BytecodeRa_")
                || eval.name.starts_with("stage6.booleanity.eval.RamRa_")
        })
        .collect::<Vec<_>>();
    evals.sort_by_key(|eval| eval.index);
    if evals.len() != chunk_count {
        return Err(Stage6KernelError::InvalidInputLength {
            input: "stage6.booleanity.eval",
            expected: chunk_count,
            actual: evals.len(),
        });
    }
    evals
        .into_iter()
        .enumerate()
        .map(|(index, eval)| {
            if eval.index != index {
                return Err(Stage6KernelError::InvalidProof {
                    driver: "stage6.booleanity.eval",
                    reason: "non-contiguous indexed eval",
                });
            }
            Ok(FactorOutput {
                name: eval.name,
                oracle: eval.oracle,
                factor: index + 1,
            })
        })
        .collect()
}

fn instruction_ra_virtual_output_plans(
    program: &'static Stage6CpuProgramPlan,
    chunk_count: usize,
) -> Result<Vec<FactorOutput>, Stage6KernelError> {
    indexed_output_plans_by_prefix(
        program,
        "stage6.instruction_ra_virtual.eval.InstructionRa_",
        chunk_count,
        1,
    )
}

fn indexed_output_plans_by_prefix(
    program: &'static Stage6CpuProgramPlan,
    prefix: &'static str,
    count: usize,
    first_factor: usize,
) -> Result<Vec<FactorOutput>, Stage6KernelError> {
    let mut outputs = vec![None; count];
    for eval in program.evals {
        let Some(suffix) = eval.name.strip_prefix(prefix) else {
            continue;
        };
        let index = suffix
            .parse::<usize>()
            .map_err(|_| Stage6KernelError::InvalidProof {
                driver: prefix,
                reason: "invalid indexed eval suffix",
            })?;
        if index >= count || outputs[index].is_some() {
            return Err(Stage6KernelError::InvalidProof {
                driver: prefix,
                reason: "invalid indexed eval",
            });
        }
        outputs[index] = Some(FactorOutput {
            name: eval.name,
            oracle: eval.oracle,
            factor: first_factor + index,
        });
    }
    outputs
        .into_iter()
        .map(|output| output.ok_or(Stage6KernelError::MissingValue { symbol: prefix }))
        .collect()
}

fn eval_by_name<F: Field>(
    evals: &[Stage6NamedEval<F>],
    name: &'static str,
) -> Result<F, Stage6KernelError> {
    evals
        .iter()
        .find(|eval| eval.name == name)
        .map(|eval| eval.value)
        .ok_or(Stage6KernelError::MissingValue { symbol: name })
}

fn indexed_evals_by_prefix_any<F: Field>(
    evals: &[Stage6NamedEval<F>],
    prefix: &'static str,
) -> Result<Vec<F>, Stage6KernelError> {
    let mut indexed_values = Vec::new();
    for eval in evals {
        let Some(suffix) = eval.name.strip_prefix(prefix) else {
            continue;
        };
        let index = suffix
            .parse::<usize>()
            .map_err(|_| Stage6KernelError::InvalidProof {
                driver: prefix,
                reason: "invalid indexed eval suffix",
            })?;
        if indexed_values
            .iter()
            .any(|(existing_index, _)| *existing_index == index)
        {
            return Err(Stage6KernelError::InvalidProof {
                driver: prefix,
                reason: "duplicate indexed eval",
            });
        }
        indexed_values.push((index, eval.value));
    }
    if indexed_values.is_empty() {
        return Err(Stage6KernelError::MissingValue { symbol: prefix });
    }
    indexed_values.sort_by_key(|(index, _)| *index);
    for (expected, (actual, _)) in indexed_values.iter().enumerate() {
        if *actual != expected {
            return Err(Stage6KernelError::InvalidProof {
                driver: prefix,
                reason: "non-contiguous indexed eval",
            });
        }
    }
    Ok(indexed_values.into_iter().map(|(_, value)| value).collect())
}

fn booleanity_evals<F: Field>(evals: &[Stage6NamedEval<F>]) -> Result<Vec<F>, Stage6KernelError> {
    let mut values = indexed_evals_by_prefix_any(evals, "stage6.booleanity.eval.InstructionRa_")?;
    values.extend(indexed_evals_by_prefix_any(
        evals,
        "stage6.booleanity.eval.BytecodeRa_",
    )?);
    values.extend(indexed_evals_by_prefix_any(
        evals,
        "stage6.booleanity.eval.RamRa_",
    )?);
    Ok(values)
}

fn claim_relation(
    program: &'static Stage6CpuProgramPlan,
    claim: &Stage6SumcheckClaimPlan,
) -> Result<Stage6Relation, Stage6KernelError> {
    if let Some(relation) = claim.relation {
        return Stage6Relation::from_symbol(relation)
            .ok_or(Stage6KernelError::UnknownRelation { relation });
    }
    let kernel_symbol = claim.kernel.ok_or(Stage6KernelError::MissingKernel {
        driver: claim.symbol,
        kernel: "<missing>",
    })?;
    let kernel = find_kernel(program, kernel_symbol).ok_or(Stage6KernelError::MissingKernel {
        driver: claim.symbol,
        kernel: kernel_symbol,
    })?;
    Stage6Relation::from_symbol(kernel.relation).ok_or(Stage6KernelError::UnknownRelation {
        relation: kernel.relation,
    })
}

fn instance_round_offset(
    program: &'static Stage6CpuProgramPlan,
    driver: &'static str,
    claim: &'static str,
) -> Result<usize, Stage6KernelError> {
    program
        .instance_results
        .iter()
        .find(|instance| instance.source == driver && instance.claim == claim)
        .map(|instance| instance.round_offset)
        .ok_or(Stage6KernelError::MissingClaim {
            batch: driver,
            claim,
        })
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
    trim_trailing_zero_coefficients(&mut combined);
    UnivariatePoly::new(combined)
}

fn trim_trailing_zero_coefficients<F: Field>(coefficients: &mut Vec<F>) {
    while coefficients.len() > 1 && coefficients.last() == Some(&F::zero()) {
        let _ = coefficients.pop();
    }
}

fn round_poly_from_dense_terms<F: Field>(
    factors: &[Vec<F>],
    terms: &[DenseTerm<F>],
    active_scale: F,
    degree_bound: usize,
    relation: Stage6Relation,
) -> Result<UnivariatePoly<F>, Stage6KernelError> {
    if degree_bound > 5 {
        return Err(Stage6KernelError::InvalidProof {
            driver: relation.symbol(),
            reason: "stage6 dense degree bound is unsupported",
        });
    }
    let half = factors.first().map_or(0, |factor| factor.len() / 2);
    for term in terms {
        if term.factors.len() > degree_bound {
            return Err(Stage6KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "stage6 dense term exceeds degree bound",
            });
        }
        if term.factors.iter().any(|factor| *factor >= factors.len()) {
            return Err(Stage6KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "stage6 dense term references missing factor",
            });
        }
    }

    let eval_count = degree_bound + 1;
    let mut evals = if half >= DENSE_BIND_PAR_THRESHOLD {
        (0..half)
            .into_par_iter()
            .fold(
                || vec![F::zero(); eval_count],
                |mut row_evals, row| {
                    accumulate_dense_row_evaluations(factors, terms, row, &mut row_evals);
                    row_evals
                },
            )
            .reduce(
                || vec![F::zero(); eval_count],
                |mut left, right| {
                    for (left, right) in left.iter_mut().zip(right) {
                        *left += right;
                    }
                    left
                },
            )
    } else {
        let mut total = vec![F::zero(); eval_count];
        for row in 0..half {
            accumulate_dense_row_evaluations(factors, terms, row, &mut total);
        }
        total
    };
    for eval in &mut evals {
        *eval *= active_scale;
    }
    Ok(UnivariatePoly::interpolate_over_integers(&evals))
}

fn accumulate_dense_row_evaluations<F: Field>(
    factors: &[Vec<F>],
    terms: &[DenseTerm<F>],
    row: usize,
    evals: &mut [F],
) {
    for (point, eval) in evals.iter_mut().enumerate() {
        let point = F::from_u64(point as u64);
        for term in terms {
            let mut term_eval = term.coefficient;
            for &factor in &term.factors {
                let low = factors[factor][2 * row];
                let high = factors[factor][2 * row + 1];
                term_eval *= low + (high - low) * point;
            }
            *eval += term_eval;
        }
    }
}

pub fn execute_stage6_program<F, T, E>(
    program: &'static Stage6CpuProgramPlan,
    mode: Stage6ExecutionMode,
    executor: &mut E,
    transcript: &mut T,
) -> Result<Stage6ExecutionArtifacts<F>, Stage6KernelError>
where
    F: Field,
    T: Transcript<Challenge = F>,
    E: Stage6KernelExecutor<F>,
{
    let mut artifacts = Stage6ExecutionArtifacts::default();
    for step in program.steps {
        match step.kind {
            "transcript_squeeze" => {
                let squeeze =
                    find_squeeze(program, step.symbol).ok_or(Stage6KernelError::MissingValue {
                        symbol: step.symbol,
                    })?;
                let values = transcript.challenge_vector(squeeze.count);
                executor.observe_challenge_vector(squeeze, &values)?;
                artifacts.challenge_vectors.push(Stage6ChallengeVector {
                    symbol: squeeze.symbol,
                    values,
                });
            }
            "transcript_absorb_bytes" => {
                let absorb = find_absorb_bytes(program, step.symbol).ok_or(
                    Stage6KernelError::MissingValue {
                        symbol: step.symbol,
                    },
                )?;
                absorb_stage6_bytes(absorb, transcript);
            }
            "sumcheck_driver" => {
                let driver =
                    find_driver(program, step.symbol).ok_or(Stage6KernelError::MissingDriver {
                        driver: step.symbol,
                    })?;
                let kernel_symbol = driver.kernel.ok_or(Stage6KernelError::MissingKernel {
                    driver: driver.symbol,
                    kernel: "<missing>",
                })?;
                let kernel = find_kernel(program, kernel_symbol).ok_or(
                    Stage6KernelError::MissingKernel {
                        driver: driver.symbol,
                        kernel: kernel_symbol,
                    },
                )?;
                let batch =
                    find_batch(program, driver.batch).ok_or(Stage6KernelError::MissingBatch {
                        driver: driver.symbol,
                        batch: driver.batch,
                    })?;
                let context = Stage6KernelContext {
                    mode,
                    program,
                    kernel,
                    batch,
                    driver,
                };
                let output = match mode {
                    Stage6ExecutionMode::Prover => executor.prove_sumcheck(context, transcript)?,
                    Stage6ExecutionMode::Verifier => {
                        executor.verify_sumcheck(context, transcript)?
                    }
                };
                executor.observe_sumcheck_output(&output)?;
                artifacts.sumchecks.push(output);
            }
            _ => {
                return Err(Stage6KernelError::InvalidProgramStep {
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

fn absorb_stage6_bytes<T>(absorb: &'static Stage6TranscriptAbsorbBytesPlan, transcript: &mut T)
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
    program: &'static Stage6CpuProgramPlan,
    symbol: &str,
) -> Option<&'static Stage6TranscriptSqueezePlan> {
    program
        .transcript_squeezes
        .iter()
        .find(|squeeze| squeeze.symbol == symbol)
}

fn find_absorb_bytes(
    program: &'static Stage6CpuProgramPlan,
    symbol: &str,
) -> Option<&'static Stage6TranscriptAbsorbBytesPlan> {
    program
        .transcript_absorb_bytes
        .iter()
        .find(|absorb| absorb.symbol == symbol)
}

fn find_driver(
    program: &'static Stage6CpuProgramPlan,
    symbol: &str,
) -> Option<&'static Stage6SumcheckDriverPlan> {
    program
        .drivers
        .iter()
        .find(|driver| driver.symbol == symbol)
}

fn find_kernel(
    program: &'static Stage6CpuProgramPlan,
    symbol: &str,
) -> Option<&'static Stage6KernelPlan> {
    program
        .kernels
        .iter()
        .find(|kernel| kernel.symbol == symbol)
}

fn find_batch(
    program: &'static Stage6CpuProgramPlan,
    symbol: &str,
) -> Option<&'static Stage6SumcheckBatchPlan> {
    program.batches.iter().find(|batch| batch.symbol == symbol)
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Fr;
    use jolt_transcript::Blake2bTranscript;

    const PARAMS: Stage6Params = Stage6Params {
        field: "bn254_fr",
        pcs: "dory",
        transcript: "blake2b_transcript",
    };
    const STEPS: &[Stage6ProgramStepPlan] = &[Stage6ProgramStepPlan {
        kind: "sumcheck_driver",
        symbol: "stage6.sumcheck",
    }];
    const CLAIM_INPUT_OPENINGS: &[&str] = &["stage6.input.claim"];
    const CLAIMS: &[Stage6SumcheckClaimPlan] = &[Stage6SumcheckClaimPlan {
        symbol: "stage6.claim",
        stage: "stage6",
        domain: "jolt.test_domain",
        num_rounds: 1,
        degree: 1,
        claim: "stage6.claim",
        kernel: Some("jolt.cpu.stage6.batched"),
        relation: None,
        claim_value: "stage6.input.claim",
        input_openings: CLAIM_INPUT_OPENINGS,
    }];
    const KERNELS: &[Stage6KernelPlan] = &[Stage6KernelPlan {
        symbol: "jolt.cpu.stage6.batched",
        relation: "jolt.stage6.batched",
        kind: "sumcheck",
        backend: "cpu",
        abi: "jolt_stage6_batched",
    }];
    const BATCHES: &[Stage6SumcheckBatchPlan] = &[Stage6SumcheckBatchPlan {
        symbol: "stage6.batch",
        stage: "stage6",
        proof_slot: "stage6.sumcheck",
        policy: "jolt_core_stage6_aligned",
        count: 0,
        ordered_claims: &[],
        claim_operands: &[],
        claim_label: "sumcheck_claim",
        round_label: "sumcheck_poly",
        round_schedule: &[],
    }];
    const DRIVERS: &[Stage6SumcheckDriverPlan] = &[Stage6SumcheckDriverPlan {
        symbol: "stage6.sumcheck",
        stage: "stage6",
        proof_slot: "stage6.sumcheck",
        kernel: Some("jolt.cpu.stage6.batched"),
        relation: None,
        batch: "stage6.batch",
        policy: "jolt_core_stage6_aligned",
        round_schedule: &[],
        claim_label: "sumcheck_claim",
        round_label: "sumcheck_poly",
        num_rounds: 0,
        degree: 0,
    }];
    const PROGRAM: Stage6CpuProgramPlan = Stage6CpuProgramPlan {
        role: "prover",
        params: PARAMS,
        steps: STEPS,
        transcript_squeezes: &[],
        transcript_absorb_bytes: &[],
        opening_inputs: &[],
        field_constants: &[],
        field_exprs: &[],
        kernels: KERNELS,
        claims: &[],
        batches: BATCHES,
        drivers: DRIVERS,
        instance_results: &[],
        evals: &[],
        point_zeros: &[],
        point_slices: &[],
        point_concats: &[],
        opening_claims: &[],
        opening_equalities: &[],
        opening_batches: &[],
    };
    const REPLAY_BATCHES: &[Stage6SumcheckBatchPlan] = &[Stage6SumcheckBatchPlan {
        symbol: "stage6.batch",
        stage: "stage6",
        proof_slot: "stage6.sumcheck",
        policy: "jolt_core_stage6_aligned",
        count: 1,
        ordered_claims: &["stage6.claim"],
        claim_operands: &["stage6.claim"],
        claim_label: "sumcheck_claim",
        round_label: "sumcheck_poly",
        round_schedule: &[1],
    }];
    const REPLAY_DRIVERS: &[Stage6SumcheckDriverPlan] = &[Stage6SumcheckDriverPlan {
        symbol: "stage6.sumcheck",
        stage: "stage6",
        proof_slot: "stage6.sumcheck",
        kernel: Some("jolt.cpu.stage6.batched"),
        relation: None,
        batch: "stage6.batch",
        policy: "jolt_core_stage6_aligned",
        round_schedule: &[1],
        claim_label: "sumcheck_claim",
        round_label: "sumcheck_poly",
        num_rounds: 1,
        degree: 1,
    }];
    const REPLAY_PROGRAM: Stage6CpuProgramPlan = Stage6CpuProgramPlan {
        role: "prover",
        params: PARAMS,
        steps: STEPS,
        transcript_squeezes: &[],
        transcript_absorb_bytes: &[],
        opening_inputs: &[],
        field_constants: &[],
        field_exprs: &[],
        kernels: KERNELS,
        claims: CLAIMS,
        batches: REPLAY_BATCHES,
        drivers: REPLAY_DRIVERS,
        instance_results: &[],
        evals: &[],
        point_zeros: &[],
        point_slices: &[],
        point_concats: &[],
        opening_claims: &[],
        opening_equalities: &[],
        opening_batches: &[],
    };
    const BYTECODE_FIELD_CONSTANTS: &[Stage6FieldConstantPlan] = &[
        Stage6FieldConstantPlan {
            symbol: "stage6.bytecode_read_raf.gamma",
            field: "bn254_fr",
            value: 2,
        },
        Stage6FieldConstantPlan {
            symbol: "stage6.bytecode_read_raf.stage1_gamma",
            field: "bn254_fr",
            value: 3,
        },
        Stage6FieldConstantPlan {
            symbol: "stage6.bytecode_read_raf.stage2_gamma",
            field: "bn254_fr",
            value: 5,
        },
        Stage6FieldConstantPlan {
            symbol: "stage6.bytecode_read_raf.stage3_gamma",
            field: "bn254_fr",
            value: 7,
        },
        Stage6FieldConstantPlan {
            symbol: "stage6.bytecode_read_raf.stage4_gamma",
            field: "bn254_fr",
            value: 11,
        },
        Stage6FieldConstantPlan {
            symbol: "stage6.bytecode_read_raf.stage5_gamma",
            field: "bn254_fr",
            value: 13,
        },
    ];
    const BYTECODE_CLAIM_INPUT_OPENINGS: &[&str] = &["stage6.input.bytecode_read_raf_claim"];
    const BYTECODE_CLAIMS: &[Stage6SumcheckClaimPlan] = &[Stage6SumcheckClaimPlan {
        symbol: "stage6.bytecode_read_raf.input",
        stage: "stage6",
        domain: "jolt.stage6_bytecode_read_raf_domain",
        num_rounds: 3,
        degree: 3,
        claim: "stage6.bytecode_read_raf.weighted_prior_stage_values",
        kernel: Some("jolt.cpu.stage6.bytecode_read_raf"),
        relation: None,
        claim_value: "stage6.input.bytecode_read_raf_claim",
        input_openings: BYTECODE_CLAIM_INPUT_OPENINGS,
    }];
    const BYTECODE_KERNELS: &[Stage6KernelPlan] = &[
        Stage6KernelPlan {
            symbol: "jolt.cpu.stage6.bytecode_read_raf",
            relation: "jolt.stage6.bytecode_read_raf",
            kind: "sumcheck",
            backend: "cpu",
            abi: "jolt_stage6_bytecode_read_raf",
        },
        Stage6KernelPlan {
            symbol: "jolt.cpu.stage6.batched",
            relation: "jolt.stage6.batched",
            kind: "sumcheck",
            backend: "cpu",
            abi: "jolt_stage6_batched",
        },
    ];
    const BYTECODE_BATCHES: &[Stage6SumcheckBatchPlan] = &[Stage6SumcheckBatchPlan {
        symbol: "stage6.batch",
        stage: "stage6",
        proof_slot: "stage6.sumcheck",
        policy: "jolt_core_stage6_aligned",
        count: 1,
        ordered_claims: &["stage6.bytecode_read_raf.input"],
        claim_operands: &["stage6.bytecode_read_raf.input"],
        claim_label: "sumcheck_claim",
        round_label: "sumcheck_poly",
        round_schedule: &[3],
    }];
    const BYTECODE_DRIVERS: &[Stage6SumcheckDriverPlan] = &[Stage6SumcheckDriverPlan {
        symbol: "stage6.sumcheck",
        stage: "stage6",
        proof_slot: "stage6.sumcheck",
        kernel: Some("jolt.cpu.stage6.batched"),
        relation: None,
        batch: "stage6.batch",
        policy: "jolt_core_stage6_aligned",
        round_schedule: &[3],
        claim_label: "sumcheck_claim",
        round_label: "sumcheck_poly",
        num_rounds: 3,
        degree: 3,
    }];
    const BYTECODE_INSTANCE_RESULTS: &[Stage6SumcheckInstanceResultPlan] = &[
        Stage6SumcheckInstanceResultPlan {
            symbol: "stage6.bytecode_read_raf.instance",
            source: "stage6.sumcheck",
            claim: "stage6.bytecode_read_raf.input",
            relation: "jolt.stage6.bytecode_read_raf",
            index: 0,
            point_arity: 3,
            num_rounds: 3,
            round_offset: 0,
            point_order: "bytecode_read_raf",
            degree: 3,
        },
        Stage6SumcheckInstanceResultPlan {
            symbol: "stage6.hamming_booleanity.instance",
            source: "stage6.sumcheck",
            claim: "stage6.hamming_booleanity.input",
            relation: "jolt.stage6.hamming_booleanity",
            index: 1,
            point_arity: 1,
            num_rounds: 1,
            round_offset: 0,
            point_order: "reverse",
            degree: 3,
        },
    ];
    const BYTECODE_EVALS: &[Stage6SumcheckEvalPlan] = &[
        Stage6SumcheckEvalPlan {
            symbol: "stage6.bytecode_read_raf.eval.BytecodeRa_0",
            source: "stage6.sumcheck",
            name: "stage6.bytecode_read_raf.eval.BytecodeRa_0",
            index: 0,
            oracle: "BytecodeRa_0",
        },
        Stage6SumcheckEvalPlan {
            symbol: "stage6.bytecode_read_raf.eval.BytecodeRa_1",
            source: "stage6.sumcheck",
            name: "stage6.bytecode_read_raf.eval.BytecodeRa_1",
            index: 1,
            oracle: "BytecodeRa_1",
        },
    ];
    const BYTECODE_PROGRAM: Stage6CpuProgramPlan = Stage6CpuProgramPlan {
        role: "prover",
        params: PARAMS,
        steps: STEPS,
        transcript_squeezes: &[],
        transcript_absorb_bytes: &[],
        opening_inputs: &[],
        field_constants: BYTECODE_FIELD_CONSTANTS,
        field_exprs: &[],
        kernels: BYTECODE_KERNELS,
        claims: BYTECODE_CLAIMS,
        batches: BYTECODE_BATCHES,
        drivers: BYTECODE_DRIVERS,
        instance_results: BYTECODE_INSTANCE_RESULTS,
        evals: BYTECODE_EVALS,
        point_zeros: &[],
        point_slices: &[],
        point_concats: &[],
        opening_claims: &[],
        opening_equalities: &[],
        opening_batches: &[],
    };
    const BOOLEANITY_OPENING_INPUTS: &[Stage6OpeningInputPlan] = &[Stage6OpeningInputPlan {
        symbol: "stage6.input.stage5.instruction_read_raf.InstructionRa_0",
        source_stage: "stage5",
        source_claim: "stage5.instruction_read_raf.opening.InstructionRa_0",
        oracle: "InstructionRa_0",
        domain: "jolt.stage5_instruction_ra_chunk_domain",
        point_arity: 4,
        claim_kind: "virtual",
    }];
    const BOOLEANITY_FIELD_CONSTANTS: &[Stage6FieldConstantPlan] = &[
        Stage6FieldConstantPlan {
            symbol: "stage6.zero",
            field: "bn254_fr",
            value: 0,
        },
        Stage6FieldConstantPlan {
            symbol: "stage6.booleanity.gamma",
            field: "bn254_fr",
            value: 2,
        },
    ];
    const BOOLEANITY_CLAIMS: &[Stage6SumcheckClaimPlan] = &[Stage6SumcheckClaimPlan {
        symbol: "stage6.booleanity.input",
        stage: "stage6",
        domain: "jolt.stage6_booleanity_domain",
        num_rounds: 3,
        degree: 3,
        claim: "stage6.booleanity.zero",
        kernel: Some("jolt.cpu.stage6.booleanity"),
        relation: None,
        claim_value: "stage6.zero",
        input_openings: &[],
    }];
    const BOOLEANITY_KERNELS: &[Stage6KernelPlan] = &[
        Stage6KernelPlan {
            symbol: "jolt.cpu.stage6.booleanity",
            relation: "jolt.stage6.booleanity",
            kind: "sumcheck",
            backend: "cpu",
            abi: "jolt_stage6_booleanity",
        },
        Stage6KernelPlan {
            symbol: "jolt.cpu.stage6.batched",
            relation: "jolt.stage6.batched",
            kind: "sumcheck",
            backend: "cpu",
            abi: "jolt_stage6_batched",
        },
    ];
    const BOOLEANITY_BATCHES: &[Stage6SumcheckBatchPlan] = &[Stage6SumcheckBatchPlan {
        symbol: "stage6.batch",
        stage: "stage6",
        proof_slot: "stage6.sumcheck",
        policy: "jolt_core_stage6_aligned",
        count: 1,
        ordered_claims: &["stage6.booleanity.input"],
        claim_operands: &["stage6.booleanity.input"],
        claim_label: "sumcheck_claim",
        round_label: "sumcheck_poly",
        round_schedule: &[3],
    }];
    const BOOLEANITY_DRIVERS: &[Stage6SumcheckDriverPlan] = &[Stage6SumcheckDriverPlan {
        symbol: "stage6.sumcheck",
        stage: "stage6",
        proof_slot: "stage6.sumcheck",
        kernel: Some("jolt.cpu.stage6.batched"),
        relation: None,
        batch: "stage6.batch",
        policy: "jolt_core_stage6_aligned",
        round_schedule: &[3],
        claim_label: "sumcheck_claim",
        round_label: "sumcheck_poly",
        num_rounds: 3,
        degree: 3,
    }];
    const BOOLEANITY_INSTANCE_RESULTS: &[Stage6SumcheckInstanceResultPlan] = &[
        Stage6SumcheckInstanceResultPlan {
            symbol: "stage6.booleanity.instance",
            source: "stage6.sumcheck",
            claim: "stage6.booleanity.input",
            relation: "jolt.stage6.booleanity",
            index: 0,
            point_arity: 3,
            num_rounds: 3,
            round_offset: 0,
            point_order: "stage6_booleanity",
            degree: 3,
        },
        Stage6SumcheckInstanceResultPlan {
            symbol: "stage6.hamming_booleanity.instance",
            source: "stage6.sumcheck",
            claim: "stage6.hamming_booleanity.input",
            relation: "jolt.stage6.hamming_booleanity",
            index: 1,
            point_arity: 2,
            num_rounds: 2,
            round_offset: 0,
            point_order: "reverse",
            degree: 3,
        },
    ];
    const BOOLEANITY_EVALS: &[Stage6SumcheckEvalPlan] = &[
        Stage6SumcheckEvalPlan {
            symbol: "stage6.booleanity.eval.InstructionRa_0",
            source: "stage6.sumcheck",
            name: "stage6.booleanity.eval.InstructionRa_0",
            index: 0,
            oracle: "InstructionRa_0",
        },
        Stage6SumcheckEvalPlan {
            symbol: "stage6.booleanity.eval.BytecodeRa_0",
            source: "stage6.sumcheck",
            name: "stage6.booleanity.eval.BytecodeRa_0",
            index: 1,
            oracle: "BytecodeRa_0",
        },
        Stage6SumcheckEvalPlan {
            symbol: "stage6.booleanity.eval.RamRa_0",
            source: "stage6.sumcheck",
            name: "stage6.booleanity.eval.RamRa_0",
            index: 2,
            oracle: "RamRa_0",
        },
    ];
    const BOOLEANITY_PROGRAM: Stage6CpuProgramPlan = Stage6CpuProgramPlan {
        role: "prover",
        params: PARAMS,
        steps: STEPS,
        transcript_squeezes: &[],
        transcript_absorb_bytes: &[],
        opening_inputs: BOOLEANITY_OPENING_INPUTS,
        field_constants: BOOLEANITY_FIELD_CONSTANTS,
        field_exprs: &[],
        kernels: BOOLEANITY_KERNELS,
        claims: BOOLEANITY_CLAIMS,
        batches: BOOLEANITY_BATCHES,
        drivers: BOOLEANITY_DRIVERS,
        instance_results: BOOLEANITY_INSTANCE_RESULTS,
        evals: BOOLEANITY_EVALS,
        point_zeros: &[],
        point_slices: &[],
        point_concats: &[],
        opening_claims: &[],
        opening_equalities: &[],
        opening_batches: &[],
    };
    const HAMMING_OPENING_INPUTS: &[Stage6OpeningInputPlan] = &[Stage6OpeningInputPlan {
        symbol: "stage6.input.stage1.LookupOutput",
        source_stage: "stage1",
        source_claim: "stage1.outer_remaining.opening.LookupOutput",
        oracle: "LookupOutput",
        domain: "jolt.trace_domain",
        point_arity: 2,
        claim_kind: "virtual",
    }];
    const HAMMING_FIELD_CONSTANTS: &[Stage6FieldConstantPlan] = &[Stage6FieldConstantPlan {
        symbol: "stage6.zero",
        field: "bn254_fr",
        value: 0,
    }];
    const HAMMING_CLAIM_INPUT_OPENINGS: &[&str] = &["stage6.input.stage1.LookupOutput"];
    const HAMMING_CLAIMS: &[Stage6SumcheckClaimPlan] = &[Stage6SumcheckClaimPlan {
        symbol: "stage6.hamming_booleanity.input",
        stage: "stage6",
        domain: "jolt.trace_domain",
        num_rounds: 2,
        degree: 3,
        claim: "stage6.hamming_booleanity.zero",
        kernel: Some("jolt.cpu.stage6.hamming_booleanity"),
        relation: None,
        claim_value: "stage6.zero",
        input_openings: HAMMING_CLAIM_INPUT_OPENINGS,
    }];
    const HAMMING_KERNELS: &[Stage6KernelPlan] = &[
        Stage6KernelPlan {
            symbol: "jolt.cpu.stage6.hamming_booleanity",
            relation: "jolt.stage6.hamming_booleanity",
            kind: "sumcheck",
            backend: "cpu",
            abi: "jolt_stage6_hamming_booleanity",
        },
        Stage6KernelPlan {
            symbol: "jolt.cpu.stage6.batched",
            relation: "jolt.stage6.batched",
            kind: "sumcheck",
            backend: "cpu",
            abi: "jolt_stage6_batched",
        },
    ];
    const HAMMING_BATCHES: &[Stage6SumcheckBatchPlan] = &[Stage6SumcheckBatchPlan {
        symbol: "stage6.batch",
        stage: "stage6",
        proof_slot: "stage6.sumcheck",
        policy: "jolt_core_stage6_aligned",
        count: 1,
        ordered_claims: &["stage6.hamming_booleanity.input"],
        claim_operands: &["stage6.hamming_booleanity.input"],
        claim_label: "sumcheck_claim",
        round_label: "sumcheck_poly",
        round_schedule: &[2],
    }];
    const HAMMING_DRIVERS: &[Stage6SumcheckDriverPlan] = &[Stage6SumcheckDriverPlan {
        symbol: "stage6.sumcheck",
        stage: "stage6",
        proof_slot: "stage6.sumcheck",
        kernel: Some("jolt.cpu.stage6.batched"),
        relation: None,
        batch: "stage6.batch",
        policy: "jolt_core_stage6_aligned",
        round_schedule: &[2],
        claim_label: "sumcheck_claim",
        round_label: "sumcheck_poly",
        num_rounds: 2,
        degree: 3,
    }];
    const HAMMING_INSTANCE_RESULTS: &[Stage6SumcheckInstanceResultPlan] =
        &[Stage6SumcheckInstanceResultPlan {
            symbol: "stage6.hamming_booleanity.instance",
            source: "stage6.sumcheck",
            claim: "stage6.hamming_booleanity.input",
            relation: "jolt.stage6.hamming_booleanity",
            index: 0,
            point_arity: 2,
            num_rounds: 2,
            round_offset: 0,
            point_order: "reverse",
            degree: 3,
        }];
    const HAMMING_EVALS: &[Stage6SumcheckEvalPlan] = &[Stage6SumcheckEvalPlan {
        symbol: "stage6.hamming_booleanity.eval.HammingWeight",
        source: "stage6.sumcheck",
        name: "stage6.hamming_booleanity.eval.HammingWeight",
        index: 0,
        oracle: "HammingWeight",
    }];
    const HAMMING_PROGRAM: Stage6CpuProgramPlan = Stage6CpuProgramPlan {
        role: "prover",
        params: PARAMS,
        steps: STEPS,
        transcript_squeezes: &[],
        transcript_absorb_bytes: &[],
        opening_inputs: HAMMING_OPENING_INPUTS,
        field_constants: HAMMING_FIELD_CONSTANTS,
        field_exprs: &[],
        kernels: HAMMING_KERNELS,
        claims: HAMMING_CLAIMS,
        batches: HAMMING_BATCHES,
        drivers: HAMMING_DRIVERS,
        instance_results: HAMMING_INSTANCE_RESULTS,
        evals: HAMMING_EVALS,
        point_zeros: &[],
        point_slices: &[],
        point_concats: &[],
        opening_claims: &[],
        opening_equalities: &[],
        opening_batches: &[],
    };
    const INC_FIELD_CONSTANTS: &[Stage6FieldConstantPlan] = &[Stage6FieldConstantPlan {
        symbol: "stage6.inc_claim_reduction.gamma",
        field: "bn254_fr",
        value: 2,
    }];
    const INC_CLAIM_INPUT_OPENINGS: &[&str] = &[
        "stage6.input.stage2.ram_read_write.RamInc",
        "stage6.input.stage4.ram_val_check.RamInc",
        "stage6.input.stage4.registers_read_write.RdInc",
        "stage6.input.stage5.registers_val_evaluation.RdInc",
    ];
    const INC_CLAIMS: &[Stage6SumcheckClaimPlan] = &[Stage6SumcheckClaimPlan {
        symbol: "stage6.inc_claim_reduction.input",
        stage: "stage6",
        domain: "jolt.trace_domain",
        num_rounds: 2,
        degree: 2,
        claim: "stage6.inc_claim_reduction.weighted_increments",
        kernel: Some("jolt.cpu.stage6.inc_claim_reduction"),
        relation: None,
        claim_value: "stage6.input.inc_claim",
        input_openings: INC_CLAIM_INPUT_OPENINGS,
    }];
    const INC_KERNELS: &[Stage6KernelPlan] = &[
        Stage6KernelPlan {
            symbol: "jolt.cpu.stage6.inc_claim_reduction",
            relation: "jolt.stage6.inc_claim_reduction",
            kind: "sumcheck",
            backend: "cpu",
            abi: "jolt_stage6_inc_claim_reduction",
        },
        Stage6KernelPlan {
            symbol: "jolt.cpu.stage6.batched",
            relation: "jolt.stage6.batched",
            kind: "sumcheck",
            backend: "cpu",
            abi: "jolt_stage6_batched",
        },
    ];
    const INC_BATCHES: &[Stage6SumcheckBatchPlan] = &[Stage6SumcheckBatchPlan {
        symbol: "stage6.batch",
        stage: "stage6",
        proof_slot: "stage6.sumcheck",
        policy: "jolt_core_stage6_aligned",
        count: 1,
        ordered_claims: &["stage6.inc_claim_reduction.input"],
        claim_operands: &["stage6.inc_claim_reduction.input"],
        claim_label: "sumcheck_claim",
        round_label: "sumcheck_poly",
        round_schedule: &[2],
    }];
    const INC_DRIVERS: &[Stage6SumcheckDriverPlan] = &[Stage6SumcheckDriverPlan {
        symbol: "stage6.sumcheck",
        stage: "stage6",
        proof_slot: "stage6.sumcheck",
        kernel: Some("jolt.cpu.stage6.batched"),
        relation: None,
        batch: "stage6.batch",
        policy: "jolt_core_stage6_aligned",
        round_schedule: &[2],
        claim_label: "sumcheck_claim",
        round_label: "sumcheck_poly",
        num_rounds: 2,
        degree: 2,
    }];
    const INC_INSTANCE_RESULTS: &[Stage6SumcheckInstanceResultPlan] =
        &[Stage6SumcheckInstanceResultPlan {
            symbol: "stage6.inc_claim_reduction.instance",
            source: "stage6.sumcheck",
            claim: "stage6.inc_claim_reduction.input",
            relation: "jolt.stage6.inc_claim_reduction",
            index: 0,
            point_arity: 2,
            num_rounds: 2,
            round_offset: 0,
            point_order: "reverse",
            degree: 2,
        }];
    const INC_EVALS: &[Stage6SumcheckEvalPlan] = &[
        Stage6SumcheckEvalPlan {
            symbol: "stage6.inc_claim_reduction.eval.RamInc",
            source: "stage6.sumcheck",
            name: "stage6.inc_claim_reduction.eval.RamInc",
            index: 0,
            oracle: "RamInc",
        },
        Stage6SumcheckEvalPlan {
            symbol: "stage6.inc_claim_reduction.eval.RdInc",
            source: "stage6.sumcheck",
            name: "stage6.inc_claim_reduction.eval.RdInc",
            index: 1,
            oracle: "RdInc",
        },
    ];
    const INC_PROGRAM: Stage6CpuProgramPlan = Stage6CpuProgramPlan {
        role: "prover",
        params: PARAMS,
        steps: STEPS,
        transcript_squeezes: &[],
        transcript_absorb_bytes: &[],
        opening_inputs: &[],
        field_constants: INC_FIELD_CONSTANTS,
        field_exprs: &[],
        kernels: INC_KERNELS,
        claims: INC_CLAIMS,
        batches: INC_BATCHES,
        drivers: INC_DRIVERS,
        instance_results: INC_INSTANCE_RESULTS,
        evals: INC_EVALS,
        point_zeros: &[],
        point_slices: &[],
        point_concats: &[],
        opening_claims: &[],
        opening_equalities: &[],
        opening_batches: &[],
    };
    const RAM_RA_CLAIM_INPUT_OPENINGS: &[&str] =
        &["stage6.input.stage5.ram_ra_claim_reduction.RamRa"];
    const RAM_RA_CLAIMS: &[Stage6SumcheckClaimPlan] = &[Stage6SumcheckClaimPlan {
        symbol: "stage6.ram_ra_virtual.input",
        stage: "stage6",
        domain: "jolt.trace_domain",
        num_rounds: 2,
        degree: 5,
        claim: "stage6.ram_ra_virtual.weighted_ram_ra",
        kernel: Some("jolt.cpu.stage6.ram_ra_virtual"),
        relation: None,
        claim_value: "stage6.input.ram_ra_virtual_claim",
        input_openings: RAM_RA_CLAIM_INPUT_OPENINGS,
    }];
    const RAM_RA_KERNELS: &[Stage6KernelPlan] = &[
        Stage6KernelPlan {
            symbol: "jolt.cpu.stage6.ram_ra_virtual",
            relation: "jolt.stage6.ram_ra_virtual",
            kind: "sumcheck",
            backend: "cpu",
            abi: "jolt_stage6_ram_ra_virtual",
        },
        Stage6KernelPlan {
            symbol: "jolt.cpu.stage6.batched",
            relation: "jolt.stage6.batched",
            kind: "sumcheck",
            backend: "cpu",
            abi: "jolt_stage6_batched",
        },
    ];
    const RAM_RA_BATCHES: &[Stage6SumcheckBatchPlan] = &[Stage6SumcheckBatchPlan {
        symbol: "stage6.batch",
        stage: "stage6",
        proof_slot: "stage6.sumcheck",
        policy: "jolt_core_stage6_aligned",
        count: 1,
        ordered_claims: &["stage6.ram_ra_virtual.input"],
        claim_operands: &["stage6.ram_ra_virtual.input"],
        claim_label: "sumcheck_claim",
        round_label: "sumcheck_poly",
        round_schedule: &[2],
    }];
    const RAM_RA_DRIVERS: &[Stage6SumcheckDriverPlan] = &[Stage6SumcheckDriverPlan {
        symbol: "stage6.sumcheck",
        stage: "stage6",
        proof_slot: "stage6.sumcheck",
        kernel: Some("jolt.cpu.stage6.batched"),
        relation: None,
        batch: "stage6.batch",
        policy: "jolt_core_stage6_aligned",
        round_schedule: &[2],
        claim_label: "sumcheck_claim",
        round_label: "sumcheck_poly",
        num_rounds: 2,
        degree: 5,
    }];
    const RAM_RA_INSTANCE_RESULTS: &[Stage6SumcheckInstanceResultPlan] =
        &[Stage6SumcheckInstanceResultPlan {
            symbol: "stage6.ram_ra_virtual.instance",
            source: "stage6.sumcheck",
            claim: "stage6.ram_ra_virtual.input",
            relation: "jolt.stage6.ram_ra_virtual",
            index: 0,
            point_arity: 2,
            num_rounds: 2,
            round_offset: 0,
            point_order: "reverse",
            degree: 5,
        }];
    const RAM_RA_EVALS: &[Stage6SumcheckEvalPlan] = &[
        Stage6SumcheckEvalPlan {
            symbol: "stage6.ram_ra_virtual.eval.RamRa_0",
            source: "stage6.sumcheck",
            name: "stage6.ram_ra_virtual.eval.RamRa_0",
            index: 0,
            oracle: "RamRa_0",
        },
        Stage6SumcheckEvalPlan {
            symbol: "stage6.ram_ra_virtual.eval.RamRa_1",
            source: "stage6.sumcheck",
            name: "stage6.ram_ra_virtual.eval.RamRa_1",
            index: 1,
            oracle: "RamRa_1",
        },
        Stage6SumcheckEvalPlan {
            symbol: "stage6.ram_ra_virtual.eval.RamRa_2",
            source: "stage6.sumcheck",
            name: "stage6.ram_ra_virtual.eval.RamRa_2",
            index: 2,
            oracle: "RamRa_2",
        },
        Stage6SumcheckEvalPlan {
            symbol: "stage6.ram_ra_virtual.eval.RamRa_3",
            source: "stage6.sumcheck",
            name: "stage6.ram_ra_virtual.eval.RamRa_3",
            index: 3,
            oracle: "RamRa_3",
        },
    ];
    const RAM_RA_PROGRAM: Stage6CpuProgramPlan = Stage6CpuProgramPlan {
        role: "prover",
        params: PARAMS,
        steps: STEPS,
        transcript_squeezes: &[],
        transcript_absorb_bytes: &[],
        opening_inputs: &[],
        field_constants: &[],
        field_exprs: &[],
        kernels: RAM_RA_KERNELS,
        claims: RAM_RA_CLAIMS,
        batches: RAM_RA_BATCHES,
        drivers: RAM_RA_DRIVERS,
        instance_results: RAM_RA_INSTANCE_RESULTS,
        evals: RAM_RA_EVALS,
        point_zeros: &[],
        point_slices: &[],
        point_concats: &[],
        opening_claims: &[],
        opening_equalities: &[],
        opening_batches: &[],
    };
    const INSTRUCTION_RA_OPENING_INPUTS: &[Stage6OpeningInputPlan] = &[
        Stage6OpeningInputPlan {
            symbol: "stage6.input.stage5.instruction_read_raf.InstructionRa_0",
            source_stage: "stage5",
            source_claim: "stage5.instruction_read_raf.opening.InstructionRa_0",
            oracle: "InstructionRa_0",
            domain: "jolt.stage5_instruction_ra_chunk_domain",
            point_arity: 2,
            claim_kind: "virtual",
        },
        Stage6OpeningInputPlan {
            symbol: "stage6.input.stage5.instruction_read_raf.InstructionRa_1",
            source_stage: "stage5",
            source_claim: "stage5.instruction_read_raf.opening.InstructionRa_1",
            oracle: "InstructionRa_1",
            domain: "jolt.stage5_instruction_ra_chunk_domain",
            point_arity: 2,
            claim_kind: "virtual",
        },
    ];
    const INSTRUCTION_RA_FIELD_CONSTANTS: &[Stage6FieldConstantPlan] = &[Stage6FieldConstantPlan {
        symbol: "stage6.instruction_ra_virtual.gamma",
        field: "bn254_fr",
        value: 3,
    }];
    const INSTRUCTION_RA_CLAIM_INPUT_OPENINGS: &[&str] = &[
        "stage6.input.stage5.instruction_read_raf.InstructionRa_0",
        "stage6.input.stage5.instruction_read_raf.InstructionRa_1",
    ];
    const INSTRUCTION_RA_CLAIMS: &[Stage6SumcheckClaimPlan] = &[Stage6SumcheckClaimPlan {
        symbol: "stage6.instruction_ra_virtual.input",
        stage: "stage6",
        domain: "jolt.trace_domain",
        num_rounds: 2,
        degree: 5,
        claim: "stage6.instruction_ra_virtual.weighted_instruction_ra",
        kernel: Some("jolt.cpu.stage6.instruction_ra_virtual"),
        relation: None,
        claim_value: "stage6.input.instruction_ra_virtual_claim",
        input_openings: INSTRUCTION_RA_CLAIM_INPUT_OPENINGS,
    }];
    const INSTRUCTION_RA_KERNELS: &[Stage6KernelPlan] = &[
        Stage6KernelPlan {
            symbol: "jolt.cpu.stage6.instruction_ra_virtual",
            relation: "jolt.stage6.instruction_ra_virtual",
            kind: "sumcheck",
            backend: "cpu",
            abi: "jolt_stage6_instruction_ra_virtual",
        },
        Stage6KernelPlan {
            symbol: "jolt.cpu.stage6.batched",
            relation: "jolt.stage6.batched",
            kind: "sumcheck",
            backend: "cpu",
            abi: "jolt_stage6_batched",
        },
    ];
    const INSTRUCTION_RA_BATCHES: &[Stage6SumcheckBatchPlan] = &[Stage6SumcheckBatchPlan {
        symbol: "stage6.batch",
        stage: "stage6",
        proof_slot: "stage6.sumcheck",
        policy: "jolt_core_stage6_aligned",
        count: 1,
        ordered_claims: &["stage6.instruction_ra_virtual.input"],
        claim_operands: &["stage6.instruction_ra_virtual.input"],
        claim_label: "sumcheck_claim",
        round_label: "sumcheck_poly",
        round_schedule: &[2],
    }];
    const INSTRUCTION_RA_DRIVERS: &[Stage6SumcheckDriverPlan] = &[Stage6SumcheckDriverPlan {
        symbol: "stage6.sumcheck",
        stage: "stage6",
        proof_slot: "stage6.sumcheck",
        kernel: Some("jolt.cpu.stage6.batched"),
        relation: None,
        batch: "stage6.batch",
        policy: "jolt_core_stage6_aligned",
        round_schedule: &[2],
        claim_label: "sumcheck_claim",
        round_label: "sumcheck_poly",
        num_rounds: 2,
        degree: 5,
    }];
    const INSTRUCTION_RA_INSTANCE_RESULTS: &[Stage6SumcheckInstanceResultPlan] =
        &[Stage6SumcheckInstanceResultPlan {
            symbol: "stage6.instruction_ra_virtual.instance",
            source: "stage6.sumcheck",
            claim: "stage6.instruction_ra_virtual.input",
            relation: "jolt.stage6.instruction_ra_virtual",
            index: 0,
            point_arity: 2,
            num_rounds: 2,
            round_offset: 0,
            point_order: "reverse",
            degree: 5,
        }];
    const INSTRUCTION_RA_EVALS: &[Stage6SumcheckEvalPlan] = &[
        Stage6SumcheckEvalPlan {
            symbol: "stage6.instruction_ra_virtual.eval.InstructionRa_0",
            source: "stage6.sumcheck",
            name: "stage6.instruction_ra_virtual.eval.InstructionRa_0",
            index: 0,
            oracle: "InstructionRa_0",
        },
        Stage6SumcheckEvalPlan {
            symbol: "stage6.instruction_ra_virtual.eval.InstructionRa_1",
            source: "stage6.sumcheck",
            name: "stage6.instruction_ra_virtual.eval.InstructionRa_1",
            index: 1,
            oracle: "InstructionRa_1",
        },
        Stage6SumcheckEvalPlan {
            symbol: "stage6.instruction_ra_virtual.eval.InstructionRa_2",
            source: "stage6.sumcheck",
            name: "stage6.instruction_ra_virtual.eval.InstructionRa_2",
            index: 2,
            oracle: "InstructionRa_2",
        },
        Stage6SumcheckEvalPlan {
            symbol: "stage6.instruction_ra_virtual.eval.InstructionRa_3",
            source: "stage6.sumcheck",
            name: "stage6.instruction_ra_virtual.eval.InstructionRa_3",
            index: 3,
            oracle: "InstructionRa_3",
        },
        Stage6SumcheckEvalPlan {
            symbol: "stage6.instruction_ra_virtual.eval.InstructionRa_4",
            source: "stage6.sumcheck",
            name: "stage6.instruction_ra_virtual.eval.InstructionRa_4",
            index: 4,
            oracle: "InstructionRa_4",
        },
        Stage6SumcheckEvalPlan {
            symbol: "stage6.instruction_ra_virtual.eval.InstructionRa_5",
            source: "stage6.sumcheck",
            name: "stage6.instruction_ra_virtual.eval.InstructionRa_5",
            index: 5,
            oracle: "InstructionRa_5",
        },
        Stage6SumcheckEvalPlan {
            symbol: "stage6.instruction_ra_virtual.eval.InstructionRa_6",
            source: "stage6.sumcheck",
            name: "stage6.instruction_ra_virtual.eval.InstructionRa_6",
            index: 6,
            oracle: "InstructionRa_6",
        },
        Stage6SumcheckEvalPlan {
            symbol: "stage6.instruction_ra_virtual.eval.InstructionRa_7",
            source: "stage6.sumcheck",
            name: "stage6.instruction_ra_virtual.eval.InstructionRa_7",
            index: 7,
            oracle: "InstructionRa_7",
        },
    ];
    const INSTRUCTION_RA_PROGRAM: Stage6CpuProgramPlan = Stage6CpuProgramPlan {
        role: "prover",
        params: PARAMS,
        steps: STEPS,
        transcript_squeezes: &[],
        transcript_absorb_bytes: &[],
        opening_inputs: INSTRUCTION_RA_OPENING_INPUTS,
        field_constants: INSTRUCTION_RA_FIELD_CONSTANTS,
        field_exprs: &[],
        kernels: INSTRUCTION_RA_KERNELS,
        claims: INSTRUCTION_RA_CLAIMS,
        batches: INSTRUCTION_RA_BATCHES,
        drivers: INSTRUCTION_RA_DRIVERS,
        instance_results: INSTRUCTION_RA_INSTANCE_RESULTS,
        evals: INSTRUCTION_RA_EVALS,
        point_zeros: &[],
        point_slices: &[],
        point_concats: &[],
        opening_claims: &[],
        opening_equalities: &[],
        opening_batches: &[],
    };

    #[test]
    fn stage6_symbols_parse_as_stage6_relations_and_abis() {
        assert_eq!(
            Stage6Relation::from_symbol("jolt.stage6.bytecode_read_raf"),
            Some(Stage6Relation::BytecodeReadRaf)
        );
        assert_eq!(
            Stage6Relation::from_symbol("jolt.stage6.batched"),
            Some(Stage6Relation::Batched)
        );
        assert_eq!(
            Stage6KernelAbi::from_name("jolt_stage6_bytecode_read_raf"),
            Some(Stage6KernelAbi::BytecodeReadRaf)
        );
        assert_eq!(
            Stage6KernelAbi::from_name("jolt_stage6_batched"),
            Some(Stage6KernelAbi::Batched)
        );
        assert_eq!(Stage6Relation::from_symbol("jolt.stage5.batched"), None);
        assert_eq!(Stage6KernelAbi::from_name("jolt_stage5_batched"), None);
    }

    #[test]
    fn unsupported_executor_reaches_stage6_batched_abi() {
        let mut executor = UnsupportedStage6KernelExecutor;
        let mut transcript = Blake2bTranscript::<Fr>::new(b"stage6_test");
        let error = execute_stage6_program(
            &PROGRAM,
            Stage6ExecutionMode::Prover,
            &mut executor,
            &mut transcript,
        )
        .expect_err("stage6 kernels are not implemented yet");

        assert_eq!(
            error,
            Stage6KernelError::KernelNotImplemented {
                abi: "jolt_stage6_batched"
            }
        );
    }

    #[test]
    fn proof_carrying_executor_replays_stage6_sumcheck_transcript() {
        let input_claim = Fr::from_u64(3);
        let (proof, opening_inputs) = replay_proof(input_claim);
        let mut executor = Stage6ProofCarryingKernelExecutor::new(&proof, &opening_inputs);
        let mut transcript = Blake2bTranscript::<Fr>::new(b"stage6_test");

        let artifacts = execute_stage6_program(
            &REPLAY_PROGRAM,
            Stage6ExecutionMode::Prover,
            &mut executor,
            &mut transcript,
        )
        .expect("proof-carrying Stage 6 replay succeeds");

        assert_eq!(artifacts.sumchecks.len(), 1);
        assert_eq!(artifacts.sumchecks[0].point, proof.sumchecks[0].point);
    }

    #[test]
    fn proof_carrying_executor_rejects_bad_round_polynomial() {
        let input_claim = Fr::from_u64(3);
        let (mut proof, opening_inputs) = replay_proof(input_claim);
        let mut coefficients = proof.sumchecks[0].proof.round_polynomials[0]
            .coefficients()
            .to_vec();
        coefficients[0] += Fr::from_u64(1);
        proof.sumchecks[0].proof.round_polynomials[0] = UnivariatePoly::new(coefficients);
        let mut executor = Stage6ProofCarryingKernelExecutor::new(&proof, &opening_inputs);
        let mut transcript = Blake2bTranscript::<Fr>::new(b"stage6_test");

        let error = execute_stage6_program(
            &REPLAY_PROGRAM,
            Stage6ExecutionMode::Prover,
            &mut executor,
            &mut transcript,
        )
        .expect_err("tampered Stage 6 replay is rejected");

        assert_eq!(
            error,
            Stage6KernelError::InvalidProof {
                driver: "stage6.sumcheck",
                reason: "batched round check failed"
            }
        );
    }

    #[test]
    fn bytecode_read_raf_prover_produces_verifiable_sumcheck() {
        let entries = bytecode_entries();
        let data = Stage6BytecodeReadRafData {
            entries: &entries,
            entry_bytecode_index: 2,
            num_lookup_tables: 2,
        };
        let bytecode_ra_0 = frs(&[1, 2, 3, 4]);
        let bytecode_ra_1 = frs(&[2, 1, 4, 3]);
        let bytecode_ra_chunks: [&[Fr]; 2] = [&bytecode_ra_0, &bytecode_ra_1];
        let mut opening_inputs = bytecode_opening_inputs();
        let input_claim = bytecode_read_raf_claim(data, &bytecode_ra_chunks, &opening_inputs);
        opening_inputs.push(Stage6OpeningInputValue {
            symbol: "stage6.input.bytecode_read_raf_claim",
            point: Vec::new(),
            eval: input_claim,
        });
        let prover_inputs = Stage6ProverInputs::new(&opening_inputs).with_bytecode_read_raf(
            Stage6BytecodeReadRafWitness {
                data,
                bytecode_ra_chunks: &bytecode_ra_chunks,
            },
        );
        let mut prover = Stage6ProverKernelExecutor::new(prover_inputs);
        let mut prover_transcript = Blake2bTranscript::<Fr>::new(b"stage6_test");
        let artifacts = execute_stage6_program(
            &BYTECODE_PROGRAM,
            Stage6ExecutionMode::Prover,
            &mut prover,
            &mut prover_transcript,
        )
        .expect("bytecode read RAF prover succeeds");

        assert_eq!(artifacts.sumchecks.len(), 1);
        assert_eq!(artifacts.sumchecks[0].evals.len(), 2);

        let proof = Stage6Proof {
            sumchecks: artifacts.sumchecks.clone(),
        };
        let mut verifier = Stage6ProofCarryingKernelExecutor::new(&proof, &opening_inputs)
            .with_bytecode_read_raf_data(data);
        let mut verifier_transcript = Blake2bTranscript::<Fr>::new(b"stage6_test");
        let verified = execute_stage6_program(
            &BYTECODE_PROGRAM,
            Stage6ExecutionMode::Verifier,
            &mut verifier,
            &mut verifier_transcript,
        )
        .expect("proof-carrying verifier accepts bytecode read RAF output");

        assert_eq!(artifacts.sumchecks[0].point, verified.sumchecks[0].point);
        assert_eq!(
            named_eval_values(&artifacts.sumchecks[0].evals),
            named_eval_values(&verified.sumchecks[0].evals)
        );
    }

    #[test]
    fn bytecode_read_raf_verifier_rejects_bad_final_eval() {
        let entries = bytecode_entries();
        let data = Stage6BytecodeReadRafData {
            entries: &entries,
            entry_bytecode_index: 2,
            num_lookup_tables: 2,
        };
        let bytecode_ra_0 = frs(&[1, 2, 3, 4]);
        let bytecode_ra_1 = frs(&[2, 1, 4, 3]);
        let bytecode_ra_chunks: [&[Fr]; 2] = [&bytecode_ra_0, &bytecode_ra_1];
        let mut opening_inputs = bytecode_opening_inputs();
        let input_claim = bytecode_read_raf_claim(data, &bytecode_ra_chunks, &opening_inputs);
        opening_inputs.push(Stage6OpeningInputValue {
            symbol: "stage6.input.bytecode_read_raf_claim",
            point: Vec::new(),
            eval: input_claim,
        });
        let prover_inputs = Stage6ProverInputs::new(&opening_inputs).with_bytecode_read_raf(
            Stage6BytecodeReadRafWitness {
                data,
                bytecode_ra_chunks: &bytecode_ra_chunks,
            },
        );
        let mut prover = Stage6ProverKernelExecutor::new(prover_inputs);
        let mut prover_transcript = Blake2bTranscript::<Fr>::new(b"stage6_test");
        let mut artifacts = execute_stage6_program(
            &BYTECODE_PROGRAM,
            Stage6ExecutionMode::Prover,
            &mut prover,
            &mut prover_transcript,
        )
        .expect("bytecode read RAF prover succeeds");

        artifacts.sumchecks[0].evals[0].value += Fr::from_u64(1);
        let proof = Stage6Proof {
            sumchecks: artifacts.sumchecks,
        };
        let mut verifier = Stage6ProofCarryingKernelExecutor::new(&proof, &opening_inputs)
            .with_bytecode_read_raf_data(data);
        let mut verifier_transcript = Blake2bTranscript::<Fr>::new(b"stage6_test");
        let error = execute_stage6_program(
            &BYTECODE_PROGRAM,
            Stage6ExecutionMode::Verifier,
            &mut verifier,
            &mut verifier_transcript,
        )
        .expect_err("tampered bytecode read RAF eval is rejected");

        assert_eq!(
            error,
            Stage6KernelError::InvalidProof {
                driver: "stage6.sumcheck",
                reason: "batched output claim mismatch"
            }
        );
    }

    #[test]
    fn booleanity_prover_produces_verifiable_sumcheck() {
        let instruction_ra = frs(&[0, 1, 0, 1, 1, 0, 1, 0]);
        let bytecode_ra = frs(&[1, 0, 1, 0, 0, 1, 0, 1]);
        let ram_ra = frs(&[0, 0, 1, 1, 0, 1, 1, 0]);
        let chunks: [&[Fr]; 3] = [&instruction_ra, &bytecode_ra, &ram_ra];
        let stage5_point = frs(&[11, 13, 2, 3]);
        let opening_inputs = vec![Stage6OpeningInputValue {
            symbol: "stage6.input.stage5.instruction_read_raf.InstructionRa_0",
            point: stage5_point,
            eval: Fr::from_u64(0),
        }];
        let prover_inputs = Stage6ProverInputs::new(&opening_inputs)
            .with_booleanity(Stage6BooleanityWitness { chunks: &chunks });
        let mut prover = Stage6ProverKernelExecutor::new(prover_inputs);
        let mut prover_transcript = Blake2bTranscript::<Fr>::new(b"stage6_test");
        let artifacts = execute_stage6_program(
            &BOOLEANITY_PROGRAM,
            Stage6ExecutionMode::Prover,
            &mut prover,
            &mut prover_transcript,
        )
        .expect("booleanity prover succeeds");

        assert_eq!(artifacts.sumchecks.len(), 1);
        assert_eq!(artifacts.sumchecks[0].evals.len(), 3);

        let proof = Stage6Proof {
            sumchecks: artifacts.sumchecks.clone(),
        };
        let mut verifier = Stage6ProofCarryingKernelExecutor::new(&proof, &opening_inputs);
        let mut verifier_transcript = Blake2bTranscript::<Fr>::new(b"stage6_test");
        let verified = execute_stage6_program(
            &BOOLEANITY_PROGRAM,
            Stage6ExecutionMode::Verifier,
            &mut verifier,
            &mut verifier_transcript,
        )
        .expect("proof-carrying verifier accepts booleanity output");

        assert_eq!(artifacts.sumchecks[0].point, verified.sumchecks[0].point);
        assert_eq!(
            named_eval_values(&artifacts.sumchecks[0].evals),
            named_eval_values(&verified.sumchecks[0].evals)
        );
    }

    #[test]
    fn booleanity_verifier_rejects_bad_final_eval() {
        let instruction_ra = frs(&[0, 1, 0, 1, 1, 0, 1, 0]);
        let bytecode_ra = frs(&[1, 0, 1, 0, 0, 1, 0, 1]);
        let ram_ra = frs(&[0, 0, 1, 1, 0, 1, 1, 0]);
        let chunks: [&[Fr]; 3] = [&instruction_ra, &bytecode_ra, &ram_ra];
        let stage5_point = frs(&[11, 13, 2, 3]);
        let opening_inputs = vec![Stage6OpeningInputValue {
            symbol: "stage6.input.stage5.instruction_read_raf.InstructionRa_0",
            point: stage5_point,
            eval: Fr::from_u64(0),
        }];
        let prover_inputs = Stage6ProverInputs::new(&opening_inputs)
            .with_booleanity(Stage6BooleanityWitness { chunks: &chunks });
        let mut prover = Stage6ProverKernelExecutor::new(prover_inputs);
        let mut prover_transcript = Blake2bTranscript::<Fr>::new(b"stage6_test");
        let mut artifacts = execute_stage6_program(
            &BOOLEANITY_PROGRAM,
            Stage6ExecutionMode::Prover,
            &mut prover,
            &mut prover_transcript,
        )
        .expect("booleanity prover succeeds");

        artifacts.sumchecks[0].evals[0].value += Fr::from_u64(1);
        let proof = Stage6Proof {
            sumchecks: artifacts.sumchecks,
        };
        let mut verifier = Stage6ProofCarryingKernelExecutor::new(&proof, &opening_inputs);
        let mut verifier_transcript = Blake2bTranscript::<Fr>::new(b"stage6_test");
        let error = execute_stage6_program(
            &BOOLEANITY_PROGRAM,
            Stage6ExecutionMode::Verifier,
            &mut verifier,
            &mut verifier_transcript,
        )
        .expect_err("tampered booleanity eval is rejected");

        assert_eq!(
            error,
            Stage6KernelError::InvalidProof {
                driver: "stage6.sumcheck",
                reason: "batched output claim mismatch"
            }
        );
    }

    #[test]
    fn hamming_booleanity_prover_produces_verifiable_sumcheck() {
        let lookup_output_point = frs(&[3, 5]);
        let hamming_weight = frs(&[0, 1, 1, 0]);
        let opening_inputs = vec![Stage6OpeningInputValue {
            symbol: "stage6.input.stage1.LookupOutput",
            point: lookup_output_point,
            eval: Fr::from_u64(9),
        }];
        let prover_inputs = Stage6ProverInputs::new(&opening_inputs).with_hamming_booleanity(
            Stage6HammingBooleanityWitness {
                hamming_weight: &hamming_weight,
            },
        );
        let mut prover = Stage6ProverKernelExecutor::new(prover_inputs);
        let mut prover_transcript = Blake2bTranscript::<Fr>::new(b"stage6_test");
        let artifacts = execute_stage6_program(
            &HAMMING_PROGRAM,
            Stage6ExecutionMode::Prover,
            &mut prover,
            &mut prover_transcript,
        )
        .expect("hamming booleanity prover succeeds");

        assert_eq!(artifacts.sumchecks.len(), 1);
        assert_eq!(artifacts.sumchecks[0].evals.len(), 1);

        let proof = Stage6Proof {
            sumchecks: artifacts.sumchecks.clone(),
        };
        let mut verifier = Stage6ProofCarryingKernelExecutor::new(&proof, &opening_inputs);
        let mut verifier_transcript = Blake2bTranscript::<Fr>::new(b"stage6_test");
        let verified = execute_stage6_program(
            &HAMMING_PROGRAM,
            Stage6ExecutionMode::Verifier,
            &mut verifier,
            &mut verifier_transcript,
        )
        .expect("proof-carrying verifier accepts hamming output");

        assert_eq!(artifacts.sumchecks[0].point, verified.sumchecks[0].point);
        assert_eq!(
            named_eval_values(&artifacts.sumchecks[0].evals),
            named_eval_values(&verified.sumchecks[0].evals)
        );
    }

    #[test]
    fn hamming_booleanity_prover_requires_witness() {
        let opening_inputs = vec![Stage6OpeningInputValue {
            symbol: "stage6.input.stage1.LookupOutput",
            point: frs(&[3, 5]),
            eval: Fr::from_u64(9),
        }];
        let mut prover = Stage6ProverKernelExecutor::new(Stage6ProverInputs::new(&opening_inputs));
        let mut transcript = Blake2bTranscript::<Fr>::new(b"stage6_test");
        let error = execute_stage6_program(
            &HAMMING_PROGRAM,
            Stage6ExecutionMode::Prover,
            &mut prover,
            &mut transcript,
        )
        .expect_err("hamming booleanity witness is required");

        assert_eq!(
            error,
            Stage6KernelError::MissingKernelInput {
                kernel: "jolt_stage6_batched",
                input: "hamming_booleanity"
            }
        );
    }

    #[test]
    fn hamming_booleanity_verifier_rejects_bad_final_eval() {
        let lookup_output_point = frs(&[3, 5]);
        let hamming_weight = frs(&[0, 1, 1, 0]);
        let opening_inputs = vec![Stage6OpeningInputValue {
            symbol: "stage6.input.stage1.LookupOutput",
            point: lookup_output_point,
            eval: Fr::from_u64(9),
        }];
        let prover_inputs = Stage6ProverInputs::new(&opening_inputs).with_hamming_booleanity(
            Stage6HammingBooleanityWitness {
                hamming_weight: &hamming_weight,
            },
        );
        let mut prover = Stage6ProverKernelExecutor::new(prover_inputs);
        let mut prover_transcript = Blake2bTranscript::<Fr>::new(b"stage6_test");
        let mut artifacts = execute_stage6_program(
            &HAMMING_PROGRAM,
            Stage6ExecutionMode::Prover,
            &mut prover,
            &mut prover_transcript,
        )
        .expect("hamming booleanity prover succeeds");

        artifacts.sumchecks[0].evals[0].value += Fr::from_u64(1);
        let proof = Stage6Proof {
            sumchecks: artifacts.sumchecks,
        };
        let mut verifier = Stage6ProofCarryingKernelExecutor::new(&proof, &opening_inputs);
        let mut verifier_transcript = Blake2bTranscript::<Fr>::new(b"stage6_test");
        let error = execute_stage6_program(
            &HAMMING_PROGRAM,
            Stage6ExecutionMode::Verifier,
            &mut verifier,
            &mut verifier_transcript,
        )
        .expect_err("tampered hamming eval is rejected");

        assert_eq!(
            error,
            Stage6KernelError::InvalidProof {
                driver: "stage6.sumcheck",
                reason: "batched output claim mismatch"
            }
        );
    }

    #[test]
    fn inc_claim_reduction_prover_produces_verifiable_sumcheck() {
        let ram_inc = frs(&[1, 3, 5, 7]);
        let rd_inc = frs(&[2, 4, 6, 8]);
        let ram_inc_stage2_point = frs(&[2, 3]);
        let ram_inc_stage4_point = frs(&[5, 7]);
        let rd_inc_stage4_point = frs(&[11, 13]);
        let rd_inc_stage5_point = frs(&[17, 19]);
        let gamma = Fr::from_u64(2);
        let input_claim = multilinear_eval(&ram_inc, &ram_inc_stage2_point)
            + gamma * multilinear_eval(&ram_inc, &ram_inc_stage4_point)
            + gamma.square() * multilinear_eval(&rd_inc, &rd_inc_stage4_point)
            + gamma.square() * gamma * multilinear_eval(&rd_inc, &rd_inc_stage5_point);
        let opening_inputs = vec![
            Stage6OpeningInputValue {
                symbol: "stage6.input.stage2.ram_read_write.RamInc",
                point: ram_inc_stage2_point.clone(),
                eval: multilinear_eval(&ram_inc, &ram_inc_stage2_point),
            },
            Stage6OpeningInputValue {
                symbol: "stage6.input.stage4.ram_val_check.RamInc",
                point: ram_inc_stage4_point.clone(),
                eval: multilinear_eval(&ram_inc, &ram_inc_stage4_point),
            },
            Stage6OpeningInputValue {
                symbol: "stage6.input.stage4.registers_read_write.RdInc",
                point: rd_inc_stage4_point.clone(),
                eval: multilinear_eval(&rd_inc, &rd_inc_stage4_point),
            },
            Stage6OpeningInputValue {
                symbol: "stage6.input.stage5.registers_val_evaluation.RdInc",
                point: rd_inc_stage5_point.clone(),
                eval: multilinear_eval(&rd_inc, &rd_inc_stage5_point),
            },
            Stage6OpeningInputValue {
                symbol: "stage6.input.inc_claim",
                point: Vec::new(),
                eval: input_claim,
            },
        ];
        let prover_inputs = Stage6ProverInputs::new(&opening_inputs).with_inc_claim_reduction(
            Stage6IncClaimReductionWitness {
                ram_inc: &ram_inc,
                rd_inc: &rd_inc,
            },
        );
        let mut prover = Stage6ProverKernelExecutor::new(prover_inputs);
        let mut prover_transcript = Blake2bTranscript::<Fr>::new(b"stage6_test");
        let artifacts = execute_stage6_program(
            &INC_PROGRAM,
            Stage6ExecutionMode::Prover,
            &mut prover,
            &mut prover_transcript,
        )
        .expect("increment claim-reduction prover succeeds");

        assert_eq!(artifacts.sumchecks.len(), 1);
        assert_eq!(artifacts.sumchecks[0].evals.len(), 2);

        let proof = Stage6Proof {
            sumchecks: artifacts.sumchecks.clone(),
        };
        let mut verifier = Stage6ProofCarryingKernelExecutor::new(&proof, &opening_inputs);
        let mut verifier_transcript = Blake2bTranscript::<Fr>::new(b"stage6_test");
        let verified = execute_stage6_program(
            &INC_PROGRAM,
            Stage6ExecutionMode::Verifier,
            &mut verifier,
            &mut verifier_transcript,
        )
        .expect("proof-carrying verifier accepts increment output");

        assert_eq!(artifacts.sumchecks[0].point, verified.sumchecks[0].point);
        assert_eq!(
            named_eval_values(&artifacts.sumchecks[0].evals),
            named_eval_values(&verified.sumchecks[0].evals)
        );
    }

    #[test]
    fn inc_claim_reduction_verifier_rejects_bad_final_eval() {
        let ram_inc = frs(&[1, 3, 5, 7]);
        let rd_inc = frs(&[2, 4, 6, 8]);
        let ram_inc_stage2_point = frs(&[2, 3]);
        let ram_inc_stage4_point = frs(&[5, 7]);
        let rd_inc_stage4_point = frs(&[11, 13]);
        let rd_inc_stage5_point = frs(&[17, 19]);
        let gamma = Fr::from_u64(2);
        let input_claim = multilinear_eval(&ram_inc, &ram_inc_stage2_point)
            + gamma * multilinear_eval(&ram_inc, &ram_inc_stage4_point)
            + gamma.square() * multilinear_eval(&rd_inc, &rd_inc_stage4_point)
            + gamma.square() * gamma * multilinear_eval(&rd_inc, &rd_inc_stage5_point);
        let opening_inputs = vec![
            Stage6OpeningInputValue {
                symbol: "stage6.input.stage2.ram_read_write.RamInc",
                point: ram_inc_stage2_point,
                eval: Fr::from_u64(0),
            },
            Stage6OpeningInputValue {
                symbol: "stage6.input.stage4.ram_val_check.RamInc",
                point: ram_inc_stage4_point,
                eval: Fr::from_u64(0),
            },
            Stage6OpeningInputValue {
                symbol: "stage6.input.stage4.registers_read_write.RdInc",
                point: rd_inc_stage4_point,
                eval: Fr::from_u64(0),
            },
            Stage6OpeningInputValue {
                symbol: "stage6.input.stage5.registers_val_evaluation.RdInc",
                point: rd_inc_stage5_point,
                eval: Fr::from_u64(0),
            },
            Stage6OpeningInputValue {
                symbol: "stage6.input.inc_claim",
                point: Vec::new(),
                eval: input_claim,
            },
        ];
        let prover_inputs = Stage6ProverInputs::new(&opening_inputs).with_inc_claim_reduction(
            Stage6IncClaimReductionWitness {
                ram_inc: &ram_inc,
                rd_inc: &rd_inc,
            },
        );
        let mut prover = Stage6ProverKernelExecutor::new(prover_inputs);
        let mut prover_transcript = Blake2bTranscript::<Fr>::new(b"stage6_test");
        let mut artifacts = execute_stage6_program(
            &INC_PROGRAM,
            Stage6ExecutionMode::Prover,
            &mut prover,
            &mut prover_transcript,
        )
        .expect("increment claim-reduction prover succeeds");

        artifacts.sumchecks[0].evals[0].value += Fr::from_u64(1);
        let proof = Stage6Proof {
            sumchecks: artifacts.sumchecks,
        };
        let mut verifier = Stage6ProofCarryingKernelExecutor::new(&proof, &opening_inputs);
        let mut verifier_transcript = Blake2bTranscript::<Fr>::new(b"stage6_test");
        let error = execute_stage6_program(
            &INC_PROGRAM,
            Stage6ExecutionMode::Verifier,
            &mut verifier,
            &mut verifier_transcript,
        )
        .expect_err("tampered increment eval is rejected");

        assert_eq!(
            error,
            Stage6KernelError::InvalidProof {
                driver: "stage6.sumcheck",
                reason: "batched output claim mismatch"
            }
        );
    }

    #[test]
    fn ram_ra_virtual_prover_produces_verifiable_sumcheck() {
        let ram_ra_0 = frs(&[1, 2, 3, 4]);
        let ram_ra_1 = frs(&[2, 3, 4, 5]);
        let ram_ra_2 = frs(&[3, 4, 5, 6]);
        let ram_ra_3 = frs(&[4, 5, 6, 7]);
        let ram_ra_chunks: [&[Fr]; 4] = [&ram_ra_0, &ram_ra_1, &ram_ra_2, &ram_ra_3];
        let input_point = frs(&[2, 3]);
        let input_claim = product_virtual_claim(&ram_ra_chunks, &input_point);
        let opening_inputs = vec![
            Stage6OpeningInputValue {
                symbol: "stage6.input.stage5.ram_ra_claim_reduction.RamRa",
                point: input_point,
                eval: input_claim,
            },
            Stage6OpeningInputValue {
                symbol: "stage6.input.ram_ra_virtual_claim",
                point: Vec::new(),
                eval: input_claim,
            },
        ];
        let prover_inputs = Stage6ProverInputs::new(&opening_inputs).with_ram_ra_virtual(
            Stage6RamRaVirtualWitness {
                ram_ra_chunks: &ram_ra_chunks,
            },
        );
        let mut prover = Stage6ProverKernelExecutor::new(prover_inputs);
        let mut prover_transcript = Blake2bTranscript::<Fr>::new(b"stage6_test");
        let artifacts = execute_stage6_program(
            &RAM_RA_PROGRAM,
            Stage6ExecutionMode::Prover,
            &mut prover,
            &mut prover_transcript,
        )
        .expect("RAM RA virtualization prover succeeds");

        assert_eq!(artifacts.sumchecks.len(), 1);
        assert_eq!(artifacts.sumchecks[0].evals.len(), 4);

        let proof = Stage6Proof {
            sumchecks: artifacts.sumchecks.clone(),
        };
        let mut verifier = Stage6ProofCarryingKernelExecutor::new(&proof, &opening_inputs);
        let mut verifier_transcript = Blake2bTranscript::<Fr>::new(b"stage6_test");
        let verified = execute_stage6_program(
            &RAM_RA_PROGRAM,
            Stage6ExecutionMode::Verifier,
            &mut verifier,
            &mut verifier_transcript,
        )
        .expect("proof-carrying verifier accepts RAM RA virtualization output");

        assert_eq!(artifacts.sumchecks[0].point, verified.sumchecks[0].point);
        assert_eq!(
            named_eval_values(&artifacts.sumchecks[0].evals),
            named_eval_values(&verified.sumchecks[0].evals)
        );
    }

    #[test]
    fn ram_ra_virtual_verifier_rejects_bad_final_eval() {
        let ram_ra_0 = frs(&[1, 2, 3, 4]);
        let ram_ra_1 = frs(&[2, 3, 4, 5]);
        let ram_ra_2 = frs(&[3, 4, 5, 6]);
        let ram_ra_3 = frs(&[4, 5, 6, 7]);
        let ram_ra_chunks: [&[Fr]; 4] = [&ram_ra_0, &ram_ra_1, &ram_ra_2, &ram_ra_3];
        let input_point = frs(&[2, 3]);
        let input_claim = product_virtual_claim(&ram_ra_chunks, &input_point);
        let opening_inputs = vec![
            Stage6OpeningInputValue {
                symbol: "stage6.input.stage5.ram_ra_claim_reduction.RamRa",
                point: input_point,
                eval: input_claim,
            },
            Stage6OpeningInputValue {
                symbol: "stage6.input.ram_ra_virtual_claim",
                point: Vec::new(),
                eval: input_claim,
            },
        ];
        let prover_inputs = Stage6ProverInputs::new(&opening_inputs).with_ram_ra_virtual(
            Stage6RamRaVirtualWitness {
                ram_ra_chunks: &ram_ra_chunks,
            },
        );
        let mut prover = Stage6ProverKernelExecutor::new(prover_inputs);
        let mut prover_transcript = Blake2bTranscript::<Fr>::new(b"stage6_test");
        let mut artifacts = execute_stage6_program(
            &RAM_RA_PROGRAM,
            Stage6ExecutionMode::Prover,
            &mut prover,
            &mut prover_transcript,
        )
        .expect("RAM RA virtualization prover succeeds");

        artifacts.sumchecks[0].evals[0].value += Fr::from_u64(1);
        let proof = Stage6Proof {
            sumchecks: artifacts.sumchecks,
        };
        let mut verifier = Stage6ProofCarryingKernelExecutor::new(&proof, &opening_inputs);
        let mut verifier_transcript = Blake2bTranscript::<Fr>::new(b"stage6_test");
        let error = execute_stage6_program(
            &RAM_RA_PROGRAM,
            Stage6ExecutionMode::Verifier,
            &mut verifier,
            &mut verifier_transcript,
        )
        .expect_err("tampered RAM RA eval is rejected");

        assert_eq!(
            error,
            Stage6KernelError::InvalidProof {
                driver: "stage6.sumcheck",
                reason: "batched output claim mismatch"
            }
        );
    }

    #[test]
    fn instruction_ra_virtual_prover_produces_verifiable_sumcheck() {
        let instruction_ra_0 = frs(&[1, 2, 3, 4]);
        let instruction_ra_1 = frs(&[2, 3, 4, 5]);
        let instruction_ra_2 = frs(&[3, 4, 5, 6]);
        let instruction_ra_3 = frs(&[4, 5, 6, 7]);
        let instruction_ra_4 = frs(&[5, 6, 7, 8]);
        let instruction_ra_5 = frs(&[6, 7, 8, 9]);
        let instruction_ra_6 = frs(&[7, 8, 9, 10]);
        let instruction_ra_7 = frs(&[8, 9, 10, 11]);
        let instruction_ra_chunks: [&[Fr]; 8] = [
            &instruction_ra_0,
            &instruction_ra_1,
            &instruction_ra_2,
            &instruction_ra_3,
            &instruction_ra_4,
            &instruction_ra_5,
            &instruction_ra_6,
            &instruction_ra_7,
        ];
        let input_point = frs(&[2, 3]);
        let gamma = Fr::from_u64(3);
        let input_claim =
            grouped_product_virtual_claim(&instruction_ra_chunks, 2, &input_point, gamma);
        let opening_inputs = vec![
            Stage6OpeningInputValue {
                symbol: "stage6.input.stage5.instruction_read_raf.InstructionRa_0",
                point: input_point.clone(),
                eval: input_claim,
            },
            Stage6OpeningInputValue {
                symbol: "stage6.input.stage5.instruction_read_raf.InstructionRa_1",
                point: input_point,
                eval: Fr::from_u64(0),
            },
            Stage6OpeningInputValue {
                symbol: "stage6.input.instruction_ra_virtual_claim",
                point: Vec::new(),
                eval: input_claim,
            },
        ];
        let prover_inputs = Stage6ProverInputs::new(&opening_inputs).with_instruction_ra_virtual(
            Stage6InstructionRaVirtualWitness {
                instruction_ra_chunks: &instruction_ra_chunks,
                virtual_count: 2,
            },
        );
        let mut prover = Stage6ProverKernelExecutor::new(prover_inputs);
        let mut prover_transcript = Blake2bTranscript::<Fr>::new(b"stage6_test");
        let artifacts = execute_stage6_program(
            &INSTRUCTION_RA_PROGRAM,
            Stage6ExecutionMode::Prover,
            &mut prover,
            &mut prover_transcript,
        )
        .expect("instruction RA virtualization prover succeeds");

        assert_eq!(artifacts.sumchecks.len(), 1);
        assert_eq!(artifacts.sumchecks[0].evals.len(), 8);

        let proof = Stage6Proof {
            sumchecks: artifacts.sumchecks.clone(),
        };
        let mut verifier = Stage6ProofCarryingKernelExecutor::new(&proof, &opening_inputs);
        let mut verifier_transcript = Blake2bTranscript::<Fr>::new(b"stage6_test");
        let verified = execute_stage6_program(
            &INSTRUCTION_RA_PROGRAM,
            Stage6ExecutionMode::Verifier,
            &mut verifier,
            &mut verifier_transcript,
        )
        .expect("proof-carrying verifier accepts instruction RA virtualization output");

        assert_eq!(artifacts.sumchecks[0].point, verified.sumchecks[0].point);
        assert_eq!(
            named_eval_values(&artifacts.sumchecks[0].evals),
            named_eval_values(&verified.sumchecks[0].evals)
        );
    }

    #[test]
    fn instruction_ra_virtual_verifier_rejects_bad_final_eval() {
        let instruction_ra_0 = frs(&[1, 2, 3, 4]);
        let instruction_ra_1 = frs(&[2, 3, 4, 5]);
        let instruction_ra_2 = frs(&[3, 4, 5, 6]);
        let instruction_ra_3 = frs(&[4, 5, 6, 7]);
        let instruction_ra_4 = frs(&[5, 6, 7, 8]);
        let instruction_ra_5 = frs(&[6, 7, 8, 9]);
        let instruction_ra_6 = frs(&[7, 8, 9, 10]);
        let instruction_ra_7 = frs(&[8, 9, 10, 11]);
        let instruction_ra_chunks: [&[Fr]; 8] = [
            &instruction_ra_0,
            &instruction_ra_1,
            &instruction_ra_2,
            &instruction_ra_3,
            &instruction_ra_4,
            &instruction_ra_5,
            &instruction_ra_6,
            &instruction_ra_7,
        ];
        let input_point = frs(&[2, 3]);
        let gamma = Fr::from_u64(3);
        let input_claim =
            grouped_product_virtual_claim(&instruction_ra_chunks, 2, &input_point, gamma);
        let opening_inputs = vec![
            Stage6OpeningInputValue {
                symbol: "stage6.input.stage5.instruction_read_raf.InstructionRa_0",
                point: input_point.clone(),
                eval: input_claim,
            },
            Stage6OpeningInputValue {
                symbol: "stage6.input.stage5.instruction_read_raf.InstructionRa_1",
                point: input_point,
                eval: Fr::from_u64(0),
            },
            Stage6OpeningInputValue {
                symbol: "stage6.input.instruction_ra_virtual_claim",
                point: Vec::new(),
                eval: input_claim,
            },
        ];
        let prover_inputs = Stage6ProverInputs::new(&opening_inputs).with_instruction_ra_virtual(
            Stage6InstructionRaVirtualWitness {
                instruction_ra_chunks: &instruction_ra_chunks,
                virtual_count: 2,
            },
        );
        let mut prover = Stage6ProverKernelExecutor::new(prover_inputs);
        let mut prover_transcript = Blake2bTranscript::<Fr>::new(b"stage6_test");
        let mut artifacts = execute_stage6_program(
            &INSTRUCTION_RA_PROGRAM,
            Stage6ExecutionMode::Prover,
            &mut prover,
            &mut prover_transcript,
        )
        .expect("instruction RA virtualization prover succeeds");

        artifacts.sumchecks[0].evals[0].value += Fr::from_u64(1);
        let proof = Stage6Proof {
            sumchecks: artifacts.sumchecks,
        };
        let mut verifier = Stage6ProofCarryingKernelExecutor::new(&proof, &opening_inputs);
        let mut verifier_transcript = Blake2bTranscript::<Fr>::new(b"stage6_test");
        let error = execute_stage6_program(
            &INSTRUCTION_RA_PROGRAM,
            Stage6ExecutionMode::Verifier,
            &mut verifier,
            &mut verifier_transcript,
        )
        .expect_err("tampered instruction RA eval is rejected");

        assert_eq!(
            error,
            Stage6KernelError::InvalidProof {
                driver: "stage6.sumcheck",
                reason: "batched output claim mismatch"
            }
        );
    }

    fn replay_proof(input_claim: Fr) -> (Stage6Proof<Fr>, Vec<Stage6OpeningInputValue<Fr>>) {
        let opening_inputs = vec![Stage6OpeningInputValue {
            symbol: "stage6.input.claim",
            point: Vec::new(),
            eval: input_claim,
        }];
        let mut transcript = Blake2bTranscript::<Fr>::new(b"stage6_test");
        append_labeled_scalar(&mut transcript, "sumcheck_claim", &input_claim);
        let batching_coeff = transcript.challenge_vector(1)[0];
        let claimed_sum = input_claim * batching_coeff;
        let round_poly = UnivariatePoly::new(vec![claimed_sum, -claimed_sum]);
        append_compressed_univariate_poly(&mut transcript, "sumcheck_poly", &round_poly);
        let point = vec![transcript.challenge()];
        let proof = Stage6Proof {
            sumchecks: vec![Stage6SumcheckOutput {
                driver: "stage6.sumcheck",
                point,
                evals: Vec::new(),
                proof: jolt_sumcheck::SumcheckProof {
                    round_polynomials: vec![round_poly],
                },
            }],
        };
        (proof, opening_inputs)
    }

    fn bytecode_entries() -> [Stage6BytecodeEntry<Fr>; 4] {
        [
            bytecode_entry(0, 3, &[0, 5, 12], Some(0), Some(1), Some(2), Some(0)),
            bytecode_entry(1, 5, &[1, 6], Some(1), None, Some(3), Some(1)),
            bytecode_entry(2, 7, &[2, 7, 10], Some(2), Some(0), None, None),
            bytecode_entry(3, 11, &[3, 8, 13], None, Some(2), Some(1), Some(0)),
        ]
    }

    fn bytecode_entry(
        address: u64,
        imm: u64,
        flags: &[usize],
        rd: Option<usize>,
        rs1: Option<usize>,
        rs2: Option<usize>,
        lookup_table: Option<usize>,
    ) -> Stage6BytecodeEntry<Fr> {
        let mut circuit_flags = [false; 14];
        for &flag in flags {
            circuit_flags[flag] = true;
        }
        Stage6BytecodeEntry {
            address: Fr::from_u64(address),
            imm: Fr::from_u64(imm),
            circuit_flags,
            rd,
            rs1,
            rs2,
            lookup_table,
            is_interleaved: address % 2 == 0,
            is_branch: address == 1,
            left_is_rs1: rs1.is_some(),
            left_is_pc: address == 2,
            right_is_rs2: rs2.is_some(),
            right_is_imm: address == 3,
            is_noop: address == 0,
        }
    }

    fn bytecode_opening_inputs() -> Vec<Stage6OpeningInputValue<Fr>> {
        vec![
            Stage6OpeningInputValue {
                symbol: "stage6.input.stage1.Imm",
                point: frs(&[17]),
                eval: Fr::from_u64(0),
            },
            Stage6OpeningInputValue {
                symbol: "stage6.input.stage2.OpFlagJump",
                point: frs(&[19]),
                eval: Fr::from_u64(0),
            },
            Stage6OpeningInputValue {
                symbol: "stage6.input.stage3.spartan_shift.UnexpandedPC",
                point: frs(&[23]),
                eval: Fr::from_u64(0),
            },
            Stage6OpeningInputValue {
                symbol: "stage6.input.stage4.Rs1Ra",
                point: frs(&[2, 3, 29]),
                eval: Fr::from_u64(0),
            },
            Stage6OpeningInputValue {
                symbol: "stage6.input.stage5.registers_val_evaluation.RdWa",
                point: frs(&[5, 7, 31]),
                eval: Fr::from_u64(0),
            },
        ]
    }

    fn bytecode_read_raf_claim(
        data: Stage6BytecodeReadRafData<'_, Fr>,
        bytecode_ra_chunks: &[&[Fr]],
        opening_inputs: &[Stage6OpeningInputValue<Fr>],
    ) -> Fr {
        let mut store = Stage6ValueStore::with_opening_inputs(opening_inputs);
        store.seed_constants(&BYTECODE_PROGRAM);
        let log_k = 2;
        let log_t = 1;
        let domain_len = 1 << (log_k + log_t);
        let weighted =
            bytecode_weighted_value_factor(data, &store, log_k, log_t, domain_len).unwrap();
        let ra_factors =
            expanded_bytecode_ra_factors(bytecode_ra_chunks, &[1, 1], log_k, log_t, domain_len)
                .unwrap();
        (0..domain_len)
            .map(|row| weighted[row] * ra_factors.iter().map(|factor| factor[row]).product::<Fr>())
            .sum()
    }

    fn frs(values: &[u64]) -> Vec<Fr> {
        values.iter().map(|value| Fr::from_u64(*value)).collect()
    }

    fn multilinear_eval(values: &[Fr], point: &[Fr]) -> Fr {
        EqPolynomial::<Fr>::evals(point, None)
            .into_iter()
            .zip(values)
            .map(|(eq, value)| eq * *value)
            .sum()
    }

    fn product_virtual_claim(chunks: &[&[Fr]], point: &[Fr]) -> Fr {
        let eq = EqPolynomial::<Fr>::evals(point, None);
        (0..eq.len())
            .map(|row| eq[row] * chunks.iter().map(|chunk| chunk[row]).product::<Fr>())
            .sum()
    }

    fn grouped_product_virtual_claim(
        chunks: &[&[Fr]],
        virtual_count: usize,
        point: &[Fr],
        gamma: Fr,
    ) -> Fr {
        let eq = EqPolynomial::<Fr>::evals(point, None);
        let chunks_per_virtual = chunks.len() / virtual_count;
        (0..eq.len())
            .map(|row| {
                let mut gamma_power = Fr::from_u64(1);
                let mut row_value = Fr::from_u64(0);
                for group in chunks.chunks(chunks_per_virtual) {
                    row_value += gamma_power * group.iter().map(|chunk| chunk[row]).product::<Fr>();
                    gamma_power *= gamma;
                }
                eq[row] * row_value
            })
            .sum()
    }

    fn named_eval_values(evals: &[Stage6NamedEval<Fr>]) -> Vec<(&'static str, &'static str, Fr)> {
        evals
            .iter()
            .map(|eval| (eval.name, eval.oracle, eval.value))
            .collect()
    }
}
