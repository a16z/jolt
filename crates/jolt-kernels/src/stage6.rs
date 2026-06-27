//! Stage 6 coarse-kernel ABI used by Bolt-generated Jolt prover code.

#[cfg(feature = "cuda")]
mod cuda;

use std::fmt::{self, Display, Formatter};
use std::{borrow::Cow, error::Error};

use crate::dense::{
    bind_dense_evals_reuse, bind_dense_evals_reuse_serial, DENSE_BIND_PAR_THRESHOLD,
};
use jolt_field::{Field, FieldAccumulator, FieldScalarAccumulator};
use jolt_poly::{
    BindingOrder, EqPolynomial, ExpandingTable, GruenSplitEqPolynomial, UnivariatePoly,
};
use jolt_transcript::{Label, LabelWithCount, Transcript};
pub use jolt_witness::Stage6WitnessParams;
use jolt_witness::{
    stage6_witness_polynomials, CycleInput, Stage6BooleanityRow, Stage6OpeningInputRef,
    Stage6WitnessInputs, Stage6WitnessPolynomials, Stage6WitnessSlices,
};
use rayon::prelude::*;

fn trace_stage6_inner_spans() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("JOLT_STAGE6_TRACE_INSTANCES").is_some())
}

pub use crate::stage4::{
    Stage4ChallengeVector as Stage6ChallengeVector,
    Stage4ExecutionArtifacts as Stage6ExecutionArtifacts,
    Stage4ExecutionMode as Stage6ExecutionMode, Stage4FieldConstantPlan as Stage6FieldConstantPlan,
    Stage4FieldExprPlan as Stage6FieldExprPlan, Stage4NamedEval as Stage6NamedEval,
    Stage4OpeningBatchPlan as Stage6OpeningBatchPlan,
    Stage4OpeningClaimEqualityPlan as Stage6OpeningClaimEqualityPlan,
    Stage4OpeningClaimPlan as Stage6OpeningClaimPlan,
    Stage4OpeningClaimValue as Stage6OpeningClaimValue,
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
    pub index_chunks: Option<&'a [&'a [Option<u8>]]>,
    pub row_indices: Option<&'a [Stage6BooleanityRow]>,
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

impl<F: Field> From<jolt_witness::Stage6BytecodeEntry<F>> for Stage6BytecodeEntry<F> {
    fn from(entry: jolt_witness::Stage6BytecodeEntry<F>) -> Self {
        Self {
            address: entry.address,
            imm: entry.imm,
            circuit_flags: entry.circuit_flags,
            rd: entry.rd,
            rs1: entry.rs1,
            rs2: entry.rs2,
            lookup_table: entry.lookup_table,
            is_interleaved: entry.is_interleaved,
            is_branch: entry.is_branch,
            left_is_rs1: entry.left_is_rs1,
            left_is_pc: entry.left_is_pc,
            right_is_rs2: entry.right_is_rs2,
            right_is_imm: entry.right_is_imm,
            is_noop: entry.is_noop,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Stage6BytecodeReadRafDataStorage<F: Field> {
    entries: Vec<Stage6BytecodeEntry<F>>,
    entry_bytecode_index: usize,
    num_lookup_tables: usize,
}

impl<F: Field> Stage6BytecodeReadRafDataStorage<F> {
    pub fn from_witness_entries(
        entries: &[jolt_witness::Stage6BytecodeEntry<F>],
        entry_bytecode_index: usize,
        num_lookup_tables: usize,
    ) -> Self {
        Self {
            entries: entries.iter().copied().map(Into::into).collect(),
            entry_bytecode_index,
            num_lookup_tables,
        }
    }

    pub fn as_input(&self) -> Stage6BytecodeReadRafData<'_, F> {
        Stage6BytecodeReadRafData {
            entries: &self.entries,
            entry_bytecode_index: self.entry_bytecode_index,
            num_lookup_tables: self.num_lookup_tables,
        }
    }
}

#[derive(Clone, Copy)]
pub struct Stage6BytecodeReadRafWitness<'a, F: Field> {
    pub data: Stage6BytecodeReadRafData<'a, F>,
    pub bytecode_ra_chunks: &'a [&'a [F]],
    pub bytecode_ra_chunk_lens: Option<&'a [usize]>,
    pub bytecode_ra_index_chunks: Option<&'a [&'a [Option<u8>]]>,
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
    pub instruction_ra_index_chunks: Option<&'a [&'a [Option<u8>]]>,
    pub virtual_count: usize,
}

pub fn stage6_witness_from_opening_inputs<F: Field>(
    params: Stage6WitnessParams,
    cycle_inputs: &[CycleInput],
    opening_inputs: &[Stage6OpeningInputValue<F>],
) -> Stage6WitnessPolynomials<F> {
    let opening_refs = opening_inputs
        .iter()
        .map(|input| Stage6OpeningInputRef {
            symbol: input.symbol,
            point: input.point.as_slice(),
        })
        .collect::<Vec<_>>();
    stage6_witness_polynomials(Stage6WitnessInputs {
        params,
        cycle_inputs,
        opening_inputs: &opening_refs,
    })
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

    pub fn with_stage6_witness(
        self,
        bytecode_data: Stage6BytecodeReadRafData<'a, F>,
        witness: &'a Stage6WitnessPolynomials<F>,
        slices: &'a Stage6WitnessSlices<'a, F>,
        instruction_ra_virtual_count: usize,
    ) -> Self {
        self.with_bytecode_read_raf(Stage6BytecodeReadRafWitness {
            data: bytecode_data,
            bytecode_ra_chunks: &slices.bytecode_ra_read_raf_chunks,
            bytecode_ra_chunk_lens: Some(&slices.bytecode_ra_read_raf_chunk_lens),
            bytecode_ra_index_chunks: Some(&slices.bytecode_ra_index_chunks),
        })
        .with_booleanity(Stage6BooleanityWitness {
            chunks: &slices.booleanity_chunks,
            index_chunks: Some(&slices.booleanity_index_chunks),
            row_indices: Some(slices.booleanity_rows),
        })
        .with_hamming_booleanity(Stage6HammingBooleanityWitness {
            hamming_weight: &witness.hamming_weight,
        })
        .with_ram_ra_virtual(Stage6RamRaVirtualWitness {
            ram_ra_chunks: &slices.ram_ra_virtual_chunks,
        })
        .with_instruction_ra_virtual(Stage6InstructionRaVirtualWitness {
            instruction_ra_chunks: &slices.instruction_ra_virtual_chunks,
            instruction_ra_index_chunks: Some(&slices.instruction_ra_index_chunks),
            virtual_count: instruction_ra_virtual_count,
        })
        .with_inc_claim_reduction(Stage6IncClaimReductionWitness {
            ram_inc: &witness.ram_inc,
            rd_inc: &witness.rd_inc,
        })
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

    fn opening_claim_values(
        &self,
        _program: &'static Stage6CpuProgramPlan,
    ) -> Result<Vec<Stage6OpeningClaimValue<F>>, Stage6KernelError> {
        Ok(Vec::new())
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

    fn opening_claim_values(
        &self,
        program: &'static Stage6CpuProgramPlan,
    ) -> Result<Vec<Stage6OpeningClaimValue<F>>, Stage6KernelError> {
        self.value_store(program)?.opening_claim_values(program)
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

    fn opening_claim_values(
        &self,
        program: &'static Stage6CpuProgramPlan,
    ) -> Result<Vec<Stage6OpeningClaimValue<F>>, Stage6KernelError> {
        self.value_store(program)?.opening_claim_values(program)
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
                "as_is" => {}
                "reverse" => point.reverse(),
                "bytecode_read_raf" => point = normalize_bytecode_read_raf_point(program, &point)?,
                "stage6_booleanity" => point = normalize_stage6_booleanity_point(program, &point)?,
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

    fn opening_claim_values(
        mut self,
        program: &'static Stage6CpuProgramPlan,
    ) -> Result<Vec<Stage6OpeningClaimValue<F>>, Stage6KernelError> {
        let _ = self.evaluate_available_points(program)?;
        let _ = self.evaluate_available_field_exprs(program)?;
        program
            .opening_claims
            .iter()
            .map(|claim| {
                Ok(Stage6OpeningClaimValue {
                    symbol: claim.symbol,
                    oracle: claim.oracle,
                    domain: claim.domain,
                    claim_kind: claim.claim_kind,
                    point: self.point(claim.point_source)?.to_vec(),
                    eval: self.scalar(claim.eval_source)?,
                })
            })
            .collect()
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
fn prove_batched_stage6<'a, F, T>(
    context: Stage6KernelContext<'_>,
    inputs: &'a Stage6ProverInputs<'a, F>,
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
    let trace_instances = std::env::var_os("JOLT_STAGE6_TRACE_INSTANCES").is_some();
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
        let state = if trace_instances {
            let _span = tracing::info_span!(
                "Stage6::instance.init",
                relation = relation.symbol(),
                claim = claim.symbol
            )
            .entered();
            Stage6ProverInstanceState::new(
                context.program,
                claim,
                inputs,
                &store,
                active_scale,
                context.kernel.backend,
            )?
        } else {
            Stage6ProverInstanceState::new(
                context.program,
                claim,
                inputs,
                &store,
                active_scale,
                context.kernel.backend,
            )?
        };
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
                let poly = if trace_instances {
                    let _span = tracing::info_span!(
                        "Stage6::instance.round_poly",
                        relation = instance.relation.symbol(),
                        claim = instance.claim.symbol,
                        round = round
                    )
                    .entered();
                    instance
                        .state
                        .round_poly(instance.previous_claim, instance.relation)?
                } else {
                    instance
                        .state
                        .round_poly(instance.previous_claim, instance.relation)?
                };
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
        check_round_claim(
            &batched_poly,
            batched_claim,
            context.driver.symbol,
            "batched round claim mismatch",
        )?;
        append_compressed_univariate_poly(transcript, context.driver.round_label, &batched_poly);
        let challenge = transcript.challenge();
        point.push(challenge);
        batched_claim = batched_poly.evaluate(challenge);
        for (index, (instance, poly)) in instances.iter_mut().zip(individual_polys).enumerate() {
            instance.previous_claim = poly.evaluate(challenge);
            if instance.is_active(round) {
                let bind_start = timing_enabled.then(std::time::Instant::now);
                if trace_instances {
                    let _span = tracing::info_span!(
                        "Stage6::instance.bind",
                        relation = instance.relation.symbol(),
                        claim = instance.claim.symbol,
                        round = round
                    )
                    .entered();
                    instance.state.ingest_challenge(challenge);
                } else {
                    instance.state.ingest_challenge(challenge);
                }
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
    let opening_claims = append_opening_claims(context.program, &mut store, transcript, &evals)?;
    if timing_enabled {
        for (relation, init_nanos, round_nanos, bind_nanos) in timing_stats {
            tracing::info!(
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
        opening_claims,
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
        opening_claims: Vec::new(),
        proof: proof.proof.clone(),
    };
    store.observe_sumcheck_output(context.program, &output)?;
    let opening_claims =
        append_opening_claims(context.program, &mut store, transcript, &output.evals)?;
    let output = Stage6SumcheckOutput {
        opening_claims,
        ..output
    };
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
    let mut verifier_point = combined_r[..log_k_chunk].to_vec();
    verifier_point.reverse();
    verifier_point.extend(combined_r[log_k_chunk..].iter().rev().copied());
    let eq_eval = EqPolynomial::<F>::mle(local_point, &verifier_point);

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
    state: Stage6ProverInstanceState<'a, F>,
}

impl<F: Field> Stage6BatchedInstance<'_, F> {
    fn is_active(&self, round: usize) -> bool {
        round >= self.offset && round < self.offset + self.claim.num_rounds
    }
}

enum Stage6ProverInstanceState<'a, F: Field> {
    Booleanity(BooleanityStage6State<F>),
    CoreBooleanity(Box<CoreBooleanityStage6State<'a, F>>),
    BytecodeReadRaf(Box<BytecodeReadRafStage6State<F>>),
    HammingBooleanity(Box<HammingBooleanityStage6State<F>>),
    RamRaVirtual(InstructionRaVirtualStage6State<'a, F>),
    InstructionRaVirtual(InstructionRaVirtualStage6State<'a, F>),
    IncClaimReduction(Box<IncClaimReductionStage6State<F>>),
    Dense(DenseStage6State<F>),
}

impl<'a, F: Field> Stage6ProverInstanceState<'a, F> {
    fn new(
        program: &'static Stage6CpuProgramPlan,
        claim: &Stage6SumcheckClaimPlan,
        inputs: &'a Stage6ProverInputs<'a, F>,
        store: &Stage6ValueStore<F>,
        active_scale: F,
        backend: &'static str,
    ) -> Result<Self, Stage6KernelError> {
        match claim_relation(program, claim)? {
            Stage6Relation::BytecodeReadRaf => {
                bytecode_read_raf_state(program, claim, inputs, store, active_scale)
            }
            Stage6Relation::Booleanity => {
                booleanity_state(program, claim, inputs, store, active_scale)
            }
            Stage6Relation::HammingBooleanity => {
                hamming_booleanity_state(program, claim, inputs, store, active_scale, backend)
                    .map(|state| Self::HammingBooleanity(Box::new(state)))
            }
            Stage6Relation::IncClaimReduction => {
                inc_claim_reduction_state(program, claim, inputs, store, active_scale, backend)
                    .map(|state| Self::IncClaimReduction(Box::new(state)))
            }
            Stage6Relation::RamRaVirtual => {
                ram_ra_virtual_state(program, claim, inputs, store, active_scale, backend)
                    .map(Self::RamRaVirtual)
            }
            Stage6Relation::InstructionRaVirtual => {
                instruction_ra_virtual_state(program, claim, inputs, store, active_scale, backend)
                    .map(Self::InstructionRaVirtual)
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
            Self::CoreBooleanity(state) => state.round_poly(previous_claim, relation),
            Self::BytecodeReadRaf(state) => state.round_poly(previous_claim, relation),
            Self::HammingBooleanity(state) => state.round_poly(previous_claim, relation),
            Self::RamRaVirtual(state) => state.round_poly(previous_claim, relation),
            Self::InstructionRaVirtual(state) => state.round_poly(previous_claim, relation),
            Self::IncClaimReduction(state) => state.round_poly(previous_claim, relation),
            Self::Dense(state) => state.round_poly(previous_claim, relation),
        }
    }

    fn ingest_challenge(&mut self, challenge: F) {
        match self {
            Self::Booleanity(state) => state.bind(challenge),
            Self::CoreBooleanity(state) => state.bind(challenge),
            Self::BytecodeReadRaf(state) => state.bind(challenge),
            Self::HammingBooleanity(state) => state.bind(challenge),
            Self::RamRaVirtual(state) => state.bind(challenge),
            Self::InstructionRaVirtual(state) => state.bind(challenge),
            Self::IncClaimReduction(state) => state.bind(challenge),
            Self::Dense(state) => state.bind(challenge),
        }
    }

    fn final_relation_eval(&self, relation: Stage6Relation) -> Result<F, Stage6KernelError> {
        match self {
            Self::Booleanity(state) => state.final_relation_eval(relation),
            Self::CoreBooleanity(state) => state.final_relation_eval(relation),
            Self::BytecodeReadRaf(state) => state.final_relation_eval(relation),
            Self::HammingBooleanity(state) => state.final_relation_eval(relation),
            Self::RamRaVirtual(state) => state.final_relation_eval(relation),
            Self::InstructionRaVirtual(state) => state.final_relation_eval(relation),
            Self::IncClaimReduction(state) => state.final_relation_eval(relation),
            Self::Dense(state) => state.final_relation_eval(relation),
        }
    }

    fn final_evals(
        &self,
        relation: Stage6Relation,
    ) -> Result<Vec<Stage6NamedEval<F>>, Stage6KernelError> {
        match self {
            Self::Booleanity(state) => state.final_evals(relation),
            Self::CoreBooleanity(state) => state.final_evals(relation),
            Self::BytecodeReadRaf(state) => state.final_evals(relation),
            Self::HammingBooleanity(state) => state.final_evals(relation),
            Self::RamRaVirtual(state) => state.final_evals(relation),
            Self::InstructionRaVirtual(state) => state.final_evals(relation),
            Self::IncClaimReduction(state) => state.final_evals(relation),
            Self::Dense(state) => state.final_evals(relation),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BytecodeReadRafPhase {
    Address,
    Cycle,
}

enum BytecodeReadRafCycleFactors<F: Field> {
    Empty,
    SparseRound1 {
        tables: Vec<Vec<F>>,
        indices: Vec<Vec<Option<u8>>>,
    },
    SparseRound2 {
        tables_0: Vec<Vec<F>>,
        tables_1: Vec<Vec<F>>,
        indices: Vec<Vec<Option<u8>>>,
    },
    SparseRound3 {
        tables_00: Vec<Vec<F>>,
        tables_01: Vec<Vec<F>>,
        tables_10: Vec<Vec<F>>,
        tables_11: Vec<Vec<F>>,
        indices: Vec<Vec<Option<u8>>>,
    },
    Bound {
        chunks: Vec<Vec<F>>,
        scratch: Vec<Vec<F>>,
    },
}

impl<F: Field> BytecodeReadRafCycleFactors<F> {
    fn empty() -> Self {
        Self::Empty
    }

    fn sparse(indices: Vec<Vec<Option<u8>>>, chunk_lens: &[usize], address_point: &[F]) -> Self {
        let tables = chunk_lens
            .iter()
            .scan(0usize, |offset, &chunk_len| {
                let start = *offset;
                *offset += chunk_len;
                Some(EqPolynomial::<F>::evals(
                    &address_point[start..start + chunk_len],
                    None,
                ))
            })
            .collect();
        Self::SparseRound1 { tables, indices }
    }

    fn dense(chunks: Vec<Vec<F>>) -> Self {
        let scratch = (0..chunks.len()).map(|_| Vec::new()).collect();
        Self::Bound { chunks, scratch }
    }

    fn len(&self) -> usize {
        match self {
            Self::Empty => 0,
            Self::SparseRound1 { tables, .. }
            | Self::SparseRound2 {
                tables_0: tables, ..
            }
            | Self::SparseRound3 {
                tables_00: tables, ..
            } => tables.len(),
            Self::Bound { chunks, .. } => chunks.len(),
        }
    }

    fn current_len(&self) -> usize {
        match self {
            Self::Empty => 0,
            Self::SparseRound1 { indices, .. } => indices.first().map_or(0, Vec::len),
            Self::SparseRound2 { indices, .. } => {
                indices.first().map_or(0, |chunk| chunk.len() / 2)
            }
            Self::SparseRound3 { indices, .. } => {
                indices.first().map_or(0, |chunk| chunk.len() / 4)
            }
            Self::Bound { chunks, .. } => chunks.first().map_or(0, Vec::len),
        }
    }

    fn factor_eval(&self, chunk: usize) -> Option<F> {
        (chunk < self.len()).then(|| self.get(chunk, 0))
    }

    fn get_pair(&self, chunk: usize, row: usize) -> (F, F) {
        (self.get(chunk, 2 * row), self.get(chunk, 2 * row + 1))
    }

    fn get(&self, chunk: usize, index: usize) -> F {
        match self {
            Self::Empty => F::zero(),
            Self::SparseRound1 { tables, indices } => {
                indices[chunk][index].map_or(F::zero(), |value| tables[chunk][usize::from(value)])
            }
            Self::SparseRound2 {
                tables_0,
                tables_1,
                indices,
            } => {
                let source = 2 * index;
                let low = indices[chunk][source]
                    .map_or(F::zero(), |value| tables_0[chunk][usize::from(value)]);
                let high = indices[chunk][source + 1]
                    .map_or(F::zero(), |value| tables_1[chunk][usize::from(value)]);
                low + high
            }
            Self::SparseRound3 {
                tables_00,
                tables_01,
                tables_10,
                tables_11,
                indices,
            } => {
                let source = 4 * index;
                let h_00 = indices[chunk][source]
                    .map_or(F::zero(), |value| tables_00[chunk][usize::from(value)]);
                let h_10 = indices[chunk][source + 1]
                    .map_or(F::zero(), |value| tables_10[chunk][usize::from(value)]);
                let h_01 = indices[chunk][source + 2]
                    .map_or(F::zero(), |value| tables_01[chunk][usize::from(value)]);
                let h_11 = indices[chunk][source + 3]
                    .map_or(F::zero(), |value| tables_11[chunk][usize::from(value)]);
                h_00 + h_10 + h_01 + h_11
            }
            Self::Bound { chunks, .. } => chunks[chunk][index],
        }
    }

    fn bind(&mut self, challenge: F) {
        let one_minus = F::one() - challenge;
        match std::mem::replace(self, Self::Empty) {
            Self::Empty => *self = Self::Empty,
            Self::SparseRound1 { tables, indices } => {
                let (tables_0, tables_1) = rayon::join(
                    || scale_bytecode_read_raf_tables(&tables, one_minus),
                    || scale_bytecode_read_raf_tables(&tables, challenge),
                );
                *self = Self::SparseRound2 {
                    tables_0,
                    tables_1,
                    indices,
                };
            }
            Self::SparseRound2 {
                tables_0,
                tables_1,
                indices,
            } => {
                let (tables_00, tables_01) = rayon::join(
                    || scale_bytecode_read_raf_tables(&tables_0, one_minus),
                    || scale_bytecode_read_raf_tables(&tables_0, challenge),
                );
                let (tables_10, tables_11) = rayon::join(
                    || scale_bytecode_read_raf_tables(&tables_1, one_minus),
                    || scale_bytecode_read_raf_tables(&tables_1, challenge),
                );
                *self = Self::SparseRound3 {
                    tables_00,
                    tables_01,
                    tables_10,
                    tables_11,
                    indices,
                };
            }
            Self::SparseRound3 {
                tables_00,
                tables_01,
                tables_10,
                tables_11,
                indices,
            } => {
                let (tables_000, tables_001) = rayon::join(
                    || scale_bytecode_read_raf_tables(&tables_00, one_minus),
                    || scale_bytecode_read_raf_tables(&tables_00, challenge),
                );
                let (tables_010, tables_011) = rayon::join(
                    || scale_bytecode_read_raf_tables(&tables_01, one_minus),
                    || scale_bytecode_read_raf_tables(&tables_01, challenge),
                );
                let (tables_100, tables_101) = rayon::join(
                    || scale_bytecode_read_raf_tables(&tables_10, one_minus),
                    || scale_bytecode_read_raf_tables(&tables_10, challenge),
                );
                let (tables_110, tables_111) = rayon::join(
                    || scale_bytecode_read_raf_tables(&tables_11, one_minus),
                    || scale_bytecode_read_raf_tables(&tables_11, challenge),
                );
                let table_groups = [
                    &tables_000,
                    &tables_100,
                    &tables_010,
                    &tables_110,
                    &tables_001,
                    &tables_101,
                    &tables_011,
                    &tables_111,
                ];
                let new_len = indices.first().map_or(0, |chunk| chunk.len() / 8);
                let materialize_chunk_size = 1 << 16;
                let chunks = (0..tables_000.len())
                    .into_par_iter()
                    .map(|chunk| {
                        let mut values = vec![F::zero(); new_len];
                        values
                            .par_chunks_mut(materialize_chunk_size)
                            .enumerate()
                            .for_each(|(chunk_index, values_chunk)| {
                                let start = chunk_index * materialize_chunk_size;
                                for (local_index, value) in values_chunk.iter_mut().enumerate() {
                                    let index = start + local_index;
                                    let source = 8 * index;
                                    let h_000 = indices[chunk][source].map_or(F::zero(), |value| {
                                        table_groups[0][chunk][usize::from(value)]
                                    });
                                    let h_100 = indices[chunk][source + 1]
                                        .map_or(F::zero(), |value| {
                                            table_groups[1][chunk][usize::from(value)]
                                        });
                                    let h_010 = indices[chunk][source + 2]
                                        .map_or(F::zero(), |value| {
                                            table_groups[2][chunk][usize::from(value)]
                                        });
                                    let h_110 = indices[chunk][source + 3]
                                        .map_or(F::zero(), |value| {
                                            table_groups[3][chunk][usize::from(value)]
                                        });
                                    let h_001 = indices[chunk][source + 4]
                                        .map_or(F::zero(), |value| {
                                            table_groups[4][chunk][usize::from(value)]
                                        });
                                    let h_101 = indices[chunk][source + 5]
                                        .map_or(F::zero(), |value| {
                                            table_groups[5][chunk][usize::from(value)]
                                        });
                                    let h_011 = indices[chunk][source + 6]
                                        .map_or(F::zero(), |value| {
                                            table_groups[6][chunk][usize::from(value)]
                                        });
                                    let h_111 = indices[chunk][source + 7]
                                        .map_or(F::zero(), |value| {
                                            table_groups[7][chunk][usize::from(value)]
                                        });
                                    *value = h_000
                                        + h_010
                                        + h_100
                                        + h_110
                                        + h_001
                                        + h_011
                                        + h_101
                                        + h_111;
                                }
                            });
                        values
                    })
                    .collect();
                *self = Self::dense(chunks);
            }
            Self::Bound {
                mut chunks,
                mut scratch,
            } => {
                if chunks.first().map_or(0, Vec::len) / 2 >= DENSE_BIND_PAR_THRESHOLD {
                    chunks.par_iter_mut().zip(scratch.par_iter_mut()).for_each(
                        |(chunk, scratch)| {
                            bind_dense_evals_reuse_serial(chunk, scratch, challenge);
                        },
                    );
                } else {
                    for (chunk, scratch) in chunks.iter_mut().zip(&mut scratch) {
                        bind_dense_evals_reuse(chunk, scratch, challenge);
                    }
                }
                *self = Self::Bound { chunks, scratch };
            }
        }
    }
}

fn scale_bytecode_read_raf_tables<F: Field>(tables: &[Vec<F>], scalar: F) -> Vec<Vec<F>> {
    tables
        .par_iter()
        .map(|table| table.iter().map(|value| *value * scalar).collect())
        .collect()
}

struct BytecodeReadRafStage6State<F: Field> {
    log_k: usize,
    log_t: usize,
    chunk_lens: Vec<usize>,
    bytecode_ra_chunks: Vec<Vec<F>>,
    bytecode_ra_indices: Option<Vec<Vec<Option<u8>>>>,
    stage_factors: [Vec<F>; BYTECODE_READ_RAF_STAGE_COUNT],
    stage_factor_scratch: [Vec<F>; BYTECODE_READ_RAF_STAGE_COUNT],
    stage_values: [Vec<F>; BYTECODE_READ_RAF_STAGE_COUNT],
    stage_value_scratch: [Vec<F>; BYTECODE_READ_RAF_STAGE_COUNT],
    entry_trace: Vec<F>,
    entry_trace_scratch: Vec<F>,
    entry_expected: Vec<F>,
    entry_expected_scratch: Vec<F>,
    address_challenges: Vec<F>,
    cycle_factors: BytecodeReadRafCycleFactors<F>,
    cycle_eqs: [Vec<F>; BYTECODE_READ_RAF_STAGE_COUNT],
    cycle_eq_scratch: [Vec<F>; BYTECODE_READ_RAF_STAGE_COUNT],
    cycle_entry_eq: Vec<F>,
    cycle_entry_eq_scratch: Vec<F>,
    cycle_combined_eq: Vec<F>,
    cycle_combined_eq_scratch: Vec<F>,
    bound_stage_values: Option<[F; BYTECODE_READ_RAF_STAGE_COUNT]>,
    bound_entry_expected: Option<F>,
    outputs: Vec<FactorOutput>,
    gamma_powers: [F; 8],
    active_scale: F,
    degree_bound: usize,
    phase: BytecodeReadRafPhase,
}

impl<F: Field> BytecodeReadRafStage6State<F> {
    #[expect(clippy::too_many_arguments)]
    fn new(
        data: Stage6BytecodeReadRafData<'_, F>,
        bytecode_ra_chunks: &[&[F]],
        bytecode_ra_index_chunks: Option<&[&[Option<u8>]]>,
        bytecode_cycle_indices: Vec<usize>,
        chunk_lens: Vec<usize>,
        store: &Stage6ValueStore<F>,
        log_k: usize,
        log_t: usize,
        active_scale: F,
        degree_bound: usize,
        outputs: Vec<FactorOutput>,
    ) -> Result<Self, Stage6KernelError> {
        if degree_bound < 2 || degree_bound < chunk_lens.len() + 1 {
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
        let bytecode_ra_indices = match bytecode_ra_index_chunks {
            Some(chunks) if !chunks.is_empty() => {
                validate_bytecode_ra_index_chunks(chunks, &chunk_lens, log_t)?;
                Some(chunks.iter().map(|chunk| (*chunk).to_vec()).collect())
            }
            _ => None,
        };

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
            bytecode_ra_indices,
            stage_factors,
            stage_factor_scratch: std::array::from_fn(|_| Vec::new()),
            stage_values,
            stage_value_scratch: std::array::from_fn(|_| Vec::new()),
            entry_trace,
            entry_trace_scratch: Vec::new(),
            entry_expected,
            entry_expected_scratch: Vec::new(),
            address_challenges: Vec::with_capacity(log_k),
            cycle_factors: BytecodeReadRafCycleFactors::empty(),
            cycle_eqs,
            cycle_eq_scratch: std::array::from_fn(|_| Vec::new()),
            cycle_entry_eq,
            cycle_entry_eq_scratch: Vec::new(),
            cycle_combined_eq: Vec::new(),
            cycle_combined_eq_scratch: Vec::new(),
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
            BytecodeReadRafPhase::Address => self.address_round_poly(previous_claim, relation)?,
            BytecodeReadRafPhase::Cycle => self.cycle_round_poly(previous_claim, relation)?,
        };
        check_round_claim(
            &poly,
            previous_claim,
            relation.symbol(),
            "stage6 relation input claim mismatch",
        )?;
        Ok(poly)
    }

    fn address_round_poly(
        &self,
        previous_claim: F,
        relation: Stage6Relation,
    ) -> Result<UnivariatePoly<F>, Stage6KernelError> {
        let first_len = self.stage_values[0].len();
        if first_len == 0 || !first_len.is_power_of_two() {
            return Err(Stage6KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "bytecode read RAF address phase has invalid length",
            });
        }
        let eval_count = 2;
        let eval_accs = if first_len / 2 >= DENSE_BIND_PAR_THRESHOLD {
            (0..first_len / 2)
                .into_par_iter()
                .fold(
                    || vec![F::Accumulator::default(); eval_count],
                    |mut row_evals, row| {
                        self.accumulate_address_row(row, &mut row_evals);
                        row_evals
                    },
                )
                .reduce(
                    || vec![F::Accumulator::default(); eval_count],
                    |mut left, right| {
                        for (left, right) in left.iter_mut().zip(right) {
                            left.merge(right);
                        }
                        left
                    },
                )
        } else {
            let mut evals = vec![F::Accumulator::default(); eval_count];
            for row in 0..first_len / 2 {
                self.accumulate_address_row(row, &mut evals);
            }
            evals
        };
        let mut evals = eval_accs
            .into_iter()
            .map(FieldAccumulator::reduce)
            .collect::<Vec<_>>();
        for eval in &mut evals {
            *eval *= self.active_scale;
        }
        Ok(UnivariatePoly::from_evals_and_hint(previous_claim, &evals))
    }

    fn accumulate_address_row(&self, row: usize, evals: &mut [F::Accumulator]) {
        for (point_index, eval) in evals.iter_mut().enumerate() {
            let point = if point_index == 0 {
                F::zero()
            } else {
                F::from_u64((point_index + 1) as u64)
            };
            for stage in 0..BYTECODE_READ_RAF_STAGE_COUNT {
                let trace_eval = pair_linear_eval(&self.stage_factors[stage], row, point);
                let value_eval = pair_linear_eval(&self.stage_values[stage], row, point);
                eval.fmadd(trace_eval, self.gamma_powers[stage] * value_eval);
            }
            let entry_trace = pair_linear_eval(&self.entry_trace, row, point);
            let entry_expected = pair_linear_eval(&self.entry_expected, row, point);
            eval.fmadd(entry_trace, self.gamma_powers[7] * entry_expected);
        }
    }

    fn cycle_round_poly(
        &self,
        previous_claim: F,
        relation: Stage6Relation,
    ) -> Result<UnivariatePoly<F>, Stage6KernelError> {
        if self.bound_stage_values.is_none() {
            return Err(Stage6KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "bytecode read RAF cycle phase missing bound values",
            });
        }
        if self.bound_entry_expected.is_none() {
            return Err(Stage6KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "bytecode read RAF cycle phase missing entry value",
            });
        }
        let first_len = self.cycle_factors.current_len();
        if first_len == 0 || !first_len.is_power_of_two() {
            return Err(Stage6KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "bytecode read RAF cycle phase has invalid length",
            });
        }
        let eval_count = self.degree_bound;
        let eval_accs = if first_len / 2 >= DENSE_BIND_PAR_THRESHOLD {
            (0..first_len / 2)
                .into_par_iter()
                .fold(
                    || vec![F::Accumulator::default(); eval_count],
                    |mut row_evals, row| {
                        self.accumulate_cycle_row(row, &mut row_evals);
                        row_evals
                    },
                )
                .reduce(
                    || vec![F::Accumulator::default(); eval_count],
                    |mut left, right| {
                        for (left, right) in left.iter_mut().zip(right) {
                            left.merge(right);
                        }
                        left
                    },
                )
        } else {
            let mut evals = vec![F::Accumulator::default(); eval_count];
            for row in 0..first_len / 2 {
                self.accumulate_cycle_row(row, &mut evals);
            }
            evals
        };
        let mut evals = eval_accs
            .into_iter()
            .map(FieldAccumulator::reduce)
            .collect::<Vec<_>>();
        for eval in &mut evals {
            *eval *= self.active_scale;
        }
        Ok(UnivariatePoly::from_evals_and_hint(previous_claim, &evals))
    }

    fn accumulate_cycle_row(&self, row: usize, evals: &mut [F::Accumulator]) {
        const MAX_BYTECODE_CYCLE_EVALS: usize = 8;
        if evals.len() <= MAX_BYTECODE_CYCLE_EVALS {
            let mut ra_products = [F::one(); MAX_BYTECODE_CYCLE_EVALS];
            for chunk in 0..self.cycle_factors.len() {
                let (low, high) = self.cycle_factors.get_pair(chunk, row);
                let slope = high - low;
                for (point_index, product) in ra_products.iter_mut().take(evals.len()).enumerate() {
                    let point = if point_index == 0 {
                        F::zero()
                    } else {
                        F::from_u64((point_index + 1) as u64)
                    };
                    *product *= low + slope * point;
                }
            }

            let combined_low = self.cycle_combined_eq[2 * row];
            let combined_high = self.cycle_combined_eq[2 * row + 1];
            let combined_slope = combined_high - combined_low;
            let mut weighted_values = [F::zero(); MAX_BYTECODE_CYCLE_EVALS];
            for (point_index, weighted_value) in
                weighted_values.iter_mut().take(evals.len()).enumerate()
            {
                let point = if point_index == 0 {
                    F::zero()
                } else {
                    F::from_u64((point_index + 1) as u64)
                };
                *weighted_value = combined_low + combined_slope * point;
            }

            for ((eval, &ra_product), &weighted_value) in evals
                .iter_mut()
                .zip(ra_products.iter())
                .zip(weighted_values.iter())
            {
                eval.fmadd(ra_product, weighted_value);
            }
            return;
        }

        for (point_index, eval) in evals.iter_mut().enumerate() {
            let point = if point_index == 0 {
                F::zero()
            } else {
                F::from_u64((point_index + 1) as u64)
            };
            let mut ra_product = F::one();
            for chunk in 0..self.cycle_factors.len() {
                let (low, high) = self.cycle_factors.get_pair(chunk, row);
                ra_product *= low + (high - low) * point;
            }
            let weighted_value = pair_linear_eval(&self.cycle_combined_eq, row, point);
            eval.fmadd(ra_product, weighted_value);
        }
    }

    fn bind(&mut self, challenge: F) {
        match self.phase {
            BytecodeReadRafPhase::Address => self.bind_address(challenge),
            BytecodeReadRafPhase::Cycle => self.bind_cycle(challenge),
        }
    }

    fn bind_address(&mut self, challenge: F) {
        rayon::join(
            || {
                rayon::join(
                    || {
                        self.stage_factors
                            .par_iter_mut()
                            .zip(self.stage_factor_scratch.par_iter_mut())
                            .for_each(|(factor, scratch)| {
                                bind_dense_evals_reuse_serial(factor, scratch, challenge);
                            });
                    },
                    || {
                        self.stage_values
                            .par_iter_mut()
                            .zip(self.stage_value_scratch.par_iter_mut())
                            .for_each(|(value, scratch)| {
                                bind_dense_evals_reuse_serial(value, scratch, challenge);
                            });
                    },
                );
            },
            || {
                rayon::join(
                    || {
                        bind_dense_evals_reuse(
                            &mut self.entry_trace,
                            &mut self.entry_trace_scratch,
                            challenge,
                        );
                    },
                    || {
                        bind_dense_evals_reuse(
                            &mut self.entry_expected,
                            &mut self.entry_expected_scratch,
                            challenge,
                        );
                    },
                );
            },
        );
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

        self.cycle_factors = if let Some(bytecode_ra_indices) = self.bytecode_ra_indices.take() {
            BytecodeReadRafCycleFactors::sparse(
                bytecode_ra_indices,
                &self.chunk_lens,
                &address_point,
            )
        } else {
            BytecodeReadRafCycleFactors::dense(self.dense_cycle_factors(&address_point))
        };
        let stage_coefficients =
            std::array::from_fn::<_, BYTECODE_READ_RAF_STAGE_COUNT, _>(|stage| {
                self.gamma_powers[stage] * bound_stage_values[stage]
            });
        let trace_len = 1usize << self.log_t;
        self.cycle_combined_eq = (0..trace_len)
            .into_par_iter()
            .map(|index| {
                let mut combined = F::zero();
                for (stage, &coefficient) in stage_coefficients.iter().enumerate() {
                    combined += coefficient * self.cycle_eqs[stage][index];
                }
                combined
            })
            .collect();
        let entry_coefficient = self.gamma_powers[7] * bound_entry_expected;
        if let Some(combined) = self.cycle_combined_eq.first_mut() {
            *combined += entry_coefficient;
        }

        self.bound_stage_values = Some(bound_stage_values);
        self.bound_entry_expected = Some(bound_entry_expected);
        self.stage_factors = std::array::from_fn(|_| Vec::new());
        self.stage_factor_scratch = std::array::from_fn(|_| Vec::new());
        self.stage_values = std::array::from_fn(|_| Vec::new());
        self.stage_value_scratch = std::array::from_fn(|_| Vec::new());
        self.entry_trace.clear();
        self.entry_trace_scratch.clear();
        self.entry_expected.clear();
        self.entry_expected_scratch.clear();
        self.cycle_eqs = std::array::from_fn(|_| Vec::new());
        self.cycle_eq_scratch = std::array::from_fn(|_| Vec::new());
        self.cycle_entry_eq.clear();
        self.cycle_entry_eq_scratch.clear();
        self.phase = BytecodeReadRafPhase::Cycle;
    }

    fn dense_cycle_factors(&self, address_point: &[F]) -> Vec<Vec<F>> {
        self.bytecode_ra_chunks
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
            .collect()
    }

    fn bind_cycle(&mut self, challenge: F) {
        self.cycle_factors.bind(challenge);
        bind_dense_evals_reuse(
            &mut self.cycle_combined_eq,
            &mut self.cycle_combined_eq_scratch,
            challenge,
        );
    }

    fn final_relation_eval(&self, relation: Stage6Relation) -> Result<F, Stage6KernelError> {
        if self.bound_stage_values.is_none() {
            return Err(Stage6KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "bytecode read RAF final eval missing bound values",
            });
        }
        if self.bound_entry_expected.is_none() {
            return Err(Stage6KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "bytecode read RAF final eval missing entry value",
            });
        }
        let mut ra_product = F::one();
        for chunk in 0..self.cycle_factors.len() {
            ra_product *=
                self.cycle_factors
                    .factor_eval(chunk)
                    .ok_or(Stage6KernelError::InvalidProof {
                        driver: relation.symbol(),
                        reason: "bytecode read RAF final eval missing RA factor",
                    })?;
        }
        let weighted_value =
            self.cycle_combined_eq
                .first()
                .copied()
                .ok_or(Stage6KernelError::InvalidProof {
                    driver: relation.symbol(),
                    reason: "bytecode read RAF final eval missing combined cycle eq",
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
                let value = self.cycle_factors.factor_eval(factor).ok_or(
                    Stage6KernelError::InvalidProof {
                        driver: relation.symbol(),
                        reason: "bytecode read RAF final eval missing output factor",
                    },
                )?;
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

#[inline]
fn multiply_linear_factor<F: Field>(
    coefficients: &mut [F; 6],
    scratch: &mut [F; 6],
    degree: &mut usize,
    constant: F,
    slope: F,
) {
    for value in scratch.iter_mut().take(*degree + 2) {
        *value = F::zero();
    }
    for index in 0..=*degree {
        scratch[index] += coefficients[index] * constant;
        scratch[index + 1] += coefficients[index] * slope;
    }
    *degree += 1;
    coefficients[..=*degree].copy_from_slice(&scratch[..=*degree]);
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
        if self.degree_bound == 3 {
            let mut evals = self.round_evals_degree3(first_len / 2);
            for eval in &mut evals {
                *eval *= self.active_scale;
            }
            let poly = UnivariatePoly::from_evals_and_hint(previous_claim, &evals);
            check_round_claim(
                &poly,
                previous_claim,
                relation.symbol(),
                "stage6 relation input claim mismatch",
            )?;
            return Ok(poly);
        }

        let mut evals = self.round_evals_generic(first_len / 2);
        for eval in &mut evals {
            *eval *= self.active_scale;
        }
        let poly = UnivariatePoly::from_evals(&evals);
        check_round_claim(
            &poly,
            previous_claim,
            relation.symbol(),
            "stage6 relation input claim mismatch",
        )?;
        Ok(poly)
    }

    fn round_evals_degree3(&self, half_len: usize) -> [F; 3] {
        if half_len >= DENSE_BIND_PAR_THRESHOLD {
            (0..half_len)
                .into_par_iter()
                .fold(
                    || [F::zero(); 3],
                    |mut row_evals, row| {
                        self.accumulate_row_degree3(row, &mut row_evals);
                        row_evals
                    },
                )
                .reduce(
                    || [F::zero(); 3],
                    |left, right| [left[0] + right[0], left[1] + right[1], left[2] + right[2]],
                )
        } else {
            let mut evals = [F::zero(); 3];
            for row in 0..half_len {
                self.accumulate_row_degree3(row, &mut evals);
            }
            evals
        }
    }

    fn accumulate_row_degree3(&self, row: usize, evals: &mut [F; 3]) {
        let eq_low = self.eq[row << 1];
        let eq_high = self.eq[(row << 1) + 1];
        let delta_eq = eq_high - eq_low;
        let eq_at_0 = eq_low;
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
            let ra_at_2 = ra_low + delta_ra.mul_u64(2);
            let ra_at_3 = ra_low + delta_ra.mul_u64(3);
            evals[0] += gamma_power * eq_at_0 * (ra_at_0.square() - ra_at_0);
            evals[1] += gamma_power * eq_at_2 * (ra_at_2.square() - ra_at_2);
            evals[2] += gamma_power * eq_at_3 * (ra_at_3.square() - ra_at_3);
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
                    bind_dense_evals_reuse_serial(chunk, scratch, challenge);
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

struct CoreBooleanityStage6State<'a, F: Field> {
    log_k_chunk: usize,
    num_polys: usize,
    address_round: usize,
    b: GruenSplitEqPolynomial<F>,
    d: GruenSplitEqPolynomial<F>,
    f_table: ExpandingTable<F>,
    eq_r_r: F,
    eq_r_r_inv: F,
    g: Vec<Vec<F>>,
    indices: Cow<'a, [Stage6BooleanityRow]>,
    h: Option<CoreBooleanityHState<'a, F>>,
    gamma_powers: Vec<F>,
    gamma_powers_inv: Vec<F>,
    gamma_powers_square: Vec<F>,
    outputs: Vec<FactorOutput>,
    active_scale: F,
}

impl<'a, F: Field> CoreBooleanityStage6State<'a, F> {
    fn new(
        r_address: &[F],
        r_cycle: &[F],
        indices: &[&[Option<u8>]],
        row_indices: Option<&'a [Stage6BooleanityRow]>,
        gamma: F,
        outputs: Vec<FactorOutput>,
        active_scale: F,
    ) -> Result<Self, Stage6KernelError> {
        let log_k_chunk = r_address.len();
        let chunk_domain = 1usize << log_k_chunk;
        let trace_len = 1usize << r_cycle.len();
        if indices.iter().any(|chunk| chunk.len() != trace_len) {
            return Err(Stage6KernelError::InvalidProof {
                driver: Stage6Relation::Booleanity.symbol(),
                reason: "booleanity index chunks have inconsistent trace lengths",
            });
        }

        let num_polys = indices.len();
        let row_indices = if let Some(rows) = row_indices {
            require_operand_count("stage6.booleanity.row_indices", trace_len, rows.len())?;
            Cow::Borrowed(rows)
        } else {
            Cow::Owned(core_booleanity_row_indices(
                indices,
                trace_len,
                chunk_domain,
            )?)
        };
        let g = core_booleanity_g_from_rows(r_cycle, &row_indices, num_polys, chunk_domain);

        let mut gamma_powers = Vec::with_capacity(num_polys);
        let mut gamma_powers_inv = Vec::with_capacity(num_polys);
        let mut gamma_powers_square = Vec::with_capacity(num_polys);
        let mut gamma_power = F::one();
        let gamma_square = gamma.square();
        let mut gamma_square_power = F::one();
        for _ in 0..num_polys {
            gamma_powers.push(gamma_power);
            gamma_powers_inv.push(gamma_power.inverse().ok_or(
                Stage6KernelError::InvalidProof {
                    driver: Stage6Relation::Booleanity.symbol(),
                    reason: "booleanity gamma power is not invertible",
                },
            )?);
            gamma_powers_square.push(gamma_square_power);
            gamma_power *= gamma;
            gamma_square_power *= gamma_square;
        }

        let mut f_table = ExpandingTable::new(chunk_domain, BindingOrder::LowToHigh);
        f_table.reset(F::one());

        Ok(Self {
            log_k_chunk,
            num_polys,
            address_round: 0,
            b: GruenSplitEqPolynomial::new(r_address, BindingOrder::LowToHigh),
            d: GruenSplitEqPolynomial::new(r_cycle, BindingOrder::LowToHigh),
            f_table,
            eq_r_r: F::zero(),
            eq_r_r_inv: F::zero(),
            g,
            indices: row_indices,
            h: None,
            gamma_powers,
            gamma_powers_inv,
            gamma_powers_square,
            outputs,
            active_scale,
        })
    }

    fn round_poly(
        &self,
        previous_claim: F,
        relation: Stage6Relation,
    ) -> Result<UnivariatePoly<F>, Stage6KernelError> {
        if relation != Stage6Relation::Booleanity {
            return Err(Stage6KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "wrong relation for core booleanity state",
            });
        }
        let mut poly = if self.h.is_none() {
            let _span = trace_stage6_inner_spans()
                .then(|| tracing::info_span!("Stage6::booleanity.address_round").entered());
            self.address_round_poly(previous_claim)
        } else {
            let _span = trace_stage6_inner_spans()
                .then(|| tracing::info_span!("Stage6::booleanity.cycle_round").entered());
            self.cycle_round_poly(previous_claim)?
        };
        if self.active_scale != F::one() {
            poly *= self.active_scale;
        }
        check_round_claim(
            &poly,
            previous_claim,
            relation.symbol(),
            "stage6 booleanity input claim mismatch",
        )?;
        Ok(poly)
    }

    fn address_round_poly(&self, previous_claim: F) -> UnivariatePoly<F> {
        let m = self.address_round + 1;
        let f_values = self.f_table.values();

        let quadratic_accs = self.b.fold_out_in(
            || [F::Accumulator::default(); 2],
            |inner, k_prime, _x_in, e_in| {
                for (g_i, &gamma_square) in self.g.iter().zip(&self.gamma_powers_square) {
                    let mut eval_0 = F::zero();
                    let mut eval_infty = F::zero();
                    let block_start = k_prime << m;
                    for (k, &g_k) in g_i[block_start..block_start + (1 << m)].iter().enumerate() {
                        let k_m = k >> (m - 1);
                        let f_k = f_values[k & ((1 << (m - 1)) - 1)];
                        let g_times_f = g_k * f_k;
                        let eval_inf = g_times_f * f_k;
                        if k_m == 0 {
                            eval_0 += eval_inf - g_times_f;
                        }
                        eval_infty += eval_inf;
                    }
                    let weight = e_in * gamma_square;
                    inner[0].fmadd(weight, eval_0);
                    inner[1].fmadd(weight, eval_infty);
                }
            },
            |_x_out, e_out, inner| {
                let mut outer = [F::Accumulator::default(); 2];
                outer[0].fmadd(e_out, inner[0].reduce());
                outer[1].fmadd(e_out, inner[1].reduce());
                outer
            },
            |mut left, right| {
                left[0].merge(right[0]);
                left[1].merge(right[1]);
                left
            },
        );
        let quadratic_coeffs = [quadratic_accs[0].reduce(), quadratic_accs[1].reduce()];
        self.b
            .gruen_poly_deg_3(quadratic_coeffs[0], quadratic_coeffs[1], previous_claim)
    }

    fn cycle_round_poly(&self, previous_claim: F) -> Result<UnivariatePoly<F>, Stage6KernelError> {
        let h = self.h.as_ref().ok_or(Stage6KernelError::InvalidProof {
            driver: Stage6Relation::Booleanity.symbol(),
            reason: "booleanity cycle state is missing",
        })?;
        if let CoreBooleanityHState::RoundN { h, .. } = h {
            return self.cycle_round_poly_round_n(previous_claim, h);
        }
        let e_out = self.d.e_out_current();
        let e_in = self.d.e_in_current();
        let in_bits = e_in.len().trailing_zeros() as usize;
        let quadratic_accs = (0..e_out.len())
            .into_par_iter()
            .map(|x_out| {
                let mut inner_accs = [F::Accumulator::default(); 2];
                for (x_in, &e_in) in e_in.iter().enumerate() {
                    let j_prime = (x_out << in_bits) | x_in;
                    let mut constant_acc = F::Accumulator::default();
                    let mut quadratic_acc = F::Accumulator::default();
                    for index in 0..h.num_polys() {
                        let (h_0, h_1) = h.get_bound_pair(index, j_prime);
                        let delta = h_1 - h_0;
                        let rho = self.gamma_powers[index];
                        constant_acc.fmadd(h_0, h_0 - rho);
                        quadratic_acc.fmadd(delta, delta);
                    }
                    inner_accs[0].fmadd(e_in, constant_acc.reduce());
                    inner_accs[1].fmadd(e_in, quadratic_acc.reduce());
                }
                let mut outer_accs = [F::Accumulator::default(); 2];
                outer_accs[0].fmadd(e_out[x_out], inner_accs[0].reduce());
                outer_accs[1].fmadd(e_out[x_out], inner_accs[1].reduce());
                outer_accs
            })
            .reduce(
                || [F::Accumulator::default(); 2],
                |mut left, right| {
                    left[0].merge(right[0]);
                    left[1].merge(right[1]);
                    left
                },
            );
        let quadratic_coeffs = [quadratic_accs[0].reduce(), quadratic_accs[1].reduce()];
        let adjusted_claim = previous_claim * self.eq_r_r_inv;
        Ok(self
            .d
            .gruen_poly_deg_3(quadratic_coeffs[0], quadratic_coeffs[1], adjusted_claim)
            * self.eq_r_r)
    }

    fn cycle_round_poly_round_n(
        &self,
        previous_claim: F,
        h: &[Vec<F>],
    ) -> Result<UnivariatePoly<F>, Stage6KernelError> {
        let e_out = self.d.e_out_current();
        let e_in = self.d.e_in_current();
        let in_bits = e_in.len().trailing_zeros() as usize;
        let quadratic_accs = (0..e_out.len())
            .into_par_iter()
            .map(|x_out| {
                let mut inner_accs = [F::Accumulator::default(); 2];
                let base = x_out << in_bits;
                for (x_in, &e_in) in e_in.iter().enumerate() {
                    let j_prime = base | x_in;
                    let mut constant_acc = F::Accumulator::default();
                    let mut quadratic_acc = F::Accumulator::default();
                    for (index, h_poly) in h.iter().enumerate() {
                        let rho = self.gamma_powers[index];
                        let h_0 = h_poly[2 * j_prime];
                        let h_1 = h_poly[2 * j_prime + 1];
                        let delta = h_1 - h_0;
                        constant_acc.fmadd(h_0, h_0 - rho);
                        quadratic_acc.fmadd(delta, delta);
                    }
                    inner_accs[0].fmadd(e_in, constant_acc.reduce());
                    inner_accs[1].fmadd(e_in, quadratic_acc.reduce());
                }
                let mut outer_accs = [F::Accumulator::default(); 2];
                outer_accs[0].fmadd(e_out[x_out], inner_accs[0].reduce());
                outer_accs[1].fmadd(e_out[x_out], inner_accs[1].reduce());
                outer_accs
            })
            .reduce(
                || [F::Accumulator::default(); 2],
                |mut left, right| {
                    left[0].merge(right[0]);
                    left[1].merge(right[1]);
                    left
                },
            );
        let quadratic_coeffs = [quadratic_accs[0].reduce(), quadratic_accs[1].reduce()];
        let adjusted_claim = previous_claim * self.eq_r_r_inv;
        Ok(self
            .d
            .gruen_poly_deg_3(quadratic_coeffs[0], quadratic_coeffs[1], adjusted_claim)
            * self.eq_r_r)
    }

    fn bind(&mut self, challenge: F) {
        if self.h.is_none() {
            self.b.bind(challenge);
            self.f_table.update(challenge);
            self.address_round += 1;
            if self.address_round == self.log_k_chunk {
                self.eq_r_r = self.b.current_scalar();
                self.eq_r_r_inv = self.eq_r_r.inverse().unwrap_or_else(|| {
                    // A non-invertible equality scalar would make the verifier's
                    // phase-2 claim normalization undefined.
                    std::process::abort();
                });
                let base_eq = self.f_table.clone_values();
                let tables = (0..self.num_polys)
                    .into_par_iter()
                    .map(|chunk_index| {
                        let rho = self.gamma_powers[chunk_index];
                        base_eq.iter().map(|value| rho * *value).collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let indices = std::mem::replace(&mut self.indices, Cow::Borrowed(&[]));
                self.h = Some(CoreBooleanityHState::new(tables, indices));
            }
        } else {
            self.d.bind(challenge);
            if let Some(h) = &mut self.h {
                h.bind(challenge);
            }
        }
    }

    fn factor_eval(&self, index: usize, relation: Stage6Relation) -> Result<F, Stage6KernelError> {
        self.h
            .as_ref()
            .map(|h| h.final_sumcheck_claim(index))
            .map(|value| value * self.gamma_powers_inv[index])
            .ok_or(Stage6KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "empty core booleanity factor",
            })
    }

    fn final_relation_eval(&self, relation: Stage6Relation) -> Result<F, Stage6KernelError> {
        if relation != Stage6Relation::Booleanity {
            return Err(Stage6KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "wrong relation for core booleanity state",
            });
        }
        let eq = self.d.current_scalar() * self.eq_r_r;
        let h = self.h.as_ref().ok_or(Stage6KernelError::InvalidProof {
            driver: relation.symbol(),
            reason: "booleanity cycle state is missing",
        })?;
        let mut booleanity = F::zero();
        for index in 0..h.num_polys() {
            let scaled = h.final_sumcheck_claim(index);
            booleanity += scaled * (scaled - self.gamma_powers[index]);
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

fn core_booleanity_g_from_rows<F: Field>(
    r_cycle: &[F],
    row_indices: &[Stage6BooleanityRow],
    num_polys: usize,
    chunk_domain: usize,
) -> Vec<Vec<F>> {
    let log_t = r_cycle.len();
    let lo_bits = log_t / 2;
    let hi_bits = log_t - lo_bits;
    let (r_hi, r_lo) = r_cycle.split_at(hi_bits);
    let (e_hi, e_lo) = rayon::join(
        || EqPolynomial::<F>::evals(r_hi, None),
        || EqPolynomial::<F>::evals(r_lo, None),
    );

    let in_len = e_lo.len();
    let chunk_size = e_hi.len().div_ceil(rayon::current_num_threads()).max(1);

    e_hi.par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_index, e_hi_chunk)| {
            let mut partial = (0..num_polys)
                .map(|_| vec![F::zero(); chunk_domain])
                .collect::<Vec<_>>();
            let mut local = (0..num_polys)
                .map(|_| vec![F::ScalarAccumulator::default(); chunk_domain])
                .collect::<Vec<_>>();
            let mut touched = (0..num_polys)
                .map(|_| Vec::<usize>::with_capacity(chunk_domain))
                .collect::<Vec<_>>();
            let mut touched_flags = (0..num_polys)
                .map(|_| vec![false; chunk_domain])
                .collect::<Vec<_>>();

            let chunk_start = chunk_index * chunk_size;
            for (local_hi, &hi_weight) in e_hi_chunk.iter().enumerate() {
                for poly_index in 0..num_polys {
                    for &index in &touched[poly_index] {
                        local[poly_index][index] = F::ScalarAccumulator::default();
                        touched_flags[poly_index][index] = false;
                    }
                    touched[poly_index].clear();
                }

                let cycle_base = (chunk_start + local_hi) * in_len;
                for (lo_index, &lo_weight) in e_lo.iter().enumerate() {
                    let cycle = cycle_base + lo_index;
                    let Some(row) = row_indices.get(cycle) else {
                        break;
                    };
                    for poly_index in 0..num_polys {
                        if let Some(index) = row.get(poly_index) {
                            let index = usize::from(index);
                            if !touched_flags[poly_index][index] {
                                touched_flags[poly_index][index] = true;
                                touched[poly_index].push(index);
                            }
                            local[poly_index][index].add(lo_weight);
                        }
                    }
                }

                for poly_index in 0..num_polys {
                    for &index in &touched[poly_index] {
                        partial[poly_index][index] += hi_weight * local[poly_index][index].reduce();
                    }
                }
            }
            partial
        })
        .reduce(
            || {
                (0..num_polys)
                    .map(|_| vec![F::zero(); chunk_domain])
                    .collect::<Vec<_>>()
            },
            |mut left, right| {
                for (left_poly, right_poly) in left.iter_mut().zip(right) {
                    for (left_value, right_value) in left_poly.iter_mut().zip(right_poly) {
                        *left_value += right_value;
                    }
                }
                left
            },
        )
}

fn core_booleanity_row_indices(
    chunks: &[&[Option<u8>]],
    trace_len: usize,
    chunk_domain: usize,
) -> Result<Vec<Stage6BooleanityRow>, Stage6KernelError> {
    if chunks.len() > jolt_witness::STAGE6_BOOLEANITY_MAX_POLYS {
        return Err(Stage6KernelError::InvalidProof {
            driver: Stage6Relation::Booleanity.symbol(),
            reason: "booleanity has too many index chunks for compact row-major state",
        });
    }

    let mut rows = vec![Stage6BooleanityRow::empty(); trace_len];
    for (chunk_index, chunk) in chunks.iter().enumerate() {
        for (cycle, index) in chunk.iter().enumerate() {
            let Some(index) = index else {
                continue;
            };
            if usize::from(*index) >= chunk_domain {
                return Err(Stage6KernelError::InvalidProof {
                    driver: Stage6Relation::Booleanity.symbol(),
                    reason: "booleanity index exceeds chunk domain",
                });
            }
            rows[cycle].set(chunk_index, *index);
        }
    }
    Ok(rows)
}

enum CoreBooleanityHState<'a, F: Field> {
    Round1 {
        tables: Vec<Vec<F>>,
        indices: Cow<'a, [Stage6BooleanityRow]>,
    },
    Round2 {
        tables_0: Vec<Vec<F>>,
        tables_1: Vec<Vec<F>>,
        indices: Cow<'a, [Stage6BooleanityRow]>,
    },
    Round3 {
        tables_00: Vec<Vec<F>>,
        tables_01: Vec<Vec<F>>,
        tables_10: Vec<Vec<F>>,
        tables_11: Vec<Vec<F>>,
        indices: Cow<'a, [Stage6BooleanityRow]>,
    },
    RoundN {
        h: Vec<Vec<F>>,
        scratch: Vec<Vec<F>>,
    },
}

impl<'a, F: Field> CoreBooleanityHState<'a, F> {
    fn new(tables: Vec<Vec<F>>, indices: Cow<'a, [Stage6BooleanityRow]>) -> Self {
        Self::Round1 { tables, indices }
    }

    fn num_polys(&self) -> usize {
        match self {
            Self::Round1 { tables, .. } => tables.len(),
            Self::Round2 { tables_0, .. } => tables_0.len(),
            Self::Round3 { tables_00, .. } => tables_00.len(),
            Self::RoundN { h, .. } => h.len(),
        }
    }

    fn get_bound_coeff(&self, poly_idx: usize, j: usize) -> F {
        match self {
            Self::Round1 { tables, indices } => indices[j]
                .get(poly_idx)
                .map_or(F::zero(), |index| tables[poly_idx][usize::from(index)]),
            Self::Round2 {
                tables_0,
                tables_1,
                indices,
            } => {
                let h_0 = indices[2 * j]
                    .get(poly_idx)
                    .map_or(F::zero(), |index| tables_0[poly_idx][usize::from(index)]);
                let h_1 = indices[2 * j + 1]
                    .get(poly_idx)
                    .map_or(F::zero(), |index| tables_1[poly_idx][usize::from(index)]);
                h_0 + h_1
            }
            Self::Round3 {
                tables_00,
                tables_01,
                tables_10,
                tables_11,
                indices,
            } => {
                let h_00 = indices[4 * j]
                    .get(poly_idx)
                    .map_or(F::zero(), |index| tables_00[poly_idx][usize::from(index)]);
                let h_10 = indices[4 * j + 1]
                    .get(poly_idx)
                    .map_or(F::zero(), |index| tables_10[poly_idx][usize::from(index)]);
                let h_01 = indices[4 * j + 2]
                    .get(poly_idx)
                    .map_or(F::zero(), |index| tables_01[poly_idx][usize::from(index)]);
                let h_11 = indices[4 * j + 3]
                    .get(poly_idx)
                    .map_or(F::zero(), |index| tables_11[poly_idx][usize::from(index)]);
                h_00 + h_10 + h_01 + h_11
            }
            Self::RoundN { h, .. } => h[poly_idx][j],
        }
    }

    fn get_bound_pair(&self, poly_idx: usize, row: usize) -> (F, F) {
        match self {
            Self::Round1 { tables, indices } => {
                let source = 2 * row;
                (
                    indices[source]
                        .get(poly_idx)
                        .map_or(F::zero(), |index| tables[poly_idx][usize::from(index)]),
                    indices[source + 1]
                        .get(poly_idx)
                        .map_or(F::zero(), |index| tables[poly_idx][usize::from(index)]),
                )
            }
            Self::Round2 {
                tables_0,
                tables_1,
                indices,
            } => {
                let source = 4 * row;
                let low = indices[source]
                    .get(poly_idx)
                    .map_or(F::zero(), |index| tables_0[poly_idx][usize::from(index)])
                    + indices[source + 1]
                        .get(poly_idx)
                        .map_or(F::zero(), |index| tables_1[poly_idx][usize::from(index)]);
                let high = indices[source + 2]
                    .get(poly_idx)
                    .map_or(F::zero(), |index| tables_0[poly_idx][usize::from(index)])
                    + indices[source + 3]
                        .get(poly_idx)
                        .map_or(F::zero(), |index| tables_1[poly_idx][usize::from(index)]);
                (low, high)
            }
            Self::Round3 {
                tables_00,
                tables_01,
                tables_10,
                tables_11,
                indices,
            } => {
                let source = 8 * row;
                let low = indices[source]
                    .get(poly_idx)
                    .map_or(F::zero(), |index| tables_00[poly_idx][usize::from(index)])
                    + indices[source + 1]
                        .get(poly_idx)
                        .map_or(F::zero(), |index| tables_10[poly_idx][usize::from(index)])
                    + indices[source + 2]
                        .get(poly_idx)
                        .map_or(F::zero(), |index| tables_01[poly_idx][usize::from(index)])
                    + indices[source + 3]
                        .get(poly_idx)
                        .map_or(F::zero(), |index| tables_11[poly_idx][usize::from(index)]);
                let high = indices[source + 4]
                    .get(poly_idx)
                    .map_or(F::zero(), |index| tables_00[poly_idx][usize::from(index)])
                    + indices[source + 5]
                        .get(poly_idx)
                        .map_or(F::zero(), |index| tables_10[poly_idx][usize::from(index)])
                    + indices[source + 6]
                        .get(poly_idx)
                        .map_or(F::zero(), |index| tables_01[poly_idx][usize::from(index)])
                    + indices[source + 7]
                        .get(poly_idx)
                        .map_or(F::zero(), |index| tables_11[poly_idx][usize::from(index)]);
                (low, high)
            }
            Self::RoundN { h, .. } => (h[poly_idx][2 * row], h[poly_idx][2 * row + 1]),
        }
    }

    fn final_sumcheck_claim(&self, poly_idx: usize) -> F {
        self.get_bound_coeff(poly_idx, 0)
    }

    fn bind(&mut self, challenge: F) {
        match std::mem::replace(
            self,
            Self::RoundN {
                h: Vec::new(),
                scratch: Vec::new(),
            },
        ) {
            Self::Round1 { tables, indices } => {
                let one_minus = F::one() - challenge;
                let (tables_0, tables_1) = rayon::join(
                    || scale_booleanity_tables(&tables, one_minus),
                    || scale_booleanity_tables(&tables, challenge),
                );
                *self = Self::Round2 {
                    tables_0,
                    tables_1,
                    indices,
                };
            }
            Self::Round2 {
                tables_0,
                tables_1,
                indices,
            } => {
                let one_minus = F::one() - challenge;
                let (tables_00, tables_01) = rayon::join(
                    || scale_booleanity_tables(&tables_0, one_minus),
                    || scale_booleanity_tables(&tables_0, challenge),
                );
                let (tables_10, tables_11) = rayon::join(
                    || scale_booleanity_tables(&tables_1, one_minus),
                    || scale_booleanity_tables(&tables_1, challenge),
                );
                *self = Self::Round3 {
                    tables_00,
                    tables_01,
                    tables_10,
                    tables_11,
                    indices,
                };
            }
            Self::Round3 {
                tables_00,
                tables_01,
                tables_10,
                tables_11,
                indices,
            } => {
                let one_minus = F::one() - challenge;
                let (tables_000, tables_001) = rayon::join(
                    || scale_booleanity_tables(&tables_00, one_minus),
                    || scale_booleanity_tables(&tables_00, challenge),
                );
                let (tables_010, tables_011) = rayon::join(
                    || scale_booleanity_tables(&tables_01, one_minus),
                    || scale_booleanity_tables(&tables_01, challenge),
                );
                let (tables_100, tables_101) = rayon::join(
                    || scale_booleanity_tables(&tables_10, one_minus),
                    || scale_booleanity_tables(&tables_10, challenge),
                );
                let (tables_110, tables_111) = rayon::join(
                    || scale_booleanity_tables(&tables_11, one_minus),
                    || scale_booleanity_tables(&tables_11, challenge),
                );
                let table_groups = [
                    &tables_000,
                    &tables_100,
                    &tables_010,
                    &tables_110,
                    &tables_001,
                    &tables_101,
                    &tables_011,
                    &tables_111,
                ];
                let new_len = indices.len() / 8;
                let num_polys = tables_000.len();
                let h = (0..num_polys)
                    .into_par_iter()
                    .map(|poly_idx| {
                        (0..new_len)
                            .into_par_iter()
                            .map(|j| {
                                (0..8)
                                    .map(|offset| {
                                        indices[8 * j + offset].get(poly_idx).map_or(
                                            F::zero(),
                                            |index| {
                                                table_groups[offset][poly_idx][usize::from(index)]
                                            },
                                        )
                                    })
                                    .sum()
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let scratch = (0..h.len()).map(|_| Vec::new()).collect();
                *self = Self::RoundN { h, scratch };
            }
            Self::RoundN { mut h, mut scratch } => {
                if h.first().map_or(0, Vec::len) / 2 >= DENSE_BIND_PAR_THRESHOLD {
                    h.par_iter_mut()
                        .zip(scratch.par_iter_mut())
                        .for_each(|(chunk, scratch)| {
                            bind_dense_evals_reuse_serial(chunk, scratch, challenge);
                        });
                } else {
                    for (chunk, scratch) in h.iter_mut().zip(&mut scratch) {
                        bind_dense_evals_reuse(chunk, scratch, challenge);
                    }
                }
                *self = Self::RoundN { h, scratch };
            }
        }
    }
}

fn scale_booleanity_tables<F: Field>(tables: &[Vec<F>], scalar: F) -> Vec<Vec<F>> {
    tables
        .par_iter()
        .map(|table| {
            table
                .iter()
                .map(|value| *value * scalar)
                .collect::<Vec<_>>()
        })
        .collect()
}

pub(crate) struct HammingBooleanityStage6State<F: Field> {
    pub(crate) eq: GruenSplitEqPolynomial<F>,
    pub(crate) hamming_weight: Vec<F>,
    hamming_weight_scratch: Vec<F>,
    output: FactorOutput,
    pub(crate) active_scale: F,
    #[cfg(feature = "cuda")]
    cuda: Option<cuda::CudaHammingBooleanityState>,
}

impl<F: Field> HammingBooleanityStage6State<F> {
    pub(crate) fn new_with_backend(
        point: &[F],
        hamming_weight: Vec<F>,
        output: FactorOutput,
        active_scale: F,
        degree_bound: usize,
        backend: &'static str,
    ) -> Result<Self, Stage6KernelError> {
        if degree_bound < 3 {
            return Err(Stage6KernelError::InvalidProof {
                driver: Stage6Relation::HammingBooleanity.symbol(),
                reason: "hamming booleanity degree bound is too small",
            });
        }
        require_operand_count(
            "stage6.hamming_booleanity.eq",
            hamming_weight.len(),
            1usize << point.len(),
        )?;
        #[cfg(feature = "cuda")]
        let cuda = if backend == "cuda" {
            cuda::CudaHammingBooleanityState::new(&hamming_weight)
        } else {
            None
        };
        #[cfg(not(feature = "cuda"))]
        let _ = backend;
        Ok(Self {
            eq: GruenSplitEqPolynomial::new(point, BindingOrder::LowToHigh),
            hamming_weight,
            hamming_weight_scratch: Vec::new(),
            output,
            active_scale,
            #[cfg(feature = "cuda")]
            cuda,
        })
    }

    pub(crate) fn round_poly(
        &self,
        previous_claim: F,
        relation: Stage6Relation,
    ) -> Result<UnivariatePoly<F>, Stage6KernelError> {
        if relation != Stage6Relation::HammingBooleanity {
            return Err(Stage6KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "wrong relation for hamming booleanity state",
            });
        }
        #[cfg(feature = "cuda")]
        if let Some(cuda) = &self.cuda {
            if let Some(q) =
                cuda.round_poly_q(self.eq.e_in_current(), self.eq.e_out_current())
            {
                if let (Some(q_constant), Some(q_top)) =
                    (crate::cuda::fr_into::<F>(q[0]), crate::cuda::fr_into::<F>(q[1]))
                {
                    let mut poly = self.eq.gruen_poly_deg_3(q_constant, q_top, previous_claim);
                    if self.active_scale != F::one() {
                        poly *= self.active_scale;
                    }
                    check_round_claim(
                        &poly,
                        previous_claim,
                        relation.symbol(),
                        "stage6 relation input claim mismatch",
                    )?;
                    return Ok(poly);
                }
            }
        }
        let e_out = self.eq.e_out_current();
        let e_in = self.eq.e_in_current();
        let in_bits = e_in.len().trailing_zeros() as usize;
        let quadratic_accs = (0..e_out.len())
            .into_par_iter()
            .map(|x_out| {
                let mut accs = [F::Accumulator::default(); 2];
                let base = x_out << in_bits;
                for (x_in, &e_in) in e_in.iter().enumerate() {
                    let group = base | x_in;
                    let h0 = self.hamming_weight[2 * group];
                    let h1 = self.hamming_weight[2 * group + 1];
                    let delta = h1 - h0;
                    let weight = e_out[x_out] * e_in;
                    accs[0].fmadd(weight, h0.square() - h0);
                    accs[1].fmadd(weight, delta.square());
                }
                accs
            })
            .reduce(
                || [F::Accumulator::default(); 2],
                |mut left, right| {
                    left[0].merge(right[0]);
                    left[1].merge(right[1]);
                    left
                },
            );
        let mut poly = self.eq.gruen_poly_deg_3(
            quadratic_accs[0].reduce(),
            quadratic_accs[1].reduce(),
            previous_claim,
        );
        if self.active_scale != F::one() {
            poly *= self.active_scale;
        }
        check_round_claim(
            &poly,
            previous_claim,
            relation.symbol(),
            "stage6 relation input claim mismatch",
        )?;
        Ok(poly)
    }

    fn bind(&mut self, challenge: F) {
        #[cfg(feature = "cuda")]
        if let Some(cuda) = &mut self.cuda {
            if let Some(challenge_fr) = crate::cuda::into_fr(challenge) {
                if cuda.bind(challenge_fr).is_ok() {
                    self.eq.bind(challenge);
                    return;
                }
            }
        }
        self.eq.bind(challenge);
        bind_dense_evals_reuse(
            &mut self.hamming_weight,
            &mut self.hamming_weight_scratch,
            challenge,
        );
    }

    fn final_relation_eval(&self, relation: Stage6Relation) -> Result<F, Stage6KernelError> {
        #[cfg(feature = "cuda")]
        if let Some(cuda) = &self.cuda {
            if let Ok(hw) = cuda.hamming_weight_first() {
                if let Some(hamming_weight) = crate::cuda::fr_into::<F>(hw) {
                    return Ok(self.eq.current_scalar()
                        * (hamming_weight.square() - hamming_weight));
                }
            }
        }
        let hamming_weight =
            self.hamming_weight
                .first()
                .copied()
                .ok_or(Stage6KernelError::InvalidProof {
                    driver: relation.symbol(),
                    reason: "empty hamming booleanity factor",
                })?;
        Ok(self.eq.current_scalar() * (hamming_weight.square() - hamming_weight))
    }

    fn final_evals(
        &self,
        relation: Stage6Relation,
    ) -> Result<Vec<Stage6NamedEval<F>>, Stage6KernelError> {
        #[cfg(feature = "cuda")]
        if let Some(cuda) = &self.cuda {
            if let Ok(hw) = cuda.hamming_weight_first() {
                if let Some(value) = crate::cuda::fr_into::<F>(hw) {
                    return Ok(vec![named_eval(self.output.name, self.output.oracle, value)]);
                }
            }
        }
        Ok(vec![named_eval(
            self.output.name,
            self.output.oracle,
            self.hamming_weight
                .first()
                .copied()
                .ok_or(Stage6KernelError::InvalidProof {
                    driver: relation.symbol(),
                    reason: "empty hamming booleanity factor",
                })?,
        )])
    }
}

struct IncClaimReductionStage6State<F: Field> {
    eq_ram: Vec<F>,
    eq_ram_scratch: Vec<F>,
    ram_inc: Vec<F>,
    ram_inc_scratch: Vec<F>,
    eq_rd: Vec<F>,
    eq_rd_scratch: Vec<F>,
    rd_inc: Vec<F>,
    rd_inc_scratch: Vec<F>,
    gamma2: F,
    outputs: [FactorOutput; 2],
    active_scale: F,
    #[cfg(feature = "cuda")]
    cuda: Option<cuda::CudaIncState>,
}

impl<F: Field> IncClaimReductionStage6State<F> {
    fn round_poly(
        &self,
        previous_claim: F,
        relation: Stage6Relation,
    ) -> Result<UnivariatePoly<F>, Stage6KernelError> {
        if relation != Stage6Relation::IncClaimReduction {
            return Err(Stage6KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "wrong relation for increment claim-reduction state",
            });
        }
        let len = self.eq_ram.len();
        if len == 0 || !len.is_power_of_two() || self.eq_rd.len() != len {
            return Err(Stage6KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "increment claim-reduction factors have invalid length",
            });
        }
        if self.ram_inc.len() != len || self.rd_inc.len() != len {
            return Err(Stage6KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "increment claim-reduction factors have inconsistent lengths",
            });
        }

        #[cfg(feature = "cuda")]
        if let Some(cuda) = &self.cuda {
            if let Ok(fr_evals) = cuda.round_poly_evals() {
                if let (Some(e0), Some(e1)) = (
                    crate::cuda::fr_into::<F>(fr_evals[0]),
                    crate::cuda::fr_into::<F>(fr_evals[1]),
                ) {
                    let mut evals = [e0, e1];
                    if self.active_scale != F::one() {
                        evals[0] *= self.active_scale;
                        evals[1] *= self.active_scale;
                    }
                    let poly = UnivariatePoly::from_evals_and_hint(previous_claim, &evals);
                    check_round_claim(
                        &poly,
                        previous_claim,
                        relation.symbol(),
                        "stage6 increment claim-reduction input claim mismatch",
                    )?;
                    return Ok(poly);
                }
            }
        }

        let half = len / 2;
        let eval_accs = if half >= DENSE_BIND_PAR_THRESHOLD {
            (0..half)
                .into_par_iter()
                .fold(
                    || [F::Accumulator::default(); 2],
                    |mut accs, row| {
                        self.accumulate_row(row, &mut accs);
                        accs
                    },
                )
                .reduce(
                    || [F::Accumulator::default(); 2],
                    |mut left, right| {
                        left[0].merge(right[0]);
                        left[1].merge(right[1]);
                        left
                    },
                )
        } else {
            let mut accs = [F::Accumulator::default(); 2];
            for row in 0..half {
                self.accumulate_row(row, &mut accs);
            }
            accs
        };
        let mut evals = eval_accs.map(FieldAccumulator::reduce);
        if self.active_scale != F::one() {
            evals[0] *= self.active_scale;
            evals[1] *= self.active_scale;
        }
        let poly = UnivariatePoly::from_evals_and_hint(previous_claim, &evals);
        check_round_claim(
            &poly,
            previous_claim,
            relation.symbol(),
            "stage6 increment claim-reduction input claim mismatch",
        )?;
        Ok(poly)
    }

    fn accumulate_row(&self, row: usize, evals: &mut [F::Accumulator; 2]) {
        let ram_eq_0 = self.eq_ram[2 * row];
        let ram_eq_1 = self.eq_ram[2 * row + 1];
        let ram_inc_0 = self.ram_inc[2 * row];
        let ram_inc_1 = self.ram_inc[2 * row + 1];
        let rd_eq_0 = self.eq_rd[2 * row];
        let rd_eq_1 = self.eq_rd[2 * row + 1];
        let rd_inc_0 = self.rd_inc[2 * row];
        let rd_inc_1 = self.rd_inc[2 * row + 1];

        evals[0].fmadd(ram_eq_0, ram_inc_0);
        evals[0].fmadd(self.gamma2 * rd_eq_0, rd_inc_0);

        let ram_eq_2 = ram_eq_1 + ram_eq_1 - ram_eq_0;
        let ram_inc_2 = ram_inc_1 + ram_inc_1 - ram_inc_0;
        let rd_eq_2 = rd_eq_1 + rd_eq_1 - rd_eq_0;
        let rd_inc_2 = rd_inc_1 + rd_inc_1 - rd_inc_0;
        evals[1].fmadd(ram_eq_2, ram_inc_2);
        evals[1].fmadd(self.gamma2 * rd_eq_2, rd_inc_2);
    }

    fn bind(&mut self, challenge: F) {
        #[cfg(feature = "cuda")]
        if let Some(cuda) = &mut self.cuda {
            if let Some(challenge_fr) = crate::cuda::into_fr(challenge) {
                if cuda.bind(challenge_fr).is_ok() {
                    return;
                }
            }
        }
        if self.eq_ram.len() / 2 >= DENSE_BIND_PAR_THRESHOLD {
            rayon::join(
                || {
                    rayon::join(
                        || {
                            bind_dense_evals_reuse_serial(
                                &mut self.eq_ram,
                                &mut self.eq_ram_scratch,
                                challenge,
                            );
                        },
                        || {
                            bind_dense_evals_reuse_serial(
                                &mut self.ram_inc,
                                &mut self.ram_inc_scratch,
                                challenge,
                            );
                        },
                    );
                },
                || {
                    rayon::join(
                        || {
                            bind_dense_evals_reuse_serial(
                                &mut self.eq_rd,
                                &mut self.eq_rd_scratch,
                                challenge,
                            );
                        },
                        || {
                            bind_dense_evals_reuse_serial(
                                &mut self.rd_inc,
                                &mut self.rd_inc_scratch,
                                challenge,
                            );
                        },
                    );
                },
            );
        } else {
            bind_dense_evals_reuse(&mut self.eq_ram, &mut self.eq_ram_scratch, challenge);
            bind_dense_evals_reuse(&mut self.ram_inc, &mut self.ram_inc_scratch, challenge);
            bind_dense_evals_reuse(&mut self.eq_rd, &mut self.eq_rd_scratch, challenge);
            bind_dense_evals_reuse(&mut self.rd_inc, &mut self.rd_inc_scratch, challenge);
        }
    }

    fn factor_eval(&self, factor: usize, relation: Stage6Relation) -> Result<F, Stage6KernelError> {
        #[cfg(feature = "cuda")]
        if let Some(cuda) = &self.cuda {
            if matches!(factor, 1 | 3) {
                if let Ok(value) = cuda.factor_first(factor) {
                    if let Some(value) = crate::cuda::fr_into::<F>(value) {
                        return Ok(value);
                    }
                }
            }
        }
        let values = match factor {
            1 => &self.ram_inc,
            3 => &self.rd_inc,
            _ => {
                return Err(Stage6KernelError::InvalidProof {
                    driver: relation.symbol(),
                    reason: "unsupported increment claim-reduction output factor",
                });
            }
        };
        values
            .first()
            .copied()
            .ok_or(Stage6KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "empty increment claim-reduction factor",
            })
    }

    fn final_relation_eval(&self, relation: Stage6Relation) -> Result<F, Stage6KernelError> {
        #[cfg(feature = "cuda")]
        if let Some(value) = cuda::inc_final_relation_eval(self) {
            return Ok(value);
        }
        let eq_ram = self
            .eq_ram
            .first()
            .copied()
            .ok_or(Stage6KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "empty increment claim-reduction eq factor",
            })?;
        let ram_inc = self.factor_eval(1, relation)?;
        let eq_rd = self
            .eq_rd
            .first()
            .copied()
            .ok_or(Stage6KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "empty increment claim-reduction eq factor",
            })?;
        let rd_inc = self.factor_eval(3, relation)?;
        Ok(eq_ram * ram_inc + self.gamma2 * eq_rd * rd_inc)
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

struct DenseStage6State<F: Field> {
    factors: Vec<Vec<F>>,
    factor_scratch: Vec<Vec<F>>,
    terms: Vec<DenseTerm<F>>,
    outputs: Vec<FactorOutput>,
    active_scale: F,
    degree_bound: usize,
}

#[derive(Clone)]
pub(crate) struct DenseTerm<F: Field> {
    pub(crate) coefficient: F,
    pub(crate) factors: Vec<usize>,
}

#[derive(Clone, Copy)]
pub(crate) struct FactorOutput {
    pub(crate) name: &'static str,
    pub(crate) oracle: &'static str,
    pub(crate) factor: usize,
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
            previous_claim,
            relation,
        )?;
        check_round_claim(
            &poly,
            previous_claim,
            relation.symbol(),
            "stage6 relation input claim mismatch",
        )?;
        Ok(poly)
    }

    fn bind(&mut self, challenge: F) {
        if self.factors.first().map_or(0, Vec::len) / 2 >= DENSE_BIND_PAR_THRESHOLD {
            self.factors
                .par_iter_mut()
                .zip(self.factor_scratch.par_iter_mut())
                .for_each(|(factor, scratch)| {
                    bind_dense_evals_reuse_serial(factor, scratch, challenge);
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

enum InstructionRaVirtualDenseChunks<'a, F: Field> {
    Borrowed(&'a [&'a [F]]),
    Bound {
        chunks: Vec<Vec<F>>,
        scratch: Vec<Vec<F>>,
    },
}

impl<'a, F: Field> InstructionRaVirtualDenseChunks<'a, F> {
    fn borrowed(chunks: &'a [&'a [F]]) -> Self {
        Self::Borrowed(chunks)
    }

    fn len(&self) -> usize {
        match self {
            Self::Borrowed(chunks) => chunks.len(),
            Self::Bound { chunks, .. } => chunks.len(),
        }
    }

    fn current_len(&self) -> usize {
        match self {
            Self::Borrowed(chunks) => chunks.first().map_or(0, |chunk| chunk.len()),
            Self::Bound { chunks, .. } => chunks.first().map_or(0, Vec::len),
        }
    }

    fn get(&self, chunk: usize, index: usize) -> F {
        match self {
            Self::Borrowed(chunks) => chunks[chunk][index],
            Self::Bound { chunks, .. } => chunks[chunk][index],
        }
    }

    fn get_pair(&self, chunk: usize, row: usize) -> (F, F) {
        match self {
            Self::Borrowed(chunks) => (chunks[chunk][2 * row], chunks[chunk][2 * row + 1]),
            Self::Bound { chunks, .. } => (chunks[chunk][2 * row], chunks[chunk][2 * row + 1]),
        }
    }

    fn final_sumcheck_claim(&self, chunk: usize) -> F {
        self.get(chunk, 0)
    }

    fn bind(&mut self, challenge: F) {
        match self {
            Self::Borrowed(chunks) => {
                let bound = if chunks.first().map_or(0, |chunk| chunk.len() / 2)
                    >= DENSE_BIND_PAR_THRESHOLD
                {
                    chunks
                        .par_iter()
                        .map(|chunk| bind_dense_evals_to_vec(chunk, challenge))
                        .collect::<Vec<_>>()
                } else {
                    chunks
                        .iter()
                        .map(|chunk| bind_dense_evals_to_vec(chunk, challenge))
                        .collect::<Vec<_>>()
                };
                let scratch = (0..bound.len()).map(|_| Vec::new()).collect();
                *self = Self::Bound {
                    chunks: bound,
                    scratch,
                };
            }
            Self::Bound { chunks, scratch } => {
                if chunks.first().map_or(0, Vec::len) / 2 >= DENSE_BIND_PAR_THRESHOLD {
                    chunks.par_iter_mut().zip(scratch.par_iter_mut()).for_each(
                        |(chunk, scratch)| {
                            bind_dense_evals_reuse_serial(chunk, scratch, challenge);
                        },
                    );
                } else {
                    for (chunk, scratch) in chunks.iter_mut().zip(scratch) {
                        bind_dense_evals_reuse(chunk, scratch, challenge);
                    }
                }
            }
        }
    }
}

enum InstructionRaVirtualSparseChunks<'a, F: Field> {
    Round1 {
        tables: Vec<Vec<F>>,
        indices: &'a [&'a [Option<u8>]],
    },
    Round2 {
        tables_0: Vec<Vec<F>>,
        tables_1: Vec<Vec<F>>,
        indices: &'a [&'a [Option<u8>]],
    },
    Round3 {
        tables_00: Vec<Vec<F>>,
        tables_01: Vec<Vec<F>>,
        tables_10: Vec<Vec<F>>,
        tables_11: Vec<Vec<F>>,
        indices: &'a [&'a [Option<u8>]],
    },
    Bound {
        chunks: Vec<Vec<F>>,
        scratch: Vec<Vec<F>>,
    },
}

impl<'a, F: Field> InstructionRaVirtualSparseChunks<'a, F> {
    fn new(tables: Vec<Vec<F>>, indices: &'a [&'a [Option<u8>]]) -> Self {
        Self::Round1 { tables, indices }
    }

    fn len(&self) -> usize {
        match self {
            Self::Round1 { tables, .. } => tables.len(),
            Self::Round2 { tables_0, .. } => tables_0.len(),
            Self::Round3 { tables_00, .. } => tables_00.len(),
            Self::Bound { chunks, .. } => chunks.len(),
        }
    }

    fn current_len(&self) -> usize {
        match self {
            Self::Round1 { indices, .. } => indices.first().map_or(0, |chunk| chunk.len()),
            Self::Round2 { indices, .. } => indices.first().map_or(0, |chunk| chunk.len() / 2),
            Self::Round3 { indices, .. } => indices.first().map_or(0, |chunk| chunk.len() / 4),
            Self::Bound { chunks, .. } => chunks.first().map_or(0, Vec::len),
        }
    }

    fn get(&self, chunk: usize, index: usize) -> F {
        match self {
            Self::Round1 { tables, indices } => {
                indices[chunk][index].map_or(F::zero(), |value| tables[chunk][usize::from(value)])
            }
            Self::Round2 {
                tables_0,
                tables_1,
                indices,
            } => {
                let low = indices[chunk][2 * index]
                    .map_or(F::zero(), |value| tables_0[chunk][usize::from(value)]);
                let high = indices[chunk][2 * index + 1]
                    .map_or(F::zero(), |value| tables_1[chunk][usize::from(value)]);
                low + high
            }
            Self::Round3 {
                tables_00,
                tables_01,
                tables_10,
                tables_11,
                indices,
            } => {
                let h_00 = indices[chunk][4 * index]
                    .map_or(F::zero(), |value| tables_00[chunk][usize::from(value)]);
                let h_10 = indices[chunk][4 * index + 1]
                    .map_or(F::zero(), |value| tables_10[chunk][usize::from(value)]);
                let h_01 = indices[chunk][4 * index + 2]
                    .map_or(F::zero(), |value| tables_01[chunk][usize::from(value)]);
                let h_11 = indices[chunk][4 * index + 3]
                    .map_or(F::zero(), |value| tables_11[chunk][usize::from(value)]);
                h_00 + h_10 + h_01 + h_11
            }
            Self::Bound { chunks, .. } => chunks[chunk][index],
        }
    }

    fn get_pair(&self, chunk: usize, row: usize) -> (F, F) {
        match self {
            Self::Round1 { tables, indices } => {
                let source = 2 * row;
                (
                    indices[chunk][source]
                        .map_or(F::zero(), |value| tables[chunk][usize::from(value)]),
                    indices[chunk][source + 1]
                        .map_or(F::zero(), |value| tables[chunk][usize::from(value)]),
                )
            }
            Self::Round2 {
                tables_0,
                tables_1,
                indices,
            } => {
                let source = 4 * row;
                let low = indices[chunk][source]
                    .map_or(F::zero(), |value| tables_0[chunk][usize::from(value)])
                    + indices[chunk][source + 1]
                        .map_or(F::zero(), |value| tables_1[chunk][usize::from(value)]);
                let high = indices[chunk][source + 2]
                    .map_or(F::zero(), |value| tables_0[chunk][usize::from(value)])
                    + indices[chunk][source + 3]
                        .map_or(F::zero(), |value| tables_1[chunk][usize::from(value)]);
                (low, high)
            }
            Self::Round3 {
                tables_00,
                tables_01,
                tables_10,
                tables_11,
                indices,
            } => {
                let source = 8 * row;
                let low = indices[chunk][source]
                    .map_or(F::zero(), |value| tables_00[chunk][usize::from(value)])
                    + indices[chunk][source + 1]
                        .map_or(F::zero(), |value| tables_10[chunk][usize::from(value)])
                    + indices[chunk][source + 2]
                        .map_or(F::zero(), |value| tables_01[chunk][usize::from(value)])
                    + indices[chunk][source + 3]
                        .map_or(F::zero(), |value| tables_11[chunk][usize::from(value)]);
                let high = indices[chunk][source + 4]
                    .map_or(F::zero(), |value| tables_00[chunk][usize::from(value)])
                    + indices[chunk][source + 5]
                        .map_or(F::zero(), |value| tables_10[chunk][usize::from(value)])
                    + indices[chunk][source + 6]
                        .map_or(F::zero(), |value| tables_01[chunk][usize::from(value)])
                    + indices[chunk][source + 7]
                        .map_or(F::zero(), |value| tables_11[chunk][usize::from(value)]);
                (low, high)
            }
            Self::Bound { chunks, .. } => (chunks[chunk][2 * row], chunks[chunk][2 * row + 1]),
        }
    }

    fn final_sumcheck_claim(&self, chunk: usize) -> F {
        self.get(chunk, 0)
    }

    fn accumulate_d4_product_terms(
        &self,
        row: usize,
        virtual_count: usize,
        evals: &mut [F::Accumulator; 4],
    ) {
        match self {
            Self::Round1 { tables, indices } => {
                let source = 2 * row;
                for virtual_index in 0..virtual_count {
                    let base = virtual_index * 4;
                    let a0 = sparse_round1_pair(tables, indices, base, source);
                    let a1 = sparse_round1_pair(tables, indices, base + 1, source);
                    let a2 = sparse_round1_pair(tables, indices, base + 2, source);
                    let a3 = sparse_round1_pair(tables, indices, base + 3, source);
                    accumulate_instruction_ra_d4_product_terms(evals, a0, a1, a2, a3);
                }
            }
            Self::Round2 {
                tables_0,
                tables_1,
                indices,
            } => {
                let source = 4 * row;
                for virtual_index in 0..virtual_count {
                    let base = virtual_index * 4;
                    let a0 = sparse_round2_pair(tables_0, tables_1, indices, base, source);
                    let a1 = sparse_round2_pair(tables_0, tables_1, indices, base + 1, source);
                    let a2 = sparse_round2_pair(tables_0, tables_1, indices, base + 2, source);
                    let a3 = sparse_round2_pair(tables_0, tables_1, indices, base + 3, source);
                    accumulate_instruction_ra_d4_product_terms(evals, a0, a1, a2, a3);
                }
            }
            Self::Round3 {
                tables_00,
                tables_01,
                tables_10,
                tables_11,
                indices,
            } => {
                let source = 8 * row;
                for virtual_index in 0..virtual_count {
                    let base = virtual_index * 4;
                    let a0 = sparse_round3_pair(
                        tables_00, tables_01, tables_10, tables_11, indices, base, source,
                    );
                    let a1 = sparse_round3_pair(
                        tables_00,
                        tables_01,
                        tables_10,
                        tables_11,
                        indices,
                        base + 1,
                        source,
                    );
                    let a2 = sparse_round3_pair(
                        tables_00,
                        tables_01,
                        tables_10,
                        tables_11,
                        indices,
                        base + 2,
                        source,
                    );
                    let a3 = sparse_round3_pair(
                        tables_00,
                        tables_01,
                        tables_10,
                        tables_11,
                        indices,
                        base + 3,
                        source,
                    );
                    accumulate_instruction_ra_d4_product_terms(evals, a0, a1, a2, a3);
                }
            }
            Self::Bound { chunks, .. } => {
                for virtual_index in 0..virtual_count {
                    let base = virtual_index * 4;
                    let a0 = (chunks[base][2 * row], chunks[base][2 * row + 1]);
                    let a1 = (chunks[base + 1][2 * row], chunks[base + 1][2 * row + 1]);
                    let a2 = (chunks[base + 2][2 * row], chunks[base + 2][2 * row + 1]);
                    let a3 = (chunks[base + 3][2 * row], chunks[base + 3][2 * row + 1]);
                    accumulate_instruction_ra_d4_product_terms(evals, a0, a1, a2, a3);
                }
            }
        }
    }

    fn bind(&mut self, challenge: F, backend: &'static str) {
        let one_minus = F::one() - challenge;
        match std::mem::replace(
            self,
            Self::Bound {
                chunks: Vec::new(),
                scratch: Vec::new(),
            },
        ) {
            Self::Round1 { tables, indices } => {
                let (tables_0, tables_1) = rayon::join(
                    || scale_instruction_ra_tables(&tables, one_minus),
                    || scale_instruction_ra_tables(&tables, challenge),
                );
                *self = Self::Round2 {
                    tables_0,
                    tables_1,
                    indices,
                };
            }
            Self::Round2 {
                tables_0,
                tables_1,
                indices,
            } => {
                let (tables_00, tables_01) = rayon::join(
                    || scale_instruction_ra_tables(&tables_0, one_minus),
                    || scale_instruction_ra_tables(&tables_0, challenge),
                );
                let (tables_10, tables_11) = rayon::join(
                    || scale_instruction_ra_tables(&tables_1, one_minus),
                    || scale_instruction_ra_tables(&tables_1, challenge),
                );
                *self = Self::Round3 {
                    tables_00,
                    tables_01,
                    tables_10,
                    tables_11,
                    indices,
                };
            }
            Self::Round3 {
                tables_00,
                tables_01,
                tables_10,
                tables_11,
                indices,
            } => {
                let (tables_000, tables_001) = rayon::join(
                    || scale_instruction_ra_tables(&tables_00, one_minus),
                    || scale_instruction_ra_tables(&tables_00, challenge),
                );
                let (tables_010, tables_011) = rayon::join(
                    || scale_instruction_ra_tables(&tables_01, one_minus),
                    || scale_instruction_ra_tables(&tables_01, challenge),
                );
                let (tables_100, tables_101) = rayon::join(
                    || scale_instruction_ra_tables(&tables_10, one_minus),
                    || scale_instruction_ra_tables(&tables_10, challenge),
                );
                let (tables_110, tables_111) = rayon::join(
                    || scale_instruction_ra_tables(&tables_11, one_minus),
                    || scale_instruction_ra_tables(&tables_11, challenge),
                );
                let table_groups = [
                    &tables_000,
                    &tables_100,
                    &tables_010,
                    &tables_110,
                    &tables_001,
                    &tables_101,
                    &tables_011,
                    &tables_111,
                ];
                let chunks = 'chunks: {
                    #[cfg(feature = "cuda")]
                    if backend == "cuda" {
                        if let Some(chunks) = cuda::materialize_gather8(&table_groups, indices) {
                            break 'chunks chunks;
                        }
                    }
                    let _ = backend;
                    materialize_gather8(&table_groups, indices)
                };
                let scratch = (0..chunks.len()).map(|_| Vec::new()).collect();
                *self = Self::Bound { chunks, scratch };
            }
            Self::Bound {
                mut chunks,
                mut scratch,
            } => {
                if chunks.first().map_or(0, Vec::len) / 2 >= DENSE_BIND_PAR_THRESHOLD {
                    chunks.par_iter_mut().zip(scratch.par_iter_mut()).for_each(
                        |(chunk, scratch)| {
                            bind_dense_evals_reuse_serial(chunk, scratch, challenge);
                        },
                    );
                } else {
                    for (chunk, scratch) in chunks.iter_mut().zip(&mut scratch) {
                        bind_dense_evals_reuse(chunk, scratch, challenge);
                    }
                }
                *self = Self::Bound { chunks, scratch };
            }
        }
    }
}

/// Gather round-3 tables for use in instruction RA bind.
// NOTE: This is factored out of the instruction RA bind function so it can be used in CUDA
// equivalence tests.
#[inline(always)]
pub(crate) fn materialize_gather8<F: Field, R: AsRef<[Option<u8>]> + Sync>(
    table_groups: &[&Vec<Vec<F>>; 8],
    indices: &[R],
) -> Vec<Vec<F>> {
    let new_len = indices.first().map_or(0, |chunk| chunk.as_ref().len() / 8);
    (0..indices.len())
        .into_par_iter()
        .map(|chunk| {
            let row = indices[chunk].as_ref();
            (0..new_len)
                .map(|index| {
                    (0..8)
                        .map(|offset| {
                            row[8 * index + offset].map_or(F::zero(), |value| {
                                table_groups[offset][chunk][usize::from(value)]
                            })
                        })
                        .sum()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
}

#[inline(always)]
fn sparse_lookup<F: Field>(table: &[F], index: Option<u8>) -> F {
    index.map_or(F::zero(), |value| table[usize::from(value)])
}

#[inline(always)]
fn sparse_round1_pair<F: Field>(
    tables: &[Vec<F>],
    indices: &[&[Option<u8>]],
    chunk: usize,
    source: usize,
) -> (F, F) {
    (
        sparse_lookup(&tables[chunk], indices[chunk][source]),
        sparse_lookup(&tables[chunk], indices[chunk][source + 1]),
    )
}

#[inline(always)]
fn sparse_round2_pair<F: Field>(
    tables_0: &[Vec<F>],
    tables_1: &[Vec<F>],
    indices: &[&[Option<u8>]],
    chunk: usize,
    source: usize,
) -> (F, F) {
    let low = sparse_lookup(&tables_0[chunk], indices[chunk][source])
        + sparse_lookup(&tables_1[chunk], indices[chunk][source + 1]);
    let high = sparse_lookup(&tables_0[chunk], indices[chunk][source + 2])
        + sparse_lookup(&tables_1[chunk], indices[chunk][source + 3]);
    (low, high)
}

#[inline(always)]
fn sparse_round3_pair<F: Field>(
    tables_00: &[Vec<F>],
    tables_01: &[Vec<F>],
    tables_10: &[Vec<F>],
    tables_11: &[Vec<F>],
    indices: &[&[Option<u8>]],
    chunk: usize,
    source: usize,
) -> (F, F) {
    let low = sparse_lookup(&tables_00[chunk], indices[chunk][source])
        + sparse_lookup(&tables_10[chunk], indices[chunk][source + 1])
        + sparse_lookup(&tables_01[chunk], indices[chunk][source + 2])
        + sparse_lookup(&tables_11[chunk], indices[chunk][source + 3]);
    let high = sparse_lookup(&tables_00[chunk], indices[chunk][source + 4])
        + sparse_lookup(&tables_10[chunk], indices[chunk][source + 5])
        + sparse_lookup(&tables_01[chunk], indices[chunk][source + 6])
        + sparse_lookup(&tables_11[chunk], indices[chunk][source + 7]);
    (low, high)
}

fn scale_instruction_ra_tables<F: Field>(tables: &[Vec<F>], scalar: F) -> Vec<Vec<F>> {
    tables
        .par_iter()
        .map(|table| table.iter().map(|value| *value * scalar).collect())
        .collect()
}

fn eval_instruction_ra_product<F: Field>(pairs: &[(F, F)], evals: &mut [F]) {
    debug_assert_eq!(pairs.len(), evals.len());
    for (point_index, eval) in evals.iter_mut().enumerate() {
        if point_index + 1 == pairs.len() {
            *eval = pairs.iter().map(|(low, high)| *high - *low).product::<F>();
        } else {
            let point = F::from_u64((point_index + 1) as u64);
            *eval = pairs
                .iter()
                .map(|(low, high)| *low + (*high - *low) * point)
                .product::<F>();
        }
    }
}

#[inline(always)]
fn accumulate_instruction_ra_d4_products<F: Field>(
    weight: F,
    evals: &mut [F::Accumulator],
    a0: (F, F),
    a1: (F, F),
    a2: (F, F),
    a3: (F, F),
) {
    let (a1_eval, a2_eval, a_inf) = eval_linear_prod_2_internal_stage6(a0, a1);
    let a3_eval = extrapolate_quadratic_next(a1_eval, a2_eval, a_inf);
    let (b1_eval, b2_eval, b_inf) = eval_linear_prod_2_internal_stage6(a2, a3);
    let b3_eval = extrapolate_quadratic_next(b1_eval, b2_eval, b_inf);

    evals[0].fmadd(weight, a1_eval * b1_eval);
    evals[1].fmadd(weight, a2_eval * b2_eval);
    evals[2].fmadd(weight, a3_eval * b3_eval);
    evals[3].fmadd(weight, a_inf * b_inf);
}

#[inline(always)]
fn accumulate_instruction_ra_d4_product_terms<F: Field>(
    evals: &mut [F::Accumulator; 4],
    a0: (F, F),
    a1: (F, F),
    a2: (F, F),
    a3: (F, F),
) {
    let (a1_eval, a2_eval, a_inf) = eval_linear_prod_2_internal_stage6(a0, a1);
    let a3_eval = extrapolate_quadratic_next(a1_eval, a2_eval, a_inf);
    let (b1_eval, b2_eval, b_inf) = eval_linear_prod_2_internal_stage6(a2, a3);
    let b3_eval = extrapolate_quadratic_next(b1_eval, b2_eval, b_inf);

    evals[0].fmadd(a1_eval, b1_eval);
    evals[1].fmadd(a2_eval, b2_eval);
    evals[2].fmadd(a3_eval, b3_eval);
    evals[3].fmadd(a_inf, b_inf);
}

#[inline(always)]
fn eval_linear_prod_2_internal_stage6<F: Field>((p0, p1): (F, F), (q0, q1): (F, F)) -> (F, F, F) {
    let p_inf = p1 - p0;
    let p2 = p1 + p_inf;
    let q_inf = q1 - q0;
    let q2 = q1 + q_inf;
    (p1 * q1, p2 * q2, p_inf * q_inf)
}

#[inline(always)]
fn extrapolate_quadratic_next<F: Field>(eval_at_1: F, eval_at_2: F, eval_at_inf: F) -> F {
    let doubled = eval_at_2 + eval_at_inf;
    doubled + doubled - eval_at_1
}

enum InstructionRaVirtualChunks<'a, F: Field> {
    Dense(InstructionRaVirtualDenseChunks<'a, F>),
    Sparse(InstructionRaVirtualSparseChunks<'a, F>),
}

impl<F: Field> InstructionRaVirtualChunks<'_, F> {
    fn len(&self) -> usize {
        match self {
            Self::Dense(chunks) => chunks.len(),
            Self::Sparse(chunks) => chunks.len(),
        }
    }

    fn current_len(&self) -> usize {
        match self {
            Self::Dense(chunks) => chunks.current_len(),
            Self::Sparse(chunks) => chunks.current_len(),
        }
    }

    fn get(&self, chunk: usize, index: usize) -> F {
        match self {
            Self::Dense(chunks) => chunks.get(chunk, index),
            Self::Sparse(chunks) => chunks.get(chunk, index),
        }
    }

    fn get_pair(&self, chunk: usize, row: usize) -> (F, F) {
        match self {
            Self::Dense(chunks) => chunks.get_pair(chunk, row),
            Self::Sparse(chunks) => chunks.get_pair(chunk, row),
        }
    }

    fn final_sumcheck_claim(&self, chunk: usize) -> F {
        match self {
            Self::Dense(chunks) => chunks.final_sumcheck_claim(chunk),
            Self::Sparse(chunks) => chunks.final_sumcheck_claim(chunk),
        }
    }

    fn bind(&mut self, challenge: F, backend: &'static str) {
        match self {
            Self::Dense(chunks) => chunks.bind(challenge),
            Self::Sparse(chunks) => chunks.bind(challenge, backend),
        }
    }
}

fn bind_dense_evals_to_vec<F: Field>(values: &[F], challenge: F) -> Vec<F> {
    values
        .chunks_exact(2)
        .map(|pair| pair[0] + (pair[1] - pair[0]) * challenge)
        .collect()
}

struct InstructionRaVirtualStage6State<'a, F: Field> {
    relation: Stage6Relation,
    eq_cycle: Vec<F>,
    eq_scratch: Vec<F>,
    split_eq: Option<GruenSplitEqPolynomial<F>>,
    chunks: InstructionRaVirtualChunks<'a, F>,
    chunks_per_virtual: usize,
    gamma_powers: Vec<F>,
    gamma_powers_inv: Vec<F>,
    gamma_absorbed: bool,
    output_factor_offset: usize,
    outputs: Vec<FactorOutput>,
    active_scale: F,
    backend: &'static str,
}

impl<'a, F: Field> InstructionRaVirtualStage6State<'a, F> {
    #[expect(clippy::too_many_arguments)]
    fn new(
        relation: Stage6Relation,
        eq_cycle: Vec<F>,
        split_eq: Option<GruenSplitEqPolynomial<F>>,
        chunks: InstructionRaVirtualChunks<'a, F>,
        chunks_per_virtual: usize,
        gamma: F,
        gamma_absorbed: bool,
        output_factor_offset: usize,
        outputs: Vec<FactorOutput>,
        active_scale: F,
        degree_bound: usize,
        backend: &'static str,
    ) -> Result<Self, Stage6KernelError> {
        if chunks.len() == 0
            || chunks_per_virtual == 0
            || !chunks.len().is_multiple_of(chunks_per_virtual)
        {
            return Err(Stage6KernelError::InvalidInputLength {
                input: relation.symbol(),
                expected: chunks_per_virtual,
                actual: chunks.len(),
            });
        }
        if degree_bound < chunks_per_virtual + 1 {
            return Err(Stage6KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "RA virtual degree bound is too small",
            });
        }
        if degree_bound > 5 {
            return Err(Stage6KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "RA virtual degree bound is unsupported",
            });
        }
        if chunks.current_len() != eq_cycle.len() {
            return Err(Stage6KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "RA virtual chunks have inconsistent lengths",
            });
        }

        let virtual_count = chunks.len() / chunks_per_virtual;
        let mut gamma_powers = Vec::with_capacity(virtual_count);
        let mut gamma_powers_inv = Vec::with_capacity(virtual_count);
        let mut gamma_power = F::one();
        for _ in 0..virtual_count {
            gamma_powers.push(gamma_power);
            gamma_powers_inv.push(gamma_power.inverse().ok_or(
                Stage6KernelError::InvalidProof {
                    driver: relation.symbol(),
                    reason: "RA virtual gamma power is not invertible",
                },
            )?);
            gamma_power *= gamma;
        }

        Ok(Self {
            relation,
            eq_cycle,
            eq_scratch: Vec::new(),
            split_eq,
            chunks,
            chunks_per_virtual,
            gamma_powers,
            gamma_powers_inv,
            gamma_absorbed,
            output_factor_offset,
            outputs,
            active_scale,
            backend,
        })
    }

    fn round_poly(
        &self,
        previous_claim: F,
        relation: Stage6Relation,
    ) -> Result<UnivariatePoly<F>, Stage6KernelError> {
        if relation != self.relation {
            return Err(Stage6KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "wrong relation for RA virtual state",
            });
        }
        if self.chunks.current_len() == 0 || !self.chunks.current_len().is_power_of_two() {
            return Err(Stage6KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "RA virtual factor has invalid length",
            });
        }
        if let Some(split_eq) = &self.split_eq {
            let _span = trace_stage6_inner_spans().then(|| {
                tracing::info_span!(
                    "Stage6::ra_virtual.round_sparse",
                    relation = relation.symbol(),
                    chunks_per_virtual = self.chunks_per_virtual
                )
                .entered()
            });
            return self.round_poly_sparse(previous_claim, relation, split_eq);
        }
        let _span = trace_stage6_inner_spans().then(|| {
            tracing::info_span!(
                "Stage6::ra_virtual.round_dense",
                relation = relation.symbol(),
                chunks_per_virtual = self.chunks_per_virtual
            )
            .entered()
        });
        let first_len = self.eq_cycle.len();
        let coefficient_count = self.chunks_per_virtual + 2;
        let half = first_len / 2;
        let coefficient_accs = if half >= DENSE_BIND_PAR_THRESHOLD {
            (0..half)
                .into_par_iter()
                .fold(
                    || vec![F::Accumulator::default(); coefficient_count],
                    |mut row_coefficients, row| {
                        self.accumulate_row_coefficients(row, &mut row_coefficients);
                        row_coefficients
                    },
                )
                .reduce(
                    || vec![F::Accumulator::default(); coefficient_count],
                    |mut left, right| {
                        for (left, right) in left.iter_mut().zip(right) {
                            left.merge(right);
                        }
                        left
                    },
                )
        } else {
            let mut coefficients = vec![F::Accumulator::default(); coefficient_count];
            for row in 0..half {
                self.accumulate_row_coefficients(row, &mut coefficients);
            }
            coefficients
        };
        let mut coefficients = coefficient_accs
            .into_iter()
            .map(FieldAccumulator::reduce)
            .collect::<Vec<_>>();
        for coefficient in &mut coefficients {
            *coefficient *= self.active_scale;
        }
        let poly = UnivariatePoly::new(coefficients);
        check_round_claim(
            &poly,
            previous_claim,
            relation.symbol(),
            "stage6 relation input claim mismatch",
        )?;
        Ok(poly)
    }

    fn round_poly_sparse(
        &self,
        previous_claim: F,
        relation: Stage6Relation,
        split_eq: &GruenSplitEqPolynomial<F>,
    ) -> Result<UnivariatePoly<F>, Stage6KernelError> {
        if self.chunks_per_virtual == 4 {
            return self.round_poly_sparse_d4(previous_claim, relation, split_eq);
        }

        let evals = split_eq.fold_out_in(
            || vec![F::Accumulator::default(); self.chunks_per_virtual],
            |inner, row, _x_in, e_in| {
                self.accumulate_sparse_row_evals(row, e_in, inner);
            },
            |_x_out, e_out, inner| {
                inner
                    .into_iter()
                    .map(|acc| {
                        let mut outer = F::Accumulator::default();
                        outer.fmadd(e_out, acc.reduce());
                        outer
                    })
                    .collect::<Vec<_>>()
            },
            |mut left, right| {
                for (left, right) in left.iter_mut().zip(right) {
                    left.merge(right);
                }
                left
            },
        );
        let mut evals = evals
            .into_iter()
            .map(FieldAccumulator::reduce)
            .collect::<Vec<_>>();
        for eval in &mut evals {
            *eval *= self.active_scale;
        }
        let poly = split_eq.gruen_poly_from_evals(&evals, previous_claim);
        check_round_claim(
            &poly,
            previous_claim,
            relation.symbol(),
            "stage6 relation input claim mismatch",
        )?;
        Ok(poly)
    }

    fn round_poly_sparse_d4(
        &self,
        previous_claim: F,
        relation: Stage6Relation,
        split_eq: &GruenSplitEqPolynomial<F>,
    ) -> Result<UnivariatePoly<F>, Stage6KernelError> {
        debug_assert_eq!(self.chunks_per_virtual, 4);

        let e_out = split_eq.e_out_current();
        let e_in = split_eq.e_in_current();
        let in_bits = e_in.len().trailing_zeros() as usize;
        let eval_accs = (0..e_out.len())
            .into_par_iter()
            .map(|x_out| {
                let mut inner = [F::Accumulator::default(); 4];
                let base = x_out << in_bits;
                for (x_in, &e_in) in e_in.iter().enumerate() {
                    self.accumulate_sparse_row_evals_d4(base | x_in, e_in, &mut inner);
                }

                let mut outer = [F::Accumulator::default(); 4];
                for (outer, inner) in outer.iter_mut().zip(inner) {
                    outer.fmadd(e_out[x_out], inner.reduce());
                }
                outer
            })
            .reduce(
                || [F::Accumulator::default(); 4],
                |mut left, right| {
                    for (left, right) in left.iter_mut().zip(right) {
                        left.merge(right);
                    }
                    left
                },
            );

        let mut evals = eval_accs.map(FieldAccumulator::reduce).to_vec();
        for eval in &mut evals {
            *eval *= self.active_scale;
        }
        let poly = split_eq.gruen_poly_from_evals(&evals, previous_claim);
        check_round_claim(
            &poly,
            previous_claim,
            relation.symbol(),
            "stage6 relation input claim mismatch",
        )?;
        Ok(poly)
    }

    fn accumulate_sparse_row_evals(&self, row: usize, e_in: F, evals: &mut [F::Accumulator]) {
        if self.chunks_per_virtual == 4 {
            self.accumulate_sparse_row_evals_d4(row, e_in, evals);
            return;
        }

        let mut pairs = vec![(F::zero(), F::zero()); self.chunks_per_virtual];
        let mut product_evals = vec![F::zero(); self.chunks_per_virtual];
        for virtual_index in 0..self.virtual_count() {
            for (offset, pair) in pairs.iter_mut().enumerate() {
                let chunk = virtual_index * self.chunks_per_virtual + offset;
                *pair = (
                    self.chunks.get(chunk, 2 * row),
                    self.chunks.get(chunk, 2 * row + 1),
                );
            }
            eval_instruction_ra_product(&pairs, &mut product_evals);
            let weight = if self.gamma_absorbed {
                e_in
            } else {
                e_in * self.gamma_powers[virtual_index]
            };
            for (eval, product_eval) in evals.iter_mut().zip(&product_evals) {
                eval.fmadd(weight, *product_eval);
            }
        }
    }

    fn accumulate_sparse_row_evals_d4(&self, row: usize, e_in: F, evals: &mut [F::Accumulator]) {
        debug_assert_eq!(self.chunks_per_virtual, 4);
        debug_assert_eq!(evals.len(), 4);

        if self.gamma_absorbed {
            let mut product_accs = [F::Accumulator::default(); 4];
            if let InstructionRaVirtualChunks::Sparse(chunks) = &self.chunks {
                chunks.accumulate_d4_product_terms(row, self.virtual_count(), &mut product_accs);
            } else {
                for virtual_index in 0..self.virtual_count() {
                    let base = virtual_index * 4;

                    let a0 = self.chunks.get_pair(base, row);
                    let a1 = self.chunks.get_pair(base + 1, row);
                    let a2 = self.chunks.get_pair(base + 2, row);
                    let a3 = self.chunks.get_pair(base + 3, row);

                    accumulate_instruction_ra_d4_product_terms(&mut product_accs, a0, a1, a2, a3);
                }
            }
            for (eval, product_acc) in evals.iter_mut().zip(product_accs) {
                eval.fmadd(e_in, product_acc.reduce());
            }
            return;
        }

        for virtual_index in 0..self.virtual_count() {
            let base = virtual_index * 4;

            let a0 = self.chunks.get_pair(base, row);
            let a1 = self.chunks.get_pair(base + 1, row);
            let a2 = self.chunks.get_pair(base + 2, row);
            let a3 = self.chunks.get_pair(base + 3, row);

            let weight = if self.gamma_absorbed {
                e_in
            } else {
                e_in * self.gamma_powers[virtual_index]
            };
            accumulate_instruction_ra_d4_products(weight, evals, a0, a1, a2, a3);
        }
    }

    fn accumulate_row_coefficients(&self, row: usize, coefficients: &mut [F::Accumulator]) {
        let eq_low = self.eq_cycle[2 * row];
        let eq_slope = self.eq_cycle[2 * row + 1] - eq_low;
        for virtual_index in 0..self.virtual_count() {
            let mut term = [F::zero(); 6];
            let mut next = [F::zero(); 6];
            term[0] = self.gamma_powers[virtual_index];
            let mut degree = 0;
            for offset in 0..self.chunks_per_virtual {
                let chunk = virtual_index * self.chunks_per_virtual + offset;
                let low = self.chunks.get(chunk, 2 * row);
                let high = self.chunks.get(chunk, 2 * row + 1);
                let slope = high - low;
                multiply_linear_factor(&mut term, &mut next, &mut degree, low, slope);
            }
            multiply_linear_factor(&mut term, &mut next, &mut degree, eq_low, eq_slope);
            for (coefficient, term_coefficient) in coefficients.iter_mut().zip(term) {
                coefficient.acc_add(term_coefficient);
            }
        }
    }

    fn bind(&mut self, challenge: F) {
        if self.split_eq.is_none() {
            bind_dense_evals_reuse(&mut self.eq_cycle, &mut self.eq_scratch, challenge);
        }
        self.chunks.bind(challenge, self.backend);
        if let Some(split_eq) = &mut self.split_eq {
            split_eq.bind(challenge);
        }
    }

    fn final_relation_eval(&self, relation: Stage6Relation) -> Result<F, Stage6KernelError> {
        if relation != self.relation {
            return Err(Stage6KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "wrong relation for RA virtual state",
            });
        }
        let eq = if let Some(split_eq) = &self.split_eq {
            split_eq.current_scalar()
        } else {
            self.eq_cycle
                .first()
                .copied()
                .ok_or(Stage6KernelError::InvalidProof {
                    driver: relation.symbol(),
                    reason: "empty instruction RA virtual eq factor",
                })?
        };
        let mut virtual_sum = F::zero();
        for virtual_index in 0..self.virtual_count() {
            let mut product = if self.gamma_absorbed {
                F::one()
            } else {
                self.gamma_powers[virtual_index]
            };
            for offset in 0..self.chunks_per_virtual {
                let chunk = virtual_index * self.chunks_per_virtual + offset;
                product *= self.chunks.final_sumcheck_claim(chunk);
            }
            virtual_sum += product;
        }
        Ok(eq * virtual_sum)
    }

    fn virtual_count(&self) -> usize {
        self.chunks.len() / self.chunks_per_virtual
    }

    fn unscaled_factor_eval(&self, factor: usize) -> F {
        let mut eval = self.chunks.final_sumcheck_claim(factor);
        if self.gamma_absorbed && factor.is_multiple_of(self.chunks_per_virtual) {
            eval *= self.gamma_powers_inv[factor / self.chunks_per_virtual];
        }
        eval
    }

    fn final_evals(
        &self,
        relation: Stage6Relation,
    ) -> Result<Vec<Stage6NamedEval<F>>, Stage6KernelError> {
        self.outputs
            .iter()
            .map(|output| {
                let factor = output.factor.checked_sub(self.output_factor_offset).ok_or(
                    Stage6KernelError::InvalidProof {
                        driver: relation.symbol(),
                        reason: "RA virtual output factor underflow",
                    },
                )?;
                Ok(named_eval(
                    output.name,
                    output.oracle,
                    (factor < self.chunks.len())
                        .then(|| self.unscaled_factor_eval(factor))
                        .ok_or(Stage6KernelError::InvalidProof {
                            driver: relation.symbol(),
                            reason: "empty RA virtual factor",
                        })?,
                ))
            })
            .collect()
    }
}

fn bytecode_read_raf_state<'a, F: Field>(
    program: &'static Stage6CpuProgramPlan,
    claim: &Stage6SumcheckClaimPlan,
    inputs: &'a Stage6ProverInputs<'a, F>,
    store: &Stage6ValueStore<F>,
    active_scale: F,
) -> Result<Stage6ProverInstanceState<'a, F>, Stage6KernelError> {
    let witness = inputs
        .bytecode_read_raf
        .ok_or(Stage6KernelError::MissingKernelInput {
            kernel: "jolt_stage6_batched",
            input: "bytecode_read_raf",
        })?;
    if witness.bytecode_ra_chunks.is_empty()
        && matches!(witness.bytecode_ra_index_chunks, None | Some([]))
    {
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

    let chunk_lens = if witness.bytecode_ra_chunks.is_empty() {
        witness
            .bytecode_ra_chunk_lens
            .ok_or(Stage6KernelError::InvalidInputLength {
                input: "stage6.bytecode_read_raf.BytecodeRa",
                expected: 1,
                actual: 0,
            })?
            .to_vec()
    } else {
        let mut chunk_lens = Vec::with_capacity(witness.bytecode_ra_chunks.len());
        for chunk in witness.bytecode_ra_chunks {
            let rounds = log2_exact(chunk.len(), "stage6.bytecode_read_raf.BytecodeRa")?;
            let chunk_len =
                rounds
                    .checked_sub(log_t)
                    .ok_or(Stage6KernelError::InvalidInputLength {
                        input: "stage6.bytecode_read_raf.BytecodeRa",
                        expected: log_t,
                        actual: rounds,
                    })?;
            chunk_lens.push(chunk_len);
        }
        chunk_lens
    };
    let covered_address_len = chunk_lens.iter().sum::<usize>();
    require_operand_count(
        "stage6.bytecode_read_raf.address_chunks",
        log_k,
        covered_address_len,
    )?;

    let sparse_cycle_indices = match witness.bytecode_ra_index_chunks {
        Some(chunks) if !chunks.is_empty() => Some(bytecode_cycle_indices_from_sparse_chunks(
            chunks,
            &chunk_lens,
            log_t,
        )?),
        _ => None,
    };

    if let Some(bytecode_cycle_indices) = sparse_cycle_indices.or_else(|| {
        bytecode_cycle_indices_from_one_hot(witness.bytecode_ra_chunks, &chunk_lens, log_t)
    }) {
        let outputs = bytecode_read_raf_output_plans(program, chunk_lens.len())?;
        return BytecodeReadRafStage6State::new(
            witness.data,
            witness.bytecode_ra_chunks,
            witness.bytecode_ra_index_chunks,
            bytecode_cycle_indices,
            chunk_lens,
            store,
            log_k,
            log_t,
            active_scale,
            claim.degree,
            outputs,
        )
        .map(|state| Stage6ProverInstanceState::BytecodeReadRaf(Box::new(state)));
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

#[expect(clippy::too_many_arguments)]
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

fn booleanity_state<'a, F: Field>(
    program: &'static Stage6CpuProgramPlan,
    claim: &Stage6SumcheckClaimPlan,
    inputs: &'a Stage6ProverInputs<'a, F>,
    store: &Stage6ValueStore<F>,
    active_scale: F,
) -> Result<Stage6ProverInstanceState<'a, F>, Stage6KernelError> {
    let witness = inputs
        .booleanity
        .ok_or(Stage6KernelError::MissingKernelInput {
            kernel: "jolt_stage6_batched",
            input: "booleanity",
        })?;
    let log_t = stage6_trace_rounds(program)?;
    if let Some(index_chunks) = witness.index_chunks {
        if index_chunks.is_empty() {
            return Err(Stage6KernelError::InvalidInputLength {
                input: "stage6.booleanity.index_chunks",
                expected: 1,
                actual: 0,
            });
        }
        let trace_len = 1usize << log_t;
        for chunk in index_chunks {
            require_operand_count("stage6.booleanity.index_chunk", trace_len, chunk.len())?;
        }
        let log_k_chunk =
            claim
                .num_rounds
                .checked_sub(log_t)
                .ok_or(Stage6KernelError::InvalidInputLength {
                    input: "stage6.booleanity.input",
                    expected: log_t,
                    actual: claim.num_rounds,
                })?;
        let combined_r = booleanity_combined_point(store, log_t, log_k_chunk)?;
        require_operand_count(
            "stage6.booleanity.combined_point",
            claim.num_rounds,
            combined_r.len(),
        )?;
        let r_address = &combined_r[..log_k_chunk];
        let r_cycle = &combined_r[log_k_chunk..];
        return CoreBooleanityStage6State::new(
            r_address,
            r_cycle,
            index_chunks,
            witness.row_indices,
            store.scalar("stage6.booleanity.gamma")?,
            booleanity_output_plans(program, index_chunks.len())?,
            active_scale,
        )
        .map(|state| Stage6ProverInstanceState::CoreBooleanity(Box::new(state)));
    }

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
    let log_k_chunk =
        booleanity_rounds
            .checked_sub(log_t)
            .ok_or(Stage6KernelError::InvalidInputLength {
                input: "stage6.booleanity.trace_len",
                expected: log_t,
                actual: booleanity_rounds,
            })?;
    let combined_r = booleanity_combined_point(store, log_t, log_k_chunk)?;
    let mut eq_point = combined_r[..log_k_chunk].to_vec();
    eq_point.reverse();
    eq_point.extend(combined_r[log_k_chunk..].iter().rev().copied());
    eq_point.reverse();
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
    .map(Stage6ProverInstanceState::Booleanity)
}

fn booleanity_combined_point<F: Field>(
    store: &Stage6ValueStore<F>,
    log_t: usize,
    log_k_chunk: usize,
) -> Result<Vec<F>, Stage6KernelError> {
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
        log_k_chunk + log_t,
        combined_r.len(),
    )?;
    Ok(combined_r)
}

fn hamming_booleanity_state<F: Field>(
    program: &'static Stage6CpuProgramPlan,
    claim: &Stage6SumcheckClaimPlan,
    inputs: &Stage6ProverInputs<'_, F>,
    store: &Stage6ValueStore<F>,
    active_scale: F,
    backend: &'static str,
) -> Result<HammingBooleanityStage6State<F>, Stage6KernelError> {
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

    HammingBooleanityStage6State::new_with_backend(
        &lookup_output_point,
        witness.hamming_weight.to_vec(),
        output,
        active_scale,
        claim.degree,
        backend,
    )
}

fn inc_claim_reduction_state<F: Field>(
    program: &'static Stage6CpuProgramPlan,
    claim: &Stage6SumcheckClaimPlan,
    inputs: &Stage6ProverInputs<'_, F>,
    store: &Stage6ValueStore<F>,
    active_scale: F,
    backend: &'static str,
) -> Result<IncClaimReductionStage6State<F>, Stage6KernelError> {
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

    let (eq_ram_combined, eq_rd_combined) = rayon::join(
        || {
            let (eq_ram_stage2, eq_ram_stage4) = rayon::join(
                || EqPolynomial::<F>::evals(ram_inc_stage2, None),
                || EqPolynomial::<F>::evals(ram_inc_stage4, None),
            );
            eq_ram_stage2
                .par_iter()
                .zip(eq_ram_stage4.par_iter())
                .map(|(&stage2, &stage4)| stage2 + gamma * stage4)
                .collect::<Vec<_>>()
        },
        || {
            let (eq_rd_stage4, eq_rd_stage5) = rayon::join(
                || EqPolynomial::<F>::evals(rd_inc_stage4, None),
                || EqPolynomial::<F>::evals(rd_inc_stage5, None),
            );
            eq_rd_stage4
                .par_iter()
                .zip(eq_rd_stage5.par_iter())
                .map(|(&stage4, &stage5)| stage4 + gamma * stage5)
                .collect::<Vec<_>>()
        },
    );
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

    let ram_inc = witness.ram_inc.to_vec();
    let rd_inc = witness.rd_inc.to_vec();
    #[cfg(feature = "cuda")]
    let cuda = if backend == "cuda" {
        cuda::CudaIncState::new(&eq_ram_combined, &ram_inc, &eq_rd_combined, &rd_inc, gamma2)
    } else {
        None
    };
    #[cfg(not(feature = "cuda"))]
    let _ = backend;
    Ok(IncClaimReductionStage6State {
        eq_ram: eq_ram_combined,
        eq_ram_scratch: Vec::new(),
        ram_inc,
        ram_inc_scratch: Vec::new(),
        eq_rd: eq_rd_combined,
        eq_rd_scratch: Vec::new(),
        rd_inc,
        rd_inc_scratch: Vec::new(),
        gamma2,
        outputs: [
            factor_output_by_name(program, "stage6.inc_claim_reduction.eval.RamInc", 1)?,
            factor_output_by_name(program, "stage6.inc_claim_reduction.eval.RdInc", 3)?,
        ],
        active_scale,
        #[cfg(feature = "cuda")]
        cuda,
    })
}

fn ram_ra_virtual_state<'a, F: Field>(
    program: &'static Stage6CpuProgramPlan,
    claim: &Stage6SumcheckClaimPlan,
    inputs: &'a Stage6ProverInputs<'a, F>,
    store: &Stage6ValueStore<F>,
    active_scale: F,
    backend: &'static str,
) -> Result<InstructionRaVirtualStage6State<'a, F>, Stage6KernelError> {
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

    let outputs = ram_ra_virtual_output_plans(program, witness.ram_ra_chunks.len())?;

    InstructionRaVirtualStage6State::new(
        Stage6Relation::RamRaVirtual,
        eq_cycle,
        Some(GruenSplitEqPolynomial::new(
            r_cycle,
            BindingOrder::LowToHigh,
        )),
        InstructionRaVirtualChunks::Dense(InstructionRaVirtualDenseChunks::borrowed(
            witness.ram_ra_chunks,
        )),
        witness.ram_ra_chunks.len(),
        F::one(),
        false,
        1,
        outputs,
        active_scale,
        claim.degree,
        backend,
    )
}

fn instruction_ra_virtual_state<'a, F: Field>(
    program: &'static Stage6CpuProgramPlan,
    claim: &Stage6SumcheckClaimPlan,
    inputs: &'a Stage6ProverInputs<'a, F>,
    store: &Stage6ValueStore<F>,
    active_scale: F,
    backend: &'static str,
) -> Result<InstructionRaVirtualStage6State<'a, F>, Stage6KernelError> {
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

    let chunks_per_virtual = witness.instruction_ra_chunks.len() / witness.virtual_count;
    let gamma = store.scalar("stage6.instruction_ra_virtual.gamma")?;
    let outputs =
        instruction_ra_virtual_output_plans(program, witness.instruction_ra_chunks.len())?;
    let sparse_chunks = match witness.instruction_ra_index_chunks {
        Some(index_chunks) if index_chunks.len() == witness.instruction_ra_chunks.len() => {
            Some(instruction_ra_virtual_sparse_chunks(
                program,
                store,
                index_chunks,
                chunks_per_virtual,
                gamma,
                trace_rounds,
            )?)
        }
        _ => None,
    };

    if let Some(sparse_chunks) = sparse_chunks {
        return InstructionRaVirtualStage6State::new(
            Stage6Relation::InstructionRaVirtual,
            eq_cycle,
            Some(GruenSplitEqPolynomial::new(
                r_cycle,
                BindingOrder::LowToHigh,
            )),
            InstructionRaVirtualChunks::Sparse(sparse_chunks),
            chunks_per_virtual,
            gamma,
            true,
            0,
            outputs,
            active_scale,
            claim.degree,
            backend,
        );
    }

    InstructionRaVirtualStage6State::new(
        Stage6Relation::InstructionRaVirtual,
        eq_cycle,
        None,
        InstructionRaVirtualChunks::Dense(InstructionRaVirtualDenseChunks::borrowed(
            witness.instruction_ra_chunks,
        )),
        chunks_per_virtual,
        gamma,
        false,
        0,
        outputs,
        active_scale,
        claim.degree,
        backend,
    )
}

fn instruction_ra_virtual_sparse_chunks<'a, F: Field>(
    program: &'static Stage6CpuProgramPlan,
    store: &Stage6ValueStore<F>,
    index_chunks: &'a [&'a [Option<u8>]],
    chunks_per_virtual: usize,
    gamma: F,
    trace_rounds: usize,
) -> Result<InstructionRaVirtualSparseChunks<'a, F>, Stage6KernelError> {
    let log_k_chunk = instruction_ra_virtual_log_k_chunk(program, trace_rounds)?;
    let virtual_address_len = chunks_per_virtual * log_k_chunk;
    let mut gamma_powers = Vec::with_capacity(index_chunks.len().div_ceil(chunks_per_virtual));
    let mut gamma_power = F::one();
    for _ in 0..gamma_powers.capacity() {
        gamma_powers.push(gamma_power);
        gamma_power *= gamma;
    }
    let tables = (0..index_chunks.len())
        .map(|chunk_index| {
            let virtual_index = chunk_index / chunks_per_virtual;
            let local_chunk = chunk_index % chunks_per_virtual;
            let symbol =
                format!("stage6.input.stage5.instruction_read_raf.InstructionRa_{virtual_index}");
            let point = store
                .try_point(&symbol)
                .ok_or(Stage6KernelError::MissingValue {
                    symbol: "stage6.input.stage5.instruction_read_raf.InstructionRa_",
                })?;
            if point.len() < virtual_address_len {
                return Err(Stage6KernelError::InvalidInputLength {
                    input: "stage6.input.stage5.instruction_read_raf.InstructionRa_",
                    expected: virtual_address_len,
                    actual: point.len(),
                });
            }
            let start = local_chunk * log_k_chunk;
            let end = start + log_k_chunk;
            let scaling_factor = (local_chunk == 0 && gamma_powers[virtual_index] != F::one())
                .then_some(gamma_powers[virtual_index]);
            Ok(EqPolynomial::<F>::evals(&point[start..end], scaling_factor))
        })
        .collect::<Result<Vec<_>, _>>()?;

    for (indices, table) in index_chunks.iter().zip(&tables) {
        require_operand_count(
            "stage6.instruction_ra_virtual.index_chunk",
            1usize << trace_rounds,
            indices.len(),
        )?;
        for index in *indices {
            if usize::from(index.unwrap_or(0)) >= table.len() {
                return Err(Stage6KernelError::InvalidInputLength {
                    input: "stage6.instruction_ra_virtual.index_chunk",
                    expected: table.len(),
                    actual: usize::from(index.unwrap_or(0)) + 1,
                });
            }
        }
    }

    Ok(InstructionRaVirtualSparseChunks::new(tables, index_chunks))
}

fn instruction_ra_virtual_log_k_chunk(
    program: &'static Stage6CpuProgramPlan,
    trace_rounds: usize,
) -> Result<usize, Stage6KernelError> {
    program
        .opening_claims
        .iter()
        .find(|claim| claim.symbol == "stage6.instruction_ra_virtual.opening.InstructionRa_0")
        .and_then(|claim| claim.point_arity.checked_sub(trace_rounds))
        .ok_or(Stage6KernelError::MissingValue {
            symbol: "stage6.instruction_ra_virtual.opening.InstructionRa_0",
        })
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

#[inline]
fn check_round_claim<F: Field>(
    poly: &UnivariatePoly<F>,
    previous_claim: F,
    driver: &'static str,
    reason: &'static str,
) -> Result<(), Stage6KernelError> {
    #[cfg(debug_assertions)]
    {
        if poly.evaluate(F::zero()) + poly.evaluate(F::one()) != previous_claim {
            return Err(Stage6KernelError::InvalidProof { driver, reason });
        }
    }
    #[cfg(not(debug_assertions))]
    {
        let _ = (poly, previous_claim, driver, reason);
    }
    Ok(())
}

fn append_opening_claims<F, T>(
    program: &'static Stage6CpuProgramPlan,
    store: &mut Stage6ValueStore<F>,
    transcript: &mut T,
    evals: &[Stage6NamedEval<F>],
) -> Result<Vec<Stage6OpeningClaimValue<F>>, Stage6KernelError>
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
            let duplicate = seen.iter().any(|(kind, oracle, seen_point)| {
                *kind == claim.claim_kind && *oracle == claim.oracle && seen_point == &point
            });
            let value = store.scalar(claim.eval_source)?;
            if !duplicate {
                append_labeled_scalar(transcript, "opening_claim", &value);
                seen.push((claim.claim_kind, claim.oracle, point.clone()));
            }
            opening_claims.push(Stage6OpeningClaimValue {
                symbol: claim.symbol,
                oracle: claim.oracle,
                domain: claim.domain,
                claim_kind: claim.claim_kind,
                point: point.clone(),
                eval: value,
            });
        }
    }
    Ok(opening_claims)
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

fn bytecode_cycle_indices_from_sparse_chunks(
    chunks: &[&[Option<u8>]],
    chunk_lens: &[usize],
    log_t: usize,
) -> Result<Vec<usize>, Stage6KernelError> {
    validate_bytecode_ra_index_chunks(chunks, chunk_lens, log_t)?;
    let trace_len =
        1usize
            .checked_shl(log_t as u32)
            .ok_or(Stage6KernelError::InvalidInputLength {
                input: "stage6.bytecode_read_raf.BytecodeRaIndex",
                expected: usize::BITS as usize,
                actual: log_t,
            })?;
    let mut indices = vec![0usize; trace_len];
    let mut remaining_address_bits = chunk_lens.iter().sum::<usize>();
    for (chunk, &chunk_len) in chunks.iter().zip(chunk_lens) {
        remaining_address_bits = remaining_address_bits.checked_sub(chunk_len).ok_or(
            Stage6KernelError::InvalidInputLength {
                input: "stage6.bytecode_read_raf.BytecodeRaIndex",
                expected: chunk_len,
                actual: remaining_address_bits,
            },
        )?;
        for (cycle, index) in chunk.iter().enumerate() {
            let Some(index) = *index else {
                return Err(Stage6KernelError::InvalidProof {
                    driver: Stage6Relation::BytecodeReadRaf.symbol(),
                    reason: "bytecode read RAF sparse index is missing",
                });
            };
            indices[cycle] |= usize::from(index) << remaining_address_bits;
        }
    }
    Ok(indices)
}

fn validate_bytecode_ra_index_chunks(
    chunks: &[&[Option<u8>]],
    chunk_lens: &[usize],
    log_t: usize,
) -> Result<(), Stage6KernelError> {
    require_operand_count(
        "stage6.bytecode_read_raf.BytecodeRaIndex",
        chunk_lens.len(),
        chunks.len(),
    )?;
    let trace_len =
        1usize
            .checked_shl(log_t as u32)
            .ok_or(Stage6KernelError::InvalidInputLength {
                input: "stage6.bytecode_read_raf.BytecodeRaIndex",
                expected: usize::BITS as usize,
                actual: log_t,
            })?;
    for (chunk, &chunk_len) in chunks.iter().zip(chunk_lens) {
        require_operand_count(
            "stage6.bytecode_read_raf.BytecodeRaIndex",
            trace_len,
            chunk.len(),
        )?;
        let chunk_domain =
            1usize
                .checked_shl(chunk_len as u32)
                .ok_or(Stage6KernelError::InvalidInputLength {
                    input: "stage6.bytecode_read_raf.BytecodeRaIndex",
                    expected: usize::BITS as usize,
                    actual: chunk_len,
                })?;
        for index in *chunk {
            let Some(index) = *index else {
                return Err(Stage6KernelError::InvalidProof {
                    driver: Stage6Relation::BytecodeReadRaf.symbol(),
                    reason: "bytecode read RAF sparse index is missing",
                });
            };
            let index = usize::from(index);
            if index >= chunk_domain {
                return Err(Stage6KernelError::InvalidInputLength {
                    input: "stage6.bytecode_read_raf.BytecodeRaIndex",
                    expected: chunk_domain,
                    actual: index + 1,
                });
            }
        }
    }
    Ok(())
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

#[expect(
    clippy::too_many_arguments,
    reason = "bytecode stage evaluator mirrors generated stage challenge layout"
)]
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
        let bit_value = usize::from(*bit != F::zero());
        (index << 1) | bit_value
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

fn normalize_stage6_booleanity_point<F: Field>(
    program: &'static Stage6CpuProgramPlan,
    point: &[F],
) -> Result<Vec<F>, Stage6KernelError> {
    let log_t = stage6_trace_rounds(program)?;
    let log_k = point
        .len()
        .checked_sub(log_t)
        .ok_or(Stage6KernelError::InvalidInputLength {
            input: "stage6.booleanity.point",
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
        0,
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
    previous_claim: F,
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

    let eval_count = degree_bound;
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
    Ok(UnivariatePoly::from_evals_and_hint(previous_claim, &evals))
}

pub(crate) fn accumulate_dense_row_evaluations<F: Field>(
    factors: &[Vec<F>],
    terms: &[DenseTerm<F>],
    row: usize,
    evals: &mut [F],
) {
    for (point_index, eval) in evals.iter_mut().enumerate() {
        let point = if point_index == 0 {
            F::zero()
        } else {
            F::from_u64((point_index + 1) as u64)
        };
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
                artifacts
                    .opening_claims
                    .extend(output.opening_claims.clone());
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
    artifacts.opening_claims = executor.opening_claim_values(program)?;
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
#[expect(
    clippy::expect_used,
    clippy::manual_is_multiple_of,
    clippy::unwrap_used,
    reason = "tests use direct assertions and unwraps for fixture setup"
)]
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
                bytecode_ra_chunk_lens: None,
                bytecode_ra_index_chunks: None,
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
    fn bytecode_read_raf_prover_accepts_sparse_index_chunks() {
        let entries = bytecode_entries();
        let data = Stage6BytecodeReadRafData {
            entries: &entries,
            entry_bytecode_index: 2,
            num_lookup_tables: 2,
        };
        let bytecode_ra_0 = frs(&[0, 1, 1, 0]);
        let bytecode_ra_1 = frs(&[1, 0, 0, 1]);
        let bytecode_ra_chunks: [&[Fr]; 2] = [&bytecode_ra_0, &bytecode_ra_1];
        let bytecode_ra_index_0 = [Some(1u8), Some(0u8)];
        let bytecode_ra_index_1 = [Some(0u8), Some(1u8)];
        let bytecode_ra_index_chunks: [&[Option<u8>]; 2] =
            [&bytecode_ra_index_0, &bytecode_ra_index_1];
        let bytecode_ra_chunk_lens = [1usize, 1usize];
        let sparse_only_bytecode_ra_chunks: [&[Fr]; 0] = [];
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
                bytecode_ra_chunks: &sparse_only_bytecode_ra_chunks,
                bytecode_ra_chunk_lens: Some(&bytecode_ra_chunk_lens),
                bytecode_ra_index_chunks: Some(&bytecode_ra_index_chunks),
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
        .expect("bytecode read RAF prover accepts sparse index chunks");

        assert_eq!(
            bytecode_cycle_indices_from_sparse_chunks(&bytecode_ra_index_chunks, &[1, 1], 1)
                .expect("sparse bytecode indices are valid"),
            vec![2, 1]
        );

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
        .expect("proof-carrying verifier accepts sparse bytecode read RAF output");

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
                bytecode_ra_chunk_lens: None,
                bytecode_ra_index_chunks: None,
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
        let prover_inputs =
            Stage6ProverInputs::new(&opening_inputs).with_booleanity(Stage6BooleanityWitness {
                chunks: &chunks,
                index_chunks: None,
                row_indices: None,
            });
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
        let prover_inputs =
            Stage6ProverInputs::new(&opening_inputs).with_booleanity(Stage6BooleanityWitness {
                chunks: &chunks,
                index_chunks: None,
                row_indices: None,
            });
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
    fn stage6_witness_helper_adapts_kernel_opening_inputs() {
        let params = Stage6WitnessParams {
            trace_len: 2,
            log_k_chunk: 1,
            log_k_bytecode: 1,
            log_k_ram: 1,
            lookups_ra_virtual_log_k_chunk: 1,
            instruction_d: 1,
            instruction_ra_virtual_d: 1,
            bytecode_d: 1,
            ram_d: 1,
        };
        let cycle_inputs = [
            CycleInput {
                dense: [2, 3],
                one_hot: [Some(1), Some(0), Some(1)],
            },
            CycleInput::PADDING,
        ];
        let opening_inputs = [
            Stage6OpeningInputValue {
                symbol: "stage6.input.stage5.ram_ra_claim_reduction.RamRa",
                point: frs(&[5]),
                eval: Fr::from_u64(0),
            },
            Stage6OpeningInputValue {
                symbol: "stage6.input.stage5.instruction_read_raf.InstructionRa_0",
                point: frs(&[7]),
                eval: Fr::from_u64(0),
            },
        ];

        let witness = stage6_witness_from_opening_inputs(params, &cycle_inputs, &opening_inputs);

        assert_eq!(witness.ram_ra_virtual.len(), 1);
        assert_eq!(witness.instruction_ra_virtual.len(), 1);
        assert_eq!(witness.rd_inc, vec![Fr::from_u64(2), Fr::from_u64(0)]);
        assert_eq!(witness.ram_inc, vec![Fr::from_u64(3), Fr::from_u64(0)]);
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
                instruction_ra_index_chunks: None,
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
                instruction_ra_index_chunks: None,
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
                opening_claims: Vec::new(),
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
