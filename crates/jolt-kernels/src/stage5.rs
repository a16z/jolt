//! Stage 5 coarse-kernel ABI used by Bolt-generated Jolt prover code.

#![expect(
    clippy::large_enum_variant,
    reason = "kernel states stay inline to avoid boxing hot prover state"
)]
#![expect(
    clippy::too_many_arguments,
    reason = "kernel constructors mirror generated staged protocol inputs"
)]

use std::collections::HashMap;
use std::error::Error;
use std::fmt::{self, Display, Formatter};

use crate::dense::{bind_dense_evals_reuse, DENSE_BIND_PAR_THRESHOLD};
use jolt_field::{AdditiveAccumulator, Field, RingAccumulator};
use jolt_lookup_tables::{
    tables::{
        prefixes::{PrefixEval, ALL_PREFIXES, NUM_PREFIXES},
        Suffixes,
    },
    uninterleave_bits, LookupBits, LookupTableKind,
};
use jolt_poly::{bind_high_to_low, EqPolynomial, UnivariatePoly};
use jolt_transcript::{Label, LabelWithCount, Transcript};
use jolt_witness::Stage45SparseTraceWitness;
use rayon::prelude::*;

type PrefixPairEvals<F> = ([PrefixEval<F>; NUM_PREFIXES], [PrefixEval<F>; NUM_PREFIXES]);

pub use crate::stage4::{
    Stage4ChallengeVector as Stage5ChallengeVector, Stage4CpuProgramPlan as Stage5CpuProgramPlan,
    Stage4ExecutionArtifacts as Stage5ExecutionArtifacts,
    Stage4ExecutionMode as Stage5ExecutionMode, Stage4FieldConstantPlan as Stage5FieldConstantPlan,
    Stage4FieldExprPlan as Stage5FieldExprPlan, Stage4KernelPlan as Stage5KernelPlan,
    Stage4NamedEval as Stage5NamedEval, Stage4OpeningBatchPlan as Stage5OpeningBatchPlan,
    Stage4OpeningClaimEqualityPlan as Stage5OpeningClaimEqualityPlan,
    Stage4OpeningClaimPlan as Stage5OpeningClaimPlan,
    Stage4OpeningClaimValue as Stage5OpeningClaimValue,
    Stage4OpeningInputPlan as Stage5OpeningInputPlan,
    Stage4OpeningInputValue as Stage5OpeningInputValue, Stage4Params as Stage5Params,
    Stage4PointConcatPlan as Stage5PointConcatPlan, Stage4PointSlicePlan as Stage5PointSlicePlan,
    Stage4ProgramStepPlan as Stage5ProgramStepPlan, Stage4Proof as Stage5Proof,
    Stage4SumcheckBatchPlan as Stage5SumcheckBatchPlan,
    Stage4SumcheckClaimPlan as Stage5SumcheckClaimPlan,
    Stage4SumcheckDriverPlan as Stage5SumcheckDriverPlan,
    Stage4SumcheckEvalPlan as Stage5SumcheckEvalPlan,
    Stage4SumcheckInstanceResultPlan as Stage5SumcheckInstanceResultPlan,
    Stage4SumcheckOutput as Stage5SumcheckOutput,
    Stage4TranscriptAbsorbBytesPlan as Stage5TranscriptAbsorbBytesPlan,
    Stage4TranscriptSqueezePlan as Stage5TranscriptSqueezePlan,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage5Relation {
    InstructionReadRaf,
    RamRaClaimReduction,
    RegistersValEvaluation,
    Batched,
}

impl Stage5Relation {
    pub fn from_symbol(symbol: &str) -> Option<Self> {
        match symbol {
            "jolt.stage5.instruction_read_raf" => Some(Self::InstructionReadRaf),
            "jolt.stage5.ram_ra_claim_reduction" => Some(Self::RamRaClaimReduction),
            "jolt.stage5.registers_val_evaluation" => Some(Self::RegistersValEvaluation),
            "jolt.stage5.batched" => Some(Self::Batched),
            _ => None,
        }
    }

    pub fn symbol(self) -> &'static str {
        match self {
            Self::InstructionReadRaf => "jolt.stage5.instruction_read_raf",
            Self::RamRaClaimReduction => "jolt.stage5.ram_ra_claim_reduction",
            Self::RegistersValEvaluation => "jolt.stage5.registers_val_evaluation",
            Self::Batched => "jolt.stage5.batched",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage5KernelAbi {
    InstructionReadRaf,
    RamRaClaimReduction,
    RegistersValEvaluation,
    Batched,
}

impl Stage5KernelAbi {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "jolt_stage5_instruction_read_raf" => Some(Self::InstructionReadRaf),
            "jolt_stage5_ram_ra_claim_reduction" => Some(Self::RamRaClaimReduction),
            "jolt_stage5_registers_val_evaluation" => Some(Self::RegistersValEvaluation),
            "jolt_stage5_batched" => Some(Self::Batched),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::InstructionReadRaf => "jolt_stage5_instruction_read_raf",
            Self::RamRaClaimReduction => "jolt_stage5_ram_ra_claim_reduction",
            Self::RegistersValEvaluation => "jolt_stage5_registers_val_evaluation",
            Self::Batched => "jolt_stage5_batched",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage5KernelError {
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
        expected: Stage5ExecutionMode,
        actual: Stage5ExecutionMode,
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

impl Display for Stage5KernelError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingClaim { batch, claim } => {
                write!(
                    formatter,
                    "stage5 batch @{batch} references missing claim @{claim}"
                )
            }
            Self::MissingValue { symbol } => {
                write!(formatter, "stage5 value @{symbol} is not available")
            }
            Self::MissingDriver { driver } => {
                write!(formatter, "stage5 driver @{driver} is not available")
            }
            Self::MissingKernel { driver, kernel } => {
                write!(
                    formatter,
                    "stage5 driver @{driver} references missing kernel @{kernel}"
                )
            }
            Self::MissingBatch { driver, batch } => {
                write!(
                    formatter,
                    "stage5 driver @{driver} references missing batch @{batch}"
                )
            }
            Self::UnknownRelation { relation } => {
                write!(formatter, "unknown stage5 relation `{relation}`")
            }
            Self::UnknownKernelAbi { abi } => {
                write!(formatter, "unknown stage5 kernel ABI `{abi}`")
            }
            Self::PlanCountMismatch {
                artifact,
                expected,
                actual,
            } => {
                write!(
                    formatter,
                    "stage5 {artifact} plan count mismatch: expected {expected}, got {actual}"
                )
            }
            Self::InvalidInputLength {
                input,
                expected,
                actual,
            } => {
                write!(
                    formatter,
                    "stage5 input `{input}` has length {actual}, expected {expected}"
                )
            }
            Self::UnsupportedFieldExpr { symbol, formula } => {
                write!(
                    formatter,
                    "stage5 field expr @{symbol} uses unsupported formula `{formula}`"
                )
            }
            Self::KernelNotImplemented { abi } => {
                write!(formatter, "stage5 kernel ABI `{abi}` is not implemented")
            }
            Self::WrongExecutorMode {
                driver,
                expected,
                actual,
            } => {
                write!(
                    formatter,
                    "stage5 driver @{driver} expected {expected:?} executor, got {actual:?}"
                )
            }
            Self::MissingProof { driver } => {
                write!(formatter, "stage5 proof for driver @{driver} is missing")
            }
            Self::MissingKernelInput { kernel, input } => {
                write!(
                    formatter,
                    "stage5 kernel `{kernel}` is missing required input `{input}`"
                )
            }
            Self::InvalidProgramStep { symbol, kind } => {
                write!(
                    formatter,
                    "stage5 program step @{symbol} has invalid kind `{kind}`"
                )
            }
            Self::InvalidProof { driver, reason } => {
                write!(
                    formatter,
                    "stage5 proof for driver @{driver} is invalid: {reason}"
                )
            }
        }
    }
}

impl Error for Stage5KernelError {}

#[derive(Clone, Copy)]
pub struct Stage5RegistersValWitness<'a, F: Field> {
    pub register_count: usize,
    pub trace_len: usize,
    pub rd_inc: &'a [F],
    pub rd_wa: &'a [F],
    pub rd_write_addresses: Option<&'a [Option<usize>]>,
}

#[derive(Clone, Copy)]
pub struct Stage5RamRaWitness<'a, F: Field> {
    pub ram_k: usize,
    pub trace_len: usize,
    pub ram_ra: &'a [F],
    pub remapped_addresses: Option<&'a [Option<usize>]>,
}

#[derive(Clone, Copy)]
pub struct Stage5InstructionReadRafWitness<'a> {
    pub trace_len: usize,
    pub lookup_indices: &'a [u128],
    pub lookup_table_indices: &'a [Option<usize>],
    pub is_interleaved_operands: &'a [bool],
    pub ra_virtual_log_k_chunk: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5InstructionReadRafEvaluations<F: Field> {
    pub lookup_table_flags: Vec<F>,
    pub instruction_ra: Vec<F>,
    pub instruction_raf_flag: F,
}

#[derive(Clone, Copy)]
pub struct Stage5ProverInputs<'a, F: Field> {
    pub opening_inputs: &'a [Stage5OpeningInputValue<F>],
    pub instruction_read_raf: Option<Stage5InstructionReadRafWitness<'a>>,
    pub ram_ra: Option<Stage5RamRaWitness<'a, F>>,
    pub registers_val: Option<Stage5RegistersValWitness<'a, F>>,
}

impl<'a, F: Field> Stage5ProverInputs<'a, F> {
    pub fn new(opening_inputs: &'a [Stage5OpeningInputValue<F>]) -> Self {
        Self {
            opening_inputs,
            instruction_read_raf: None,
            ram_ra: None,
            registers_val: None,
        }
    }

    pub fn empty() -> Self {
        Self {
            opening_inputs: &[],
            instruction_read_raf: None,
            ram_ra: None,
            registers_val: None,
        }
    }

    pub fn with_instruction_read_raf(
        mut self,
        instruction_read_raf: Stage5InstructionReadRafWitness<'a>,
    ) -> Self {
        self.instruction_read_raf = Some(instruction_read_raf);
        self
    }

    pub fn with_ram_ra(mut self, ram_ra: Stage5RamRaWitness<'a, F>) -> Self {
        self.ram_ra = Some(ram_ra);
        self
    }

    pub fn with_registers_val(mut self, registers_val: Stage5RegistersValWitness<'a, F>) -> Self {
        self.registers_val = Some(registers_val);
        self
    }

    pub fn with_sparse_trace_witness(
        self,
        trace_len: usize,
        ram_k: usize,
        register_count: usize,
        lookup_indices: &'a [u128],
        lookup_table_indices: &'a [Option<usize>],
        is_interleaved_operands: &'a [bool],
        ra_virtual_log_k_chunk: usize,
        remapped_addresses: &'a [Option<usize>],
        rd_inc: &'a [F],
        rd_write_addresses: &'a [Option<usize>],
    ) -> Self {
        self.with_instruction_read_raf(Stage5InstructionReadRafWitness {
            trace_len,
            lookup_indices,
            lookup_table_indices,
            is_interleaved_operands,
            ra_virtual_log_k_chunk,
        })
        .with_ram_ra(Stage5RamRaWitness {
            ram_k,
            trace_len,
            ram_ra: &[],
            remapped_addresses: Some(remapped_addresses),
        })
        .with_registers_val(Stage5RegistersValWitness {
            register_count,
            trace_len,
            rd_inc,
            rd_wa: &[],
            rd_write_addresses: Some(rd_write_addresses),
        })
    }

    pub fn with_stage45_sparse_trace_witness(
        self,
        trace_len: usize,
        ram_k: usize,
        register_count: usize,
        lookup_indices: &'a [u128],
        lookup_table_indices: &'a [Option<usize>],
        is_interleaved_operands: &'a [bool],
        ra_virtual_log_k_chunk: usize,
        witness: &'a Stage45SparseTraceWitness<F>,
    ) -> Self {
        self.with_sparse_trace_witness(
            trace_len,
            ram_k,
            register_count,
            lookup_indices,
            lookup_table_indices,
            is_interleaved_operands,
            ra_virtual_log_k_chunk,
            &witness.ram_addresses,
            &witness.rd_inc,
            &witness.rd_write_addresses,
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Stage5KernelContext<'a> {
    pub mode: Stage5ExecutionMode,
    pub program: &'static Stage5CpuProgramPlan,
    pub kernel: &'a Stage5KernelPlan,
    pub batch: &'a Stage5SumcheckBatchPlan,
    pub driver: &'a Stage5SumcheckDriverPlan,
}

impl Stage5KernelContext<'_> {
    pub fn relation_kind(&self) -> Result<Stage5Relation, Stage5KernelError> {
        Stage5Relation::from_symbol(self.kernel.relation).ok_or(
            Stage5KernelError::UnknownRelation {
                relation: self.kernel.relation,
            },
        )
    }

    pub fn abi_kind(&self) -> Result<Stage5KernelAbi, Stage5KernelError> {
        Stage5KernelAbi::from_name(self.kernel.abi).ok_or(Stage5KernelError::UnknownKernelAbi {
            abi: self.kernel.abi,
        })
    }

    pub fn batch_claims(&self) -> Result<Vec<&'static Stage5SumcheckClaimPlan>, Stage5KernelError> {
        self.batch
            .claim_operands
            .iter()
            .map(|symbol| {
                self.program
                    .claim(symbol)
                    .ok_or(Stage5KernelError::MissingClaim {
                        batch: self.batch.symbol,
                        claim: symbol,
                    })
            })
            .collect()
    }
}

pub trait Stage5KernelExecutor<F: Field> {
    fn observe_challenge_vector(
        &mut self,
        _plan: &'static Stage5TranscriptSqueezePlan,
        _values: &[F],
    ) -> Result<(), Stage5KernelError> {
        Ok(())
    }

    fn observe_sumcheck_output(
        &mut self,
        _output: &Stage5SumcheckOutput<F>,
    ) -> Result<(), Stage5KernelError> {
        Ok(())
    }

    fn prove_sumcheck<T>(
        &mut self,
        context: Stage5KernelContext<'_>,
        transcript: &mut T,
    ) -> Result<Stage5SumcheckOutput<F>, Stage5KernelError>
    where
        T: Transcript<Challenge = F>;

    fn verify_sumcheck<T>(
        &mut self,
        context: Stage5KernelContext<'_>,
        transcript: &mut T,
    ) -> Result<Stage5SumcheckOutput<F>, Stage5KernelError>
    where
        T: Transcript<Challenge = F>;
}

#[derive(Clone, Debug, Default)]
pub struct UnsupportedStage5KernelExecutor;

impl<F: Field> Stage5KernelExecutor<F> for UnsupportedStage5KernelExecutor {
    fn prove_sumcheck<T>(
        &mut self,
        context: Stage5KernelContext<'_>,
        _transcript: &mut T,
    ) -> Result<Stage5SumcheckOutput<F>, Stage5KernelError>
    where
        T: Transcript<Challenge = F>,
    {
        Err(Stage5KernelError::KernelNotImplemented {
            abi: context.kernel.abi,
        })
    }

    fn verify_sumcheck<T>(
        &mut self,
        context: Stage5KernelContext<'_>,
        _transcript: &mut T,
    ) -> Result<Stage5SumcheckOutput<F>, Stage5KernelError>
    where
        T: Transcript<Challenge = F>,
    {
        Err(Stage5KernelError::KernelNotImplemented {
            abi: context.kernel.abi,
        })
    }
}

#[derive(Clone)]
pub struct Stage5ProverKernelExecutor<'a, F: Field> {
    pub inputs: Stage5ProverInputs<'a, F>,
    challenge_vectors: Vec<Stage5ChallengeVector<F>>,
    completed_sumchecks: Vec<Stage5SumcheckOutput<F>>,
}

impl<'a, F: Field> Stage5ProverKernelExecutor<'a, F> {
    pub fn new(inputs: Stage5ProverInputs<'a, F>) -> Self {
        Self {
            inputs,
            challenge_vectors: Vec::new(),
            completed_sumchecks: Vec::new(),
        }
    }

    fn value_store(
        &self,
        program: &'static Stage5CpuProgramPlan,
    ) -> Result<Stage5ValueStore<F>, Stage5KernelError> {
        value_store_from_observations(
            program,
            self.inputs.opening_inputs,
            &self.challenge_vectors,
            &self.completed_sumchecks,
        )
    }
}

impl<F: Field> Stage5KernelExecutor<F> for Stage5ProverKernelExecutor<'_, F> {
    fn observe_challenge_vector(
        &mut self,
        plan: &'static Stage5TranscriptSqueezePlan,
        values: &[F],
    ) -> Result<(), Stage5KernelError> {
        self.challenge_vectors.push(Stage5ChallengeVector {
            symbol: plan.symbol,
            values: values.to_vec(),
        });
        Ok(())
    }

    fn observe_sumcheck_output(
        &mut self,
        output: &Stage5SumcheckOutput<F>,
    ) -> Result<(), Stage5KernelError> {
        self.completed_sumchecks.push(output.clone());
        Ok(())
    }

    fn prove_sumcheck<T>(
        &mut self,
        context: Stage5KernelContext<'_>,
        transcript: &mut T,
    ) -> Result<Stage5SumcheckOutput<F>, Stage5KernelError>
    where
        T: Transcript<Challenge = F>,
    {
        prove_stage5_kernel(
            context,
            &self.inputs,
            self.value_store(context.program)?,
            transcript,
        )
    }

    fn verify_sumcheck<T>(
        &mut self,
        context: Stage5KernelContext<'_>,
        _transcript: &mut T,
    ) -> Result<Stage5SumcheckOutput<F>, Stage5KernelError>
    where
        T: Transcript<Challenge = F>,
    {
        Err(Stage5KernelError::WrongExecutorMode {
            driver: context.driver.symbol,
            expected: Stage5ExecutionMode::Prover,
            actual: Stage5ExecutionMode::Verifier,
        })
    }
}

#[derive(Clone)]
pub struct Stage5ProofCarryingKernelExecutor<'a, F: Field> {
    pub proof: &'a Stage5Proof<F>,
    pub opening_inputs: &'a [Stage5OpeningInputValue<F>],
    pub cursor: usize,
    challenge_vectors: Vec<Stage5ChallengeVector<F>>,
    completed_sumchecks: Vec<Stage5SumcheckOutput<F>>,
}

impl<'a, F: Field> Stage5ProofCarryingKernelExecutor<'a, F> {
    pub fn new(
        proof: &'a Stage5Proof<F>,
        opening_inputs: &'a [Stage5OpeningInputValue<F>],
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
        program: &'static Stage5CpuProgramPlan,
    ) -> Result<Stage5ValueStore<F>, Stage5KernelError> {
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
    ) -> Result<&'a Stage5SumcheckOutput<F>, Stage5KernelError> {
        let proof = self
            .proof
            .sumchecks
            .get(self.cursor)
            .ok_or(Stage5KernelError::MissingProof { driver })?;
        self.cursor += 1;
        Ok(proof)
    }
}

impl<F: Field> Stage5KernelExecutor<F> for Stage5ProofCarryingKernelExecutor<'_, F> {
    fn observe_challenge_vector(
        &mut self,
        plan: &'static Stage5TranscriptSqueezePlan,
        values: &[F],
    ) -> Result<(), Stage5KernelError> {
        self.challenge_vectors.push(Stage5ChallengeVector {
            symbol: plan.symbol,
            values: values.to_vec(),
        });
        Ok(())
    }

    fn observe_sumcheck_output(
        &mut self,
        output: &Stage5SumcheckOutput<F>,
    ) -> Result<(), Stage5KernelError> {
        self.completed_sumchecks.push(output.clone());
        Ok(())
    }

    fn prove_sumcheck<T>(
        &mut self,
        context: Stage5KernelContext<'_>,
        transcript: &mut T,
    ) -> Result<Stage5SumcheckOutput<F>, Stage5KernelError>
    where
        T: Transcript<Challenge = F>,
    {
        let proof = self.next_proof(context.driver.symbol)?;
        verify_stage5_kernel(
            context,
            self.value_store(context.program)?,
            proof,
            transcript,
        )
    }

    fn verify_sumcheck<T>(
        &mut self,
        context: Stage5KernelContext<'_>,
        transcript: &mut T,
    ) -> Result<Stage5SumcheckOutput<F>, Stage5KernelError>
    where
        T: Transcript<Challenge = F>,
    {
        let proof = self.next_proof(context.driver.symbol)?;
        verify_stage5_kernel(
            context,
            self.value_store(context.program)?,
            proof,
            transcript,
        )
    }
}

#[derive(Clone, Debug, Default)]
pub struct Stage5ValueStore<F: Field> {
    scalars: Vec<(&'static str, F)>,
    points: Vec<(&'static str, Vec<F>)>,
}

impl<F: Field> Stage5ValueStore<F> {
    pub fn with_opening_inputs(inputs: &[Stage5OpeningInputValue<F>]) -> Self {
        let mut store = Self::default();
        for input in inputs {
            store.insert_scalar(input.symbol, input.eval);
            store.insert_point(input.symbol, input.point.clone());
        }
        store
    }

    pub fn seed_constants(&mut self, program: &'static Stage5CpuProgramPlan) {
        for constant in program.field_constants {
            self.insert_scalar(constant.symbol, F::from_u64(constant.value as u64));
        }
    }

    pub fn observe_challenge_vector(
        &mut self,
        program: &'static Stage5CpuProgramPlan,
        plan: &'static Stage5TranscriptSqueezePlan,
        values: &[F],
    ) -> Result<(), Stage5KernelError> {
        self.insert_point(plan.symbol, values.to_vec());
        if matches!(plan.kind, "challenge_scalar" | "scalar") {
            require_operand_count(plan.symbol, 1, values.len())?;
            self.insert_scalar(plan.symbol, values[0]);
        }
        let _ = self.evaluate_available_field_exprs(program)?;
        Ok(())
    }

    pub fn observe_sumcheck_output(
        &mut self,
        program: &'static Stage5CpuProgramPlan,
        output: &Stage5SumcheckOutput<F>,
    ) -> Result<(), Stage5KernelError> {
        self.observe_sumcheck_values(program, output.driver, &output.point, &output.evals)
    }

    pub fn observe_sumcheck_values(
        &mut self,
        program: &'static Stage5CpuProgramPlan,
        driver: &'static str,
        point: &[F],
        evals: &[Stage5NamedEval<F>],
    ) -> Result<(), Stage5KernelError> {
        self.insert_point(driver, point.to_vec());
        for instance in program
            .instance_results
            .iter()
            .filter(|instance| instance.source == driver)
        {
            let end = instance.round_offset + instance.point_arity;
            let mut instance_point = point
                .get(instance.round_offset..end)
                .ok_or(Stage5KernelError::InvalidInputLength {
                    input: instance.symbol,
                    expected: end,
                    actual: point.len(),
                })?
                .to_vec();
            match instance.point_order {
                "as_is" => {}
                "instruction_read_raf" => {
                    instance_point = normalize_instruction_read_raf_point(&instance_point)?;
                }
                "reverse" => instance_point.reverse(),
                _ => {
                    return Err(Stage5KernelError::InvalidProof {
                        driver,
                        reason: "unsupported point order",
                    });
                }
            }
            self.insert_point(instance.symbol, instance_point);
        }
        for eval in program.evals.iter().filter(|eval| eval.source == driver) {
            let value = evals
                .iter()
                .find(|value| value.name == eval.name)
                .or_else(|| evals.get(eval.index))
                .ok_or(Stage5KernelError::MissingValue {
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

    pub fn claim_value(
        &mut self,
        program: &'static Stage5CpuProgramPlan,
        claim: &Stage5SumcheckClaimPlan,
    ) -> Result<F, Stage5KernelError> {
        let _ = self.evaluate_available_field_exprs(program)?;
        self.scalar(claim.claim_value)
    }

    pub fn batch_claim_values(
        &mut self,
        program: &'static Stage5CpuProgramPlan,
        batch: &Stage5SumcheckBatchPlan,
    ) -> Result<Vec<F>, Stage5KernelError> {
        batch
            .claim_operands
            .iter()
            .map(|symbol| {
                let claim = program
                    .claim(symbol)
                    .ok_or(Stage5KernelError::MissingClaim {
                        batch: batch.symbol,
                        claim: symbol,
                    })?;
                self.claim_value(program, claim)
            })
            .collect()
    }

    pub fn evaluate_available_points(
        &mut self,
        program: &'static Stage5CpuProgramPlan,
    ) -> Result<usize, Stage5KernelError> {
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
                    .ok_or(Stage5KernelError::InvalidInputLength {
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

    pub fn evaluate_available_field_exprs(
        &mut self,
        program: &'static Stage5CpuProgramPlan,
    ) -> Result<usize, Stage5KernelError> {
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
                self.insert_scalar(expr.symbol, evaluate_stage5_field_expr(expr, &operands)?);
                progress += 1;
            }
            inserted += progress;
            if progress == 0 {
                return Ok(inserted);
            }
        }
    }

    pub fn verify_opening_equalities(
        &self,
        program: &'static Stage5CpuProgramPlan,
    ) -> Result<(), Stage5KernelError> {
        for equality in program.opening_equalities {
            match equality.mode {
                "point_and_eval" => {
                    if self.point(equality.lhs)? != self.point(equality.rhs)?
                        || self.scalar(equality.lhs)? != self.scalar(equality.rhs)?
                    {
                        return Err(Stage5KernelError::InvalidProof {
                            driver: equality.symbol,
                            reason: "opening claim equality failed",
                        });
                    }
                }
                _ => {
                    return Err(Stage5KernelError::InvalidProof {
                        driver: equality.symbol,
                        reason: "unsupported opening equality mode",
                    });
                }
            }
        }
        Ok(())
    }

    pub fn insert_scalar(&mut self, symbol: &'static str, value: F) {
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

    pub fn insert_point(&mut self, symbol: &'static str, point: Vec<F>) {
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

    pub fn scalar(&self, symbol: &'static str) -> Result<F, Stage5KernelError> {
        self.try_scalar(symbol)
            .ok_or(Stage5KernelError::MissingValue { symbol })
    }

    pub fn try_scalar(&self, symbol: &str) -> Option<F> {
        self.scalars
            .iter()
            .find(|(existing, _)| *existing == symbol)
            .map(|(_, value)| *value)
    }

    pub fn point(&self, symbol: &'static str) -> Result<&[F], Stage5KernelError> {
        self.try_point(symbol)
            .ok_or(Stage5KernelError::MissingValue { symbol })
    }

    pub fn try_point(&self, symbol: &str) -> Option<&[F]> {
        self.points
            .iter()
            .find(|(existing, _)| *existing == symbol)
            .map(|(_, point)| point.as_slice())
    }

    fn try_expr_operands(&self, expr: &Stage5FieldExprPlan) -> Option<Vec<F>> {
        expr.operands
            .iter()
            .map(|operand| self.try_scalar(operand))
            .collect()
    }

    fn try_concat_point(&self, concat: &Stage5PointConcatPlan) -> Option<Vec<F>> {
        let mut point = Vec::with_capacity(concat.arity);
        for input in concat.inputs {
            point.extend_from_slice(self.try_point(input)?);
        }
        Some(point)
    }
}

fn value_store_from_observations<F: Field>(
    program: &'static Stage5CpuProgramPlan,
    opening_inputs: &[Stage5OpeningInputValue<F>],
    challenge_vectors: &[Stage5ChallengeVector<F>],
    completed_sumchecks: &[Stage5SumcheckOutput<F>],
) -> Result<Stage5ValueStore<F>, Stage5KernelError> {
    let mut store = Stage5ValueStore::with_opening_inputs(opening_inputs);
    store.seed_constants(program);
    for challenge in challenge_vectors {
        let plan = program
            .transcript_squeezes
            .iter()
            .find(|plan| plan.symbol == challenge.symbol)
            .ok_or(Stage5KernelError::MissingValue {
                symbol: challenge.symbol,
            })?;
        store.observe_challenge_vector(program, plan, &challenge.values)?;
    }
    for output in completed_sumchecks {
        store.observe_sumcheck_output(program, output)?;
    }
    let _ = store.evaluate_available_points(program)?;
    let _ = store.evaluate_available_field_exprs(program)?;
    store.verify_opening_equalities(program)?;
    Ok(store)
}

pub fn evaluate_stage5_field_expr<F: Field>(
    expr: &Stage5FieldExprPlan,
    operands: &[F],
) -> Result<F, Stage5KernelError> {
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
                    Stage5KernelError::UnsupportedFieldExpr {
                        symbol: expr.symbol,
                        formula,
                    }
                })?;
                return Ok(pow_field(operands[0], exponent));
            }
            Err(Stage5KernelError::UnsupportedFieldExpr {
                symbol: expr.symbol,
                formula,
            })
        }
    }
}

fn prove_stage5_kernel<F, T>(
    context: Stage5KernelContext<'_>,
    inputs: &Stage5ProverInputs<'_, F>,
    store: Stage5ValueStore<F>,
    transcript: &mut T,
) -> Result<Stage5SumcheckOutput<F>, Stage5KernelError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    match context.abi_kind()? {
        Stage5KernelAbi::Batched => prove_batched_stage5(context, inputs, store, transcript),
        abi => Err(Stage5KernelError::KernelNotImplemented { abi: abi.name() }),
    }
}

fn verify_stage5_kernel<F, T>(
    context: Stage5KernelContext<'_>,
    store: Stage5ValueStore<F>,
    proof: &Stage5SumcheckOutput<F>,
    transcript: &mut T,
) -> Result<Stage5SumcheckOutput<F>, Stage5KernelError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    match context.abi_kind()? {
        Stage5KernelAbi::Batched => verify_batched_stage5(context, store, proof, transcript),
        abi => Err(Stage5KernelError::KernelNotImplemented { abi: abi.name() }),
    }
}

#[tracing::instrument(skip_all, name = "Stage5::prove_batched")]
fn prove_batched_stage5<F, T>(
    context: Stage5KernelContext<'_>,
    inputs: &Stage5ProverInputs<'_, F>,
    mut store: Stage5ValueStore<F>,
    transcript: &mut T,
) -> Result<Stage5SumcheckOutput<F>, Stage5KernelError>
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
        .ok_or(Stage5KernelError::InvalidProof {
            driver: context.driver.symbol,
            reason: "field element 2 is not invertible",
        })?;
    let mut instances = Vec::with_capacity(claims.len());
    for (index, claim) in claims.iter().enumerate() {
        let offset = instance_round_offset(context.program, context.driver.symbol, claim.symbol)?;
        if offset + claim.num_rounds > max_rounds {
            return Err(Stage5KernelError::InvalidInputLength {
                input: claim.symbol,
                expected: max_rounds,
                actual: offset + claim.num_rounds,
            });
        }
        let active_scale = F::one().mul_pow_2(max_rounds - offset - claim.num_rounds);
        instances.push(Stage5BatchedInstance {
            claim,
            relation: claim_relation(context.program, claim)?,
            offset,
            previous_claim: input_claims[index].mul_pow_2(max_rounds - claim.num_rounds),
            state: Stage5ProverInstanceState::new(
                context.program,
                claim,
                inputs,
                &store,
                active_scale,
            )?,
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
        if batched_poly.evaluate(F::zero()) + batched_poly.evaluate(F::one()) != batched_claim {
            return Err(Stage5KernelError::InvalidProof {
                driver: context.driver.symbol,
                reason: "batched round claim mismatch",
            });
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
    for instance in &instances {
        evals.extend(instance.state.final_evals(instance.relation)?);
    }
    let expected =
        expected_batched_output_claim(context, &store, &evals, &point, &batching_coeffs)?;
    if batched_claim != expected {
        return Err(Stage5KernelError::InvalidProof {
            driver: context.driver.symbol,
            reason: "batched output claim mismatch",
        });
    }
    store.observe_sumcheck_values(context.program, context.driver.symbol, &point, &evals)?;
    let opening_claims = append_opening_claims(context.program, &mut store, transcript, &evals)?;
    Ok(Stage5SumcheckOutput {
        driver: context.driver.symbol,
        point,
        evals,
        opening_claims,
        proof: jolt_sumcheck::SumcheckProof { round_polynomials },
    })
}

fn verify_batched_stage5<F, T>(
    context: Stage5KernelContext<'_>,
    mut store: Stage5ValueStore<F>,
    proof: &Stage5SumcheckOutput<F>,
    transcript: &mut T,
) -> Result<Stage5SumcheckOutput<F>, Stage5KernelError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    if proof.driver != context.driver.symbol {
        return Err(Stage5KernelError::InvalidProof {
            driver: context.driver.symbol,
            reason: "driver symbol mismatch",
        });
    }
    if proof.proof.round_polynomials.len() != context.driver.num_rounds {
        return Err(Stage5KernelError::InvalidProof {
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
            return Err(Stage5KernelError::InvalidProof {
                driver: context.driver.symbol,
                reason: "batched polynomial exceeds degree bound",
            });
        }
        if poly.evaluate(F::zero()) + poly.evaluate(F::one()) != running_claim {
            return Err(Stage5KernelError::InvalidProof {
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
        return Err(Stage5KernelError::InvalidProof {
            driver: context.driver.symbol,
            reason: "batched point mismatch",
        });
    }
    let expected =
        expected_batched_output_claim(context, &store, &proof.evals, &point, &batching_coeffs)?;
    if running_claim != expected {
        return Err(Stage5KernelError::InvalidProof {
            driver: context.driver.symbol,
            reason: "batched output claim mismatch",
        });
    }

    let output = Stage5SumcheckOutput {
        driver: context.driver.symbol,
        point,
        evals: proof.evals.clone(),
        opening_claims: Vec::new(),
        proof: proof.proof.clone(),
    };
    store.observe_sumcheck_output(context.program, &output)?;
    let opening_claims =
        append_opening_claims(context.program, &mut store, transcript, &output.evals)?;
    let output = Stage5SumcheckOutput {
        opening_claims,
        ..output
    };
    Ok(output)
}

struct Stage5BatchedInstance<'a, F: Field> {
    claim: &'a Stage5SumcheckClaimPlan,
    relation: Stage5Relation,
    offset: usize,
    previous_claim: F,
    state: Stage5ProverInstanceState<F>,
}

impl<F: Field> Stage5BatchedInstance<'_, F> {
    fn is_active(&self, round: usize) -> bool {
        round >= self.offset && round < self.offset + self.claim.num_rounds
    }
}

enum Stage5ProverInstanceState<F: Field> {
    Dense(DenseStage5State<F>),
    InstructionReadRaf(InstructionReadRafStage5State<F>),
}

impl<F: Field> Stage5ProverInstanceState<F> {
    fn new(
        program: &'static Stage5CpuProgramPlan,
        claim: &Stage5SumcheckClaimPlan,
        inputs: &Stage5ProverInputs<'_, F>,
        store: &Stage5ValueStore<F>,
        active_scale: F,
    ) -> Result<Self, Stage5KernelError> {
        match claim_relation(program, claim)? {
            Stage5Relation::InstructionReadRaf => {
                instruction_read_raf_state(program, claim, inputs, store, active_scale)
                    .map(Self::InstructionReadRaf)
            }
            Stage5Relation::RegistersValEvaluation => {
                registers_val_evaluation_state(claim, inputs, store, active_scale).map(Self::Dense)
            }
            Stage5Relation::RamRaClaimReduction => {
                ram_ra_claim_reduction_state(claim, inputs, store, active_scale).map(Self::Dense)
            }
            relation @ Stage5Relation::Batched => Err(Stage5KernelError::KernelNotImplemented {
                abi: relation.symbol(),
            }),
        }
    }

    fn round_poly(
        &mut self,
        previous_claim: F,
        relation: Stage5Relation,
    ) -> Result<UnivariatePoly<F>, Stage5KernelError> {
        match self {
            Self::Dense(state) => state.round_poly(previous_claim, relation),
            Self::InstructionReadRaf(state) => state.round_poly(previous_claim, relation),
        }
    }

    fn ingest_challenge(&mut self, challenge: F) {
        match self {
            Self::Dense(state) => state.bind(challenge),
            Self::InstructionReadRaf(state) => state.bind(challenge),
        }
    }

    fn final_evals(
        &self,
        relation: Stage5Relation,
    ) -> Result<Vec<Stage5NamedEval<F>>, Stage5KernelError> {
        match self {
            Self::Dense(state) => state.final_evals(relation),
            Self::InstructionReadRaf(state) => state.final_evals(relation),
        }
    }
}

struct DenseStage5State<F: Field> {
    factors: Vec<Vec<F>>,
    factor_scratch: Vec<Vec<F>>,
    terms: Vec<DenseTerm<F>>,
    outputs: Vec<FactorOutput>,
    active_scale: F,
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

#[derive(Clone, Copy)]
enum InstructionReadRafOutputKind {
    LookupTableFlag(usize),
    InstructionRa(usize),
    InstructionRafFlag,
}

#[derive(Clone, Copy)]
struct InstructionReadRafOutputPlan {
    index: usize,
    name: &'static str,
    oracle: &'static str,
    kind: InstructionReadRafOutputKind,
}

struct InstructionReadRafStage5State<F: Field> {
    trace_len: usize,
    lookup_indices: Vec<u128>,
    lookup_table_indices: Vec<Option<usize>>,
    is_interleaved_operands: Vec<bool>,
    lookup_groups: Vec<InstructionReadRafLookupGroup<F>>,
    lookup_group_indices_by_cycle: Vec<usize>,
    lookup_groups_by_table: Vec<Vec<usize>>,
    ra_virtual_log_k_chunk: usize,
    u_evals: Vec<F>,
    gamma: F,
    gamma2: F,
    active_scale: F,
    round: usize,
    address_challenges: Vec<F>,
    cycle_challenges: Vec<F>,
    address_phase: Option<InstructionReadRafAddressPhase<F>>,
    left_operand_checkpoint: F,
    right_operand_checkpoint: F,
    identity_checkpoint: F,
    read_prefix_checkpoints: Vec<PrefixEval<F>>,
    cycle_state: Option<InstructionReadRafCycleState<F>>,
    outputs: Vec<InstructionReadRafOutputPlan>,
}

struct InstructionReadRafLookupGroup<F: Field> {
    lookup_index: u128,
    lookup_table_index: Option<usize>,
    is_interleaved_operands: bool,
    u_eval_sum: F,
    phase_u_eval_sum: F,
}

struct InstructionReadRafAddressPhase<F: Field> {
    phase: usize,
    left_operand_prefix: Vec<F>,
    right_operand_prefix: Vec<F>,
    identity_prefix: Vec<F>,
    raf_shift_half_q: Vec<F>,
    raf_left_q: Vec<F>,
    raf_right_q: Vec<F>,
    raf_shift_full_q: Vec<F>,
    raf_identity_q: Vec<F>,
    read_prefix_polys: Vec<Vec<F>>,
    read_suffix_polys: Vec<InstructionReadRafReadTablePhase<F>>,
}

struct InstructionReadRafReadTablePhase<F: Field> {
    table: LookupTableKind<64>,
    suffix_polys: Vec<Vec<F>>,
}

struct InstructionReadRafCycleState<F: Field> {
    factors: Vec<Vec<F>>,
    factor_scratch: Vec<Vec<F>>,
}

impl<F: Field> DenseStage5State<F> {
    fn new(
        factors: Vec<Vec<F>>,
        terms: Vec<DenseTerm<F>>,
        outputs: Vec<FactorOutput>,
        active_scale: F,
    ) -> Self {
        let factor_scratch = (0..factors.len()).map(|_| Vec::new()).collect();
        Self {
            factors,
            factor_scratch,
            terms,
            outputs,
            active_scale,
        }
    }

    fn round_poly(
        &self,
        previous_claim: F,
        relation: Stage5Relation,
    ) -> Result<UnivariatePoly<F>, Stage5KernelError> {
        let first_len = self.factors.first().map_or(0, Vec::len);
        if first_len == 0 || !first_len.is_power_of_two() {
            return Err(Stage5KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "stage5 dense factor has invalid length",
            });
        }
        if self.factors.iter().any(|factor| factor.len() != first_len) {
            return Err(Stage5KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "stage5 dense factors have inconsistent lengths",
            });
        }
        let poly =
            round_poly_from_dense_terms(&self.factors, &self.terms, self.active_scale, relation)?;
        if poly.evaluate(F::zero()) + poly.evaluate(F::one()) != previous_claim {
            return Err(Stage5KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "stage5 relation input claim mismatch",
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

    fn factor_eval(&self, index: usize, relation: Stage5Relation) -> Result<F, Stage5KernelError> {
        self.factors
            .get(index)
            .and_then(|values| values.first())
            .copied()
            .ok_or(Stage5KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "empty stage5 factor",
            })
    }

    fn final_evals(
        &self,
        relation: Stage5Relation,
    ) -> Result<Vec<Stage5NamedEval<F>>, Stage5KernelError> {
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

impl<F: Field> InstructionReadRafStage5State<F> {
    const LOG_K: usize = 128;
    const ADDRESS_CHUNK_BITS: usize = 8;

    fn round_poly(
        &mut self,
        previous_claim: F,
        relation: Stage5Relation,
    ) -> Result<UnivariatePoly<F>, Stage5KernelError> {
        if self.round < Self::LOG_K {
            self.ensure_address_phase();
            let Some(address_phase) = self.address_phase.as_ref() else {
                std::process::abort();
            };
            let (read, raf_components) = if address_phase.left_operand_prefix.len() <= 32 {
                (
                    address_phase.read_table_round_evals(),
                    address_phase.raf_round_component_evals(),
                )
            } else {
                rayon::join(
                    || address_phase.read_table_round_evals(),
                    || address_phase.raf_round_component_evals(),
                )
            };
            let eval_at_0 = (read[0]
                + self.gamma * raf_components[0][0]
                + self.gamma2 * (raf_components[1][0] + raf_components[2][0]))
                * self.active_scale;
            let eval_at_2 = (read[1]
                + self.gamma * raf_components[0][1]
                + self.gamma2 * (raf_components[1][1] + raf_components[2][1]))
                * self.active_scale;
            let result = Ok(UnivariatePoly::from_evals_and_hint(
                previous_claim,
                &[eval_at_0, eval_at_2],
            ));
            return result;
        }

        if self.cycle_state.is_none() {
            self.cycle_state = Some(self.materialize_cycle_state()?);
        }
        let Some(cycle_state) = self.cycle_state.as_ref() else {
            std::process::abort();
        };
        cycle_state.round_poly(previous_claim, self.active_scale, relation)
    }

    fn bind(&mut self, challenge: F) {
        if self.round < Self::LOG_K {
            self.ensure_address_phase();
            self.address_challenges.push(challenge);
            if let Some(phase) = &mut self.address_phase {
                phase.bind(challenge);
            }
            if (self.round + 1).is_multiple_of(Self::ADDRESS_CHUNK_BITS) {
                self.finish_address_phase();
            }
        } else {
            self.cycle_challenges.push(challenge);
            if let Some(cycle_state) = &mut self.cycle_state {
                cycle_state.bind(challenge);
            }
        }
        self.round += 1;
    }

    fn final_evals(
        &self,
        relation: Stage5Relation,
    ) -> Result<Vec<Stage5NamedEval<F>>, Stage5KernelError> {
        require_operand_count(
            "stage5.instruction_read_raf.address_challenges",
            Self::LOG_K,
            self.address_challenges.len(),
        )?;
        require_operand_count(
            "stage5.instruction_read_raf.cycle_challenges",
            log2_exact(self.trace_len, "stage5.instruction_read_raf.trace_len")?,
            self.cycle_challenges.len(),
        )?;

        let normalized_cycle_point = reverse_slice(&self.cycle_challenges);
        let evaluations = self.instruction_read_raf_output_evals_from_groups(
            &self.address_challenges,
            &normalized_cycle_point,
        )?;
        self.outputs
            .iter()
            .map(|output| {
                let value = match output.kind {
                    InstructionReadRafOutputKind::LookupTableFlag(index) => {
                        evaluations.lookup_table_flags.get(index).copied().ok_or(
                            Stage5KernelError::InvalidInputLength {
                                input: "stage5.instruction_read_raf.lookup_table_flags",
                                expected: evaluations.lookup_table_flags.len(),
                                actual: index + 1,
                            },
                        )?
                    }
                    InstructionReadRafOutputKind::InstructionRa(index) => {
                        evaluations.instruction_ra.get(index).copied().ok_or(
                            Stage5KernelError::InvalidInputLength {
                                input: "stage5.instruction_read_raf.instruction_ra",
                                expected: evaluations.instruction_ra.len(),
                                actual: index + 1,
                            },
                        )?
                    }
                    InstructionReadRafOutputKind::InstructionRafFlag => {
                        evaluations.instruction_raf_flag
                    }
                };
                Ok(named_eval(output.name, output.oracle, value))
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| match error {
                Stage5KernelError::InvalidInputLength { .. } => error,
                _ => Stage5KernelError::InvalidProof {
                    driver: relation.symbol(),
                    reason: "invalid instruction read raf output",
                },
            })
    }

    fn ensure_address_phase(&mut self) {
        debug_assert!(Self::LOG_K.is_multiple_of(Self::ADDRESS_CHUNK_BITS));
        let phase = self.round / Self::ADDRESS_CHUNK_BITS;
        if self
            .address_phase
            .as_ref()
            .is_some_and(|address_phase| address_phase.phase == phase)
        {
            return;
        }
        self.address_phase = Some(self.build_address_phase(phase));
    }

    fn build_address_phase(&self, phase: usize) -> InstructionReadRafAddressPhase<F> {
        let chunk_bits = Self::ADDRESS_CHUNK_BITS;
        let poly_len = 1usize << chunk_bits;
        let suffix_len = Self::LOG_K - (phase + 1) * chunk_bits;
        let left_operand_prefix =
            operand_prefix_poly(self.left_operand_checkpoint, chunk_bits, true);
        let right_operand_prefix =
            operand_prefix_poly(self.right_operand_checkpoint, chunk_bits, false);
        let identity_prefix = identity_prefix_poly(self.identity_checkpoint, chunk_bits);
        let read_prefix_polys = ALL_PREFIXES
            .par_iter()
            .map(|prefix| {
                (0..poly_len)
                    .map(|bits| {
                        prefix
                            .evaluate(
                                &self.read_prefix_checkpoints,
                                LookupBits::new(bits as u128, chunk_bits),
                                suffix_len,
                            )
                            .into_inner()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let shift_half_value = 1u128 << (suffix_len / 2);
        let shift_full_value = 1u128 << suffix_len;
        let shift_half = F::from_u128(shift_half_value);
        let shift_full = F::from_u128(shift_full_value);
        let suffix_mask = if suffix_len == 128 {
            u128::MAX
        } else {
            (1u128 << suffix_len) - 1
        };

        let q_total_len = 5 * poly_len;
        let q_chunk_size = self
            .lookup_groups
            .len()
            .div_ceil(rayon::current_num_threads())
            .max(1);
        let q_rows = self
            .lookup_groups
            .par_chunks(q_chunk_size)
            .fold(
                || vec![F::zero(); q_total_len],
                |mut acc, groups| {
                    let shift_half_offset = 0;
                    let left_offset = poly_len;
                    let right_offset = 2 * poly_len;
                    let shift_full_offset = 3 * poly_len;
                    let identity_offset = 4 * poly_len;

                    for group in groups {
                        let index = ((group.lookup_index >> suffix_len) as usize) & (poly_len - 1);
                        let suffix_bits = group.lookup_index & suffix_mask;
                        let weight = group.phase_u_eval_sum;

                        if group.is_interleaved_operands {
                            acc[shift_half_offset + index] += weight;
                            let (left_suffix, right_suffix) = uninterleave_bits(suffix_bits);
                            if left_suffix != 0 {
                                acc[left_offset + index] += weight.mul_u64(left_suffix);
                            }
                            if right_suffix != 0 {
                                acc[right_offset + index] += weight.mul_u64(right_suffix);
                            }
                        } else {
                            acc[shift_full_offset + index] += weight;
                            if suffix_bits != 0 {
                                acc[identity_offset + index] += weight.mul_u128(suffix_bits);
                            }
                        }
                    }
                    acc
                },
            )
            .reduce(
                || vec![F::zero(); q_total_len],
                |mut left, right| {
                    for (left_value, right_value) in left.iter_mut().zip(right) {
                        *left_value += right_value;
                    }
                    left
                },
            );
        let mut raf_shift_half_q = q_rows[..poly_len].to_vec();
        let raf_left_q = q_rows[poly_len..2 * poly_len].to_vec();
        let raf_right_q = q_rows[2 * poly_len..3 * poly_len].to_vec();
        let mut raf_shift_full_q = q_rows[3 * poly_len..4 * poly_len].to_vec();
        let raf_identity_q = q_rows[4 * poly_len..5 * poly_len].to_vec();
        if shift_half_value != 1 {
            for value in &mut raf_shift_half_q {
                *value *= shift_half;
            }
        }
        if shift_full_value != 1 {
            for value in &mut raf_shift_full_q {
                *value *= shift_full;
            }
        }

        let tables = LookupTableKind::<64>::all();
        let read_suffix_polys = tables
            .par_iter()
            .enumerate()
            .filter_map(|(table_index, table)| {
                if self.lookup_groups_by_table[table_index].is_empty() {
                    return None;
                }
                let suffixes = table.suffixes();
                let mut accumulators =
                    vec![vec![F::Accumulator::default(); poly_len]; suffixes.len()];
                let mut one_suffix = None;
                let mut boolean_suffixes = Vec::new();
                let mut valued_suffixes = Vec::new();
                for (suffix_index, suffix) in suffixes.iter().enumerate() {
                    if matches!(suffix, Suffixes::One) {
                        one_suffix = Some(suffix_index);
                    } else if suffix.is_01_valued() {
                        boolean_suffixes.push((suffix_index, suffix));
                    } else {
                        valued_suffixes.push((suffix_index, suffix));
                    }
                }
                for &group_index in &self.lookup_groups_by_table[table_index] {
                    let group = &self.lookup_groups[group_index];
                    let index = ((group.lookup_index >> suffix_len) as usize) & (poly_len - 1);
                    let suffix_bits = LookupBits::new(group.lookup_index & suffix_mask, suffix_len);
                    let weight = group.phase_u_eval_sum;
                    if let Some(suffix_index) = one_suffix {
                        accumulators[suffix_index][index].add(weight);
                    }
                    for &(suffix_index, suffix) in &boolean_suffixes {
                        let suffix_value = suffix.suffix_mle(suffix_bits);
                        debug_assert!(suffix_value == 0 || suffix_value == 1);
                        if suffix_value == 1 {
                            accumulators[suffix_index][index].add(weight);
                        }
                    }
                    for &(suffix_index, suffix) in &valued_suffixes {
                        let suffix_value = suffix.suffix_mle(suffix_bits);
                        accumulators[suffix_index][index].fmadd_u64(weight, suffix_value);
                    }
                }
                let polys = accumulators
                    .into_iter()
                    .map(|poly| {
                        poly.into_iter()
                            .map(AdditiveAccumulator::reduce)
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                Some(InstructionReadRafReadTablePhase {
                    table: *table,
                    suffix_polys: polys,
                })
            })
            .collect::<Vec<_>>();

        InstructionReadRafAddressPhase {
            phase,
            left_operand_prefix,
            right_operand_prefix,
            identity_prefix,
            raf_shift_half_q,
            raf_left_q,
            raf_right_q,
            raf_shift_full_q,
            raf_identity_q,
            read_prefix_polys,
            read_suffix_polys,
        }
    }

    fn finish_address_phase(&mut self) {
        let Some(phase) = self.address_phase.take() else {
            return;
        };
        self.left_operand_checkpoint = phase.left_operand_prefix[0];
        self.right_operand_checkpoint = phase.right_operand_prefix[0];
        self.identity_checkpoint = phase.identity_prefix[0];
        self.read_prefix_checkpoints = phase
            .read_prefix_polys
            .iter()
            .map(|poly| PrefixEval::from(poly[0]))
            .collect();

        let chunk_bits = Self::ADDRESS_CHUNK_BITS;
        let start = phase.phase * chunk_bits;
        let end = start + chunk_bits;
        let point = &self.address_challenges[start..end];
        let shift = Self::LOG_K - end;
        let mask = (1u128 << chunk_bits) - 1;
        let eq_table = (0..(1usize << chunk_bits))
            .map(|bits| eq_eval_at_bits(point, bits as u128, chunk_bits))
            .collect::<Vec<_>>();
        self.lookup_groups.par_iter_mut().for_each(|group| {
            let chunk_value = (group.lookup_index >> shift) & mask;
            group.phase_u_eval_sum *= eq_table[chunk_value as usize];
        });
    }

    fn materialize_cycle_state(
        &self,
    ) -> Result<InstructionReadRafCycleState<F>, Stage5KernelError> {
        require_operand_count(
            "stage5.instruction_read_raf.address_challenges",
            Self::LOG_K,
            self.address_challenges.len(),
        )?;
        let tables = LookupTableKind::<64>::all();
        let ra_chunks = Self::LOG_K / self.ra_virtual_log_k_chunk;
        let mut factors = Vec::with_capacity(2 + ra_chunks);
        factors.push(self.u_evals.clone());
        let table_values_at_address = tables
            .par_iter()
            .enumerate()
            .map(|(table_index, table)| {
                if self.lookup_groups_by_table[table_index].is_empty() {
                    F::zero()
                } else {
                    table.evaluate_mle::<F, F>(&self.address_challenges)
                }
            })
            .collect::<Vec<_>>();
        let raf_interleaved = self.gamma * operand_polynomial_eval(&self.address_challenges, true)
            + self.gamma2 * operand_polynomial_eval(&self.address_challenges, false);
        let raf_identity = self.gamma2 * identity_polynomial_eval(&self.address_challenges);
        factors.push(
            (0..self.trace_len)
                .into_par_iter()
                .map(|cycle| {
                    let table_value = self.lookup_table_indices[cycle]
                        .map_or_else(F::zero, |table_index| table_values_at_address[table_index]);
                    let raf_value = if self.is_interleaved_operands[cycle] {
                        raf_interleaved
                    } else {
                        raf_identity
                    };
                    table_value + raf_value
                })
                .collect(),
        );

        let chunk_bits = self.ra_virtual_log_k_chunk;
        let chunk_mask = if chunk_bits == 128 {
            u128::MAX
        } else {
            (1u128 << chunk_bits) - 1
        };
        for chunk in 0..ra_chunks {
            let chunk_point =
                &self.address_challenges[chunk * chunk_bits..(chunk + 1) * chunk_bits];
            let eq_tables = eq_eval_bit_chunk_tables(chunk_point, 8);
            let shift = Self::LOG_K - (chunk + 1) * chunk_bits;
            factors.push(
                self.lookup_indices
                    .par_iter()
                    .map(|&lookup_index| {
                        let chunk_value = (lookup_index >> shift) & chunk_mask;
                        eq_eval_at_bits_from_chunk_tables(&eq_tables, chunk_value, chunk_bits, 8)
                    })
                    .collect(),
            );
        }
        InstructionReadRafCycleState::new(factors, Stage5Relation::InstructionReadRaf)
    }

    fn instruction_read_raf_output_evals_from_groups(
        &self,
        address_point: &[F],
        cycle_point: &[F],
    ) -> Result<Stage5InstructionReadRafEvaluations<F>, Stage5KernelError> {
        require_operand_count(
            "stage5.instruction_read_raf.address_point",
            Self::LOG_K,
            address_point.len(),
        )?;
        let trace_len_from_point = 1usize.checked_shl(cycle_point.len() as u32).ok_or(
            Stage5KernelError::InvalidInputLength {
                input: "stage5.instruction_read_raf.cycle_point",
                expected: usize::BITS as usize,
                actual: cycle_point.len(),
            },
        )?;
        require_operand_count(
            "stage5.instruction_read_raf.trace_len",
            trace_len_from_point,
            self.trace_len,
        )?;

        let tables = LookupTableKind::<64>::all();
        let table_count = tables.len();
        let cycle_eq = EqPolynomial::<F>::evals(cycle_point, None);
        require_operand_count(
            "stage5.instruction_read_raf.eq_cycle",
            self.trace_len,
            cycle_eq.len(),
        )?;

        let mut lookup_table_flags = vec![F::zero(); table_count];
        let mut instruction_raf_flag = F::zero();
        let mut group_weights = vec![F::zero(); self.lookup_groups.len()];
        for (&group_index, &weight) in self.lookup_group_indices_by_cycle.iter().zip(&cycle_eq) {
            group_weights[group_index] += weight;
        }

        for (group, &weight) in self.lookup_groups.iter().zip(&group_weights) {
            if let Some(table_index) = group.lookup_table_index {
                let Some(flag) = lookup_table_flags.get_mut(table_index) else {
                    return Err(Stage5KernelError::InvalidInputLength {
                        input: "stage5.instruction_read_raf.lookup_table_indices",
                        expected: table_count,
                        actual: table_index + 1,
                    });
                };
                *flag += weight;
            }
            if !group.is_interleaved_operands {
                instruction_raf_flag += weight;
            }
        }

        let ra_chunks = Self::LOG_K / self.ra_virtual_log_k_chunk;
        let cycle_state = self
            .cycle_state
            .as_ref()
            .ok_or(Stage5KernelError::InvalidProof {
                driver: Stage5Relation::InstructionReadRaf.symbol(),
                reason: "instruction read raf cycle state is not materialized",
            })?;
        let instruction_ra = (0..ra_chunks)
            .map(|chunk| {
                cycle_state
                    .factor_eval(2 + chunk)
                    .ok_or(Stage5KernelError::InvalidInputLength {
                        input: "stage5.instruction_read_raf.instruction_ra",
                        expected: cycle_state.factors.len(),
                        actual: 2 + chunk + 1,
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Stage5InstructionReadRafEvaluations {
            lookup_table_flags,
            instruction_ra,
            instruction_raf_flag,
        })
    }
}

impl<F: Field> InstructionReadRafAddressPhase<F> {
    fn read_table_round_evals(&self) -> [F; 2] {
        let len = self.read_prefix_polys.first().map_or(0, |poly| poly.len());
        debug_assert!(len > 1);
        debug_assert!(self.read_prefix_polys.iter().all(|poly| poly.len() == len));
        debug_assert!(self
            .read_suffix_polys
            .iter()
            .flat_map(|read_table| read_table.suffix_polys.iter())
            .all(|poly| poly.len() == len));
        let half = len / 2;
        let prefix_evals = (0..half)
            .map(|row| {
                (
                    self.read_prefix_evals(row, false),
                    self.read_prefix_evals(row, true),
                )
            })
            .collect::<Vec<_>>();
        if half <= 16 {
            self.read_suffix_polys
                .iter()
                .fold([F::zero(), F::zero()], |mut total, read_table| {
                    let eval = read_table_component_eval(read_table, half, &prefix_evals);
                    total[0] += eval[0];
                    total[1] += eval[1];
                    total
                })
        } else {
            self.read_suffix_polys
                .par_iter()
                .map(|read_table| read_table_component_eval(read_table, half, &prefix_evals))
                .reduce(
                    || [F::zero(), F::zero()],
                    |mut left, right| {
                        left[0] += right[0];
                        left[1] += right[1];
                        left
                    },
                )
        }
    }

    fn read_prefix_evals(&self, row: usize, at_2: bool) -> [PrefixEval<F>; NUM_PREFIXES] {
        let half = self.read_prefix_polys[0].len() / 2;
        let mut values = [PrefixEval::from(F::zero()); NUM_PREFIXES];
        for (value, poly) in values.iter_mut().zip(&self.read_prefix_polys) {
            let low = poly[row];
            let eval = if at_2 {
                let high = poly[row + half];
                high + high - low
            } else {
                low
            };
            *value = PrefixEval::from(eval);
        }
        values
    }

    fn raf_round_component_evals(&self) -> [[F; 2]; 3] {
        let (left_0, left_2) = prefix_suffix_round_evals(
            Some(&self.left_operand_prefix),
            &self.raf_shift_half_q,
            &self.raf_left_q,
        );
        let (right_0, right_2) = prefix_suffix_round_evals(
            Some(&self.right_operand_prefix),
            &self.raf_shift_half_q,
            &self.raf_right_q,
        );
        let (identity_0, identity_2) = prefix_suffix_round_evals(
            Some(&self.identity_prefix),
            &self.raf_shift_full_q,
            &self.raf_identity_q,
        );
        [
            [left_0, left_2],
            [right_0, right_2],
            [identity_0, identity_2],
        ]
    }

    fn bind(&mut self, challenge: F) {
        if self.left_operand_prefix.len() <= 32 {
            bind_high_to_low(&mut self.left_operand_prefix, challenge);
            bind_high_to_low(&mut self.right_operand_prefix, challenge);
            bind_high_to_low(&mut self.identity_prefix, challenge);
            bind_high_to_low(&mut self.raf_shift_half_q, challenge);
            bind_high_to_low(&mut self.raf_left_q, challenge);
            bind_high_to_low(&mut self.raf_right_q, challenge);
            bind_high_to_low(&mut self.raf_shift_full_q, challenge);
            bind_high_to_low(&mut self.raf_identity_q, challenge);
            for poly in &mut self.read_prefix_polys {
                bind_high_to_low(poly, challenge);
            }
            for read_table in &mut self.read_suffix_polys {
                for poly in &mut read_table.suffix_polys {
                    bind_high_to_low(poly, challenge);
                }
            }
            return;
        }

        let left_operand_prefix = &mut self.left_operand_prefix;
        let right_operand_prefix = &mut self.right_operand_prefix;
        let identity_prefix = &mut self.identity_prefix;
        let raf_shift_half_q = &mut self.raf_shift_half_q;
        let raf_left_q = &mut self.raf_left_q;
        let raf_right_q = &mut self.raf_right_q;
        let raf_shift_full_q = &mut self.raf_shift_full_q;
        let raf_identity_q = &mut self.raf_identity_q;
        let read_prefix_polys = &mut self.read_prefix_polys;
        let read_suffix_polys = &mut self.read_suffix_polys;
        rayon::scope(|scope| {
            scope.spawn(|_| {
                bind_high_to_low(left_operand_prefix, challenge);
                bind_high_to_low(right_operand_prefix, challenge);
                bind_high_to_low(identity_prefix, challenge);
            });
            scope.spawn(|_| {
                bind_high_to_low(raf_shift_half_q, challenge);
                bind_high_to_low(raf_left_q, challenge);
                bind_high_to_low(raf_right_q, challenge);
                bind_high_to_low(raf_shift_full_q, challenge);
                bind_high_to_low(raf_identity_q, challenge);
            });
            scope.spawn(|_| {
                read_prefix_polys
                    .par_iter_mut()
                    .for_each(|poly| bind_high_to_low(poly, challenge));
            });
            scope.spawn(|_| {
                read_suffix_polys.par_iter_mut().for_each(|read_table| {
                    for poly in &mut read_table.suffix_polys {
                        bind_high_to_low(poly, challenge);
                    }
                });
            });
        });
    }
}

fn read_table_component_eval<F: Field>(
    read_table: &InstructionReadRafReadTablePhase<F>,
    half: usize,
    prefix_evals: &[PrefixPairEvals<F>],
) -> [F; 2] {
    let mut eval_0 = F::zero();
    let mut eval_2_left = F::zero();
    let mut eval_2_right = F::zero();
    for row in 0..half {
        let (prefixes_0, prefixes_2) = &prefix_evals[row];
        let mut suffixes_left = [F::zero(); 4];
        let mut suffixes_right = [F::zero(); 4];
        for (suffix_index, poly) in read_table.suffix_polys.iter().enumerate() {
            suffixes_left[suffix_index] = poly[row];
            suffixes_right[suffix_index] = poly[row + half];
        }
        let suffix_count = read_table.suffix_polys.len();
        eval_0 += read_table
            .table
            .combine(prefixes_0, &suffixes_left[..suffix_count]);
        eval_2_left += read_table
            .table
            .combine(prefixes_2, &suffixes_left[..suffix_count]);
        eval_2_right += read_table
            .table
            .combine(prefixes_2, &suffixes_right[..suffix_count]);
    }
    [eval_0, eval_2_right + eval_2_right - eval_2_left]
}

#[inline]
fn prefix_suffix_round_evals<F: Field>(prefix: Option<&[F]>, q0: &[F], q1: &[F]) -> (F, F) {
    let len = q0.len();
    debug_assert_eq!(q1.len(), len);
    debug_assert!(len > 1);
    let half = len / 2;
    let mut eval_0 = F::zero();
    let mut eval_2_left = F::zero();
    let mut eval_2_right = F::zero();
    for row in 0..half {
        let (prefix_0, prefix_2) = prefix.map_or((F::one(), F::one()), |poly| {
            debug_assert_eq!(poly.len(), len);
            let low = poly[row];
            let high = poly[row + half];
            (low, high + high - low)
        });
        eval_0 += prefix_0 * q0[row] + q1[row];
        eval_2_left += prefix_2 * q0[row] + q1[row];
        eval_2_right += prefix_2 * q0[row + half] + q1[row + half];
    }
    (eval_0, eval_2_right + eval_2_right - eval_2_left)
}

fn operand_prefix_poly<F: Field>(checkpoint: F, chunk_bits: usize, left: bool) -> Vec<F> {
    debug_assert!(chunk_bits.is_multiple_of(2));
    let shift = 1u128 << (chunk_bits / 2);
    (0..(1usize << chunk_bits))
        .map(|bits| {
            let lookup_bits = LookupBits::new(bits as u128, chunk_bits);
            let (left_bits, right_bits) = lookup_bits.uninterleave();
            let operand_bits: u64 = if left {
                left_bits.into()
            } else {
                right_bits.into()
            };
            checkpoint.mul_u128(shift) + F::from_u64(operand_bits)
        })
        .collect()
}

fn identity_prefix_poly<F: Field>(checkpoint: F, chunk_bits: usize) -> Vec<F> {
    let shift = 1u128 << chunk_bits;
    (0..(1usize << chunk_bits))
        .map(|bits| checkpoint.mul_u128(shift) + F::from_u64(bits as u64))
        .collect()
}

impl<F: Field> InstructionReadRafCycleState<F> {
    fn new(factors: Vec<Vec<F>>, relation: Stage5Relation) -> Result<Self, Stage5KernelError> {
        let first_len = factors.first().map_or(0, Vec::len);
        if first_len == 0 || !first_len.is_power_of_two() {
            return Err(Stage5KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "instruction read raf cycle factor has invalid length",
            });
        }
        if factors.iter().any(|factor| factor.len() != first_len) {
            return Err(Stage5KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "instruction read raf cycle factors have inconsistent lengths",
            });
        }
        let factor_scratch = (0..factors.len()).map(|_| Vec::new()).collect();
        Ok(Self {
            factors,
            factor_scratch,
        })
    }

    fn round_poly(
        &self,
        previous_claim: F,
        active_scale: F,
        relation: Stage5Relation,
    ) -> Result<UnivariatePoly<F>, Stage5KernelError> {
        let coefficients = self.round_coefficients(active_scale);
        let poly = UnivariatePoly::new(coefficients);
        if poly.evaluate(F::zero()) + poly.evaluate(F::one()) != previous_claim {
            return Err(Stage5KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "instruction read raf cycle input claim mismatch",
            });
        }
        Ok(poly)
    }

    fn factor_eval(&self, index: usize) -> Option<F> {
        self.factors
            .get(index)
            .and_then(|factor| factor.first())
            .copied()
    }

    fn round_coefficients(&self, active_scale: F) -> Vec<F> {
        const MAX_DEGREE_PLUS_ONE: usize = 16;
        let degree = self.factors.len();
        debug_assert!(degree < MAX_DEGREE_PLUS_ONE);
        let half = self.factors[0].len() / 2;
        let mut sums = if half >= DENSE_BIND_PAR_THRESHOLD {
            (0..half)
                .into_par_iter()
                .fold(
                    || [F::zero(); MAX_DEGREE_PLUS_ONE],
                    |mut sums, row| {
                        accumulate_cycle_row_coefficients(&mut sums, &self.factors, degree, row);
                        sums
                    },
                )
                .reduce(
                    || [F::zero(); MAX_DEGREE_PLUS_ONE],
                    |mut left, right| {
                        for coefficient_index in 0..=degree {
                            left[coefficient_index] += right[coefficient_index];
                        }
                        left
                    },
                )
        } else {
            (0..half).fold([F::zero(); MAX_DEGREE_PLUS_ONE], |mut sums, row| {
                accumulate_cycle_row_coefficients(&mut sums, &self.factors, degree, row);
                sums
            })
        };
        sums[..=degree]
            .iter_mut()
            .map(|coefficient| *coefficient * active_scale)
            .collect()
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
}

fn accumulate_cycle_row_coefficients<F: Field, const N: usize>(
    sums: &mut [F; N],
    factors: &[Vec<F>],
    degree: usize,
    row: usize,
) {
    let mut coefficients = [F::zero(); N];
    coefficients[0] = F::one();
    for (current_degree, factor) in factors.iter().enumerate() {
        let low = factor[2 * row];
        let diff = factor[2 * row + 1] - low;
        coefficients[current_degree + 1] = F::zero();
        for coefficient_index in (0..=current_degree).rev() {
            let coefficient = coefficients[coefficient_index];
            coefficients[coefficient_index + 1] += coefficient * diff;
            coefficients[coefficient_index] = coefficient * low;
        }
    }
    for coefficient_index in 0..=degree {
        sums[coefficient_index] += coefficients[coefficient_index];
    }
}

fn instruction_read_raf_state<F: Field>(
    program: &'static Stage5CpuProgramPlan,
    claim: &Stage5SumcheckClaimPlan,
    inputs: &Stage5ProverInputs<'_, F>,
    store: &Stage5ValueStore<F>,
    active_scale: F,
) -> Result<InstructionReadRafStage5State<F>, Stage5KernelError> {
    const LOG_K: usize = 128;
    const XLEN: usize = 64;

    let witness = inputs
        .instruction_read_raf
        .ok_or(Stage5KernelError::MissingKernelInput {
            kernel: "jolt_stage5_instruction_read_raf",
            input: "instruction_read_raf",
        })?;
    let trace_rounds = log2_exact(witness.trace_len, "stage5.instruction_read_raf.trace_len")?;
    require_operand_count(
        "stage5.instruction_read_raf.input",
        LOG_K + trace_rounds,
        claim.num_rounds,
    )?;
    require_operand_count(
        "stage5.instruction_read_raf.lookup_indices",
        witness.trace_len,
        witness.lookup_indices.len(),
    )?;
    require_operand_count(
        "stage5.instruction_read_raf.lookup_table_indices",
        witness.trace_len,
        witness.lookup_table_indices.len(),
    )?;
    require_operand_count(
        "stage5.instruction_read_raf.is_interleaved_operands",
        witness.trace_len,
        witness.is_interleaved_operands.len(),
    )?;
    if witness.ra_virtual_log_k_chunk == 0 || !LOG_K.is_multiple_of(witness.ra_virtual_log_k_chunk)
    {
        return Err(Stage5KernelError::InvalidInputLength {
            input: "stage5.instruction_read_raf.ra_virtual_log_k_chunk",
            expected: LOG_K,
            actual: witness.ra_virtual_log_k_chunk,
        });
    }

    let table_count = LookupTableKind::<XLEN>::all().len();
    for table_index in witness.lookup_table_indices.iter().flatten() {
        if *table_index >= table_count {
            return Err(Stage5KernelError::InvalidInputLength {
                input: "stage5.instruction_read_raf.lookup_table_indices",
                expected: table_count,
                actual: table_index + 1,
            });
        }
    }

    let r_reduction = store.point("stage5.input.stage2.instruction.LookupOutput")?;
    require_operand_count(
        "stage5.input.stage2.instruction.LookupOutput",
        trace_rounds,
        r_reduction.len(),
    )?;
    let u_evals = EqPolynomial::<F>::evals(r_reduction, None);
    require_operand_count(
        "stage5.instruction_read_raf.u_evals",
        witness.trace_len,
        u_evals.len(),
    )?;

    let gamma = store.scalar("stage5.instruction_read_raf.gamma")?;
    let gamma2 = store
        .try_scalar("stage5.instruction_read_raf.gamma2")
        .unwrap_or_else(|| gamma * gamma);
    let ra_chunks = LOG_K / witness.ra_virtual_log_k_chunk;
    let outputs = instruction_read_raf_output_plans(program, claim, table_count, ra_chunks)?;

    let (lookup_groups, lookup_group_indices_by_cycle) =
        instruction_read_raf_lookup_groups(witness, &u_evals)?;
    let mut lookup_groups_by_table = vec![Vec::new(); table_count];
    for (group_index, group) in lookup_groups.iter().enumerate() {
        if let Some(table_index) = group.lookup_table_index {
            lookup_groups_by_table[table_index].push(group_index);
        }
    }

    Ok(InstructionReadRafStage5State {
        trace_len: witness.trace_len,
        lookup_indices: witness.lookup_indices.to_vec(),
        lookup_table_indices: witness.lookup_table_indices.to_vec(),
        is_interleaved_operands: witness.is_interleaved_operands.to_vec(),
        lookup_groups,
        lookup_group_indices_by_cycle,
        lookup_groups_by_table,
        ra_virtual_log_k_chunk: witness.ra_virtual_log_k_chunk,
        u_evals,
        gamma,
        gamma2,
        active_scale,
        round: 0,
        address_challenges: Vec::with_capacity(LOG_K),
        cycle_challenges: Vec::with_capacity(trace_rounds),
        address_phase: None,
        left_operand_checkpoint: F::zero(),
        right_operand_checkpoint: F::zero(),
        identity_checkpoint: F::zero(),
        read_prefix_checkpoints: ALL_PREFIXES
            .iter()
            .map(|prefix| prefix.default_checkpoint::<F>())
            .collect(),
        cycle_state: None,
        outputs,
    })
}

fn instruction_read_raf_lookup_groups<F: Field>(
    witness: Stage5InstructionReadRafWitness<'_>,
    u_evals: &[F],
) -> Result<(Vec<InstructionReadRafLookupGroup<F>>, Vec<usize>), Stage5KernelError> {
    require_operand_count(
        "stage5.instruction_read_raf.group_u_evals",
        witness.trace_len,
        u_evals.len(),
    )?;

    let mut index_by_key: std::collections::HashMap<(u128, Option<usize>, bool), usize> =
        std::collections::HashMap::with_capacity(witness.trace_len);
    let mut groups = Vec::<InstructionReadRafLookupGroup<F>>::new();
    let mut group_indices_by_cycle = Vec::with_capacity(witness.trace_len);
    for (cycle, u_eval) in u_evals.iter().copied().enumerate().take(witness.trace_len) {
        let key = (
            witness.lookup_indices[cycle],
            witness.lookup_table_indices[cycle],
            witness.is_interleaved_operands[cycle],
        );
        if let Some(&group_index) = index_by_key.get(&key) {
            groups[group_index].u_eval_sum += u_eval;
            groups[group_index].phase_u_eval_sum += u_eval;
            group_indices_by_cycle.push(group_index);
        } else {
            let group_index = groups.len();
            let _ = index_by_key.insert(key, group_index);
            groups.push(InstructionReadRafLookupGroup {
                lookup_index: key.0,
                lookup_table_index: key.1,
                is_interleaved_operands: key.2,
                u_eval_sum: u_eval,
                phase_u_eval_sum: u_eval,
            });
            group_indices_by_cycle.push(group_index);
        }
    }
    Ok((groups, group_indices_by_cycle))
}

fn instruction_read_raf_output_plans(
    program: &'static Stage5CpuProgramPlan,
    claim: &Stage5SumcheckClaimPlan,
    table_count: usize,
    ra_chunks: usize,
) -> Result<Vec<InstructionReadRafOutputPlan>, Stage5KernelError> {
    let instance = program
        .instance_results
        .iter()
        .find(|instance| {
            instance.claim == claim.symbol
                && instance.relation == Stage5Relation::InstructionReadRaf.symbol()
        })
        .ok_or(Stage5KernelError::MissingClaim {
            batch: "stage5.instruction_read_raf.outputs",
            claim: claim.symbol,
        })?;

    let mut outputs = Vec::with_capacity(table_count + ra_chunks + 1);
    let mut table_flags = vec![false; table_count];
    let mut instruction_ra = vec![false; ra_chunks];
    let mut has_raf_flag = false;

    for eval in program
        .evals
        .iter()
        .filter(|eval| eval.source == instance.source)
    {
        if let Some(suffix) = eval
            .name
            .strip_prefix("stage5.instruction_read_raf.eval.LookupTableFlag_")
        {
            let index = parse_instruction_read_raf_eval_index(
                suffix,
                "stage5.instruction_read_raf.eval.LookupTableFlag_",
            )?;
            if index >= table_count || table_flags[index] {
                return Err(Stage5KernelError::InvalidProof {
                    driver: eval.name,
                    reason: "invalid instruction read raf table flag eval",
                });
            }
            table_flags[index] = true;
            outputs.push(InstructionReadRafOutputPlan {
                index: eval.index,
                name: eval.name,
                oracle: eval.oracle,
                kind: InstructionReadRafOutputKind::LookupTableFlag(index),
            });
            continue;
        }

        if let Some(suffix) = eval
            .name
            .strip_prefix("stage5.instruction_read_raf.eval.InstructionRa_")
        {
            let index = parse_instruction_read_raf_eval_index(
                suffix,
                "stage5.instruction_read_raf.eval.InstructionRa_",
            )?;
            if index >= ra_chunks || instruction_ra[index] {
                return Err(Stage5KernelError::InvalidProof {
                    driver: eval.name,
                    reason: "invalid instruction read raf ra eval",
                });
            }
            instruction_ra[index] = true;
            outputs.push(InstructionReadRafOutputPlan {
                index: eval.index,
                name: eval.name,
                oracle: eval.oracle,
                kind: InstructionReadRafOutputKind::InstructionRa(index),
            });
            continue;
        }

        if eval.name == "stage5.instruction_read_raf.eval.InstructionRafFlag" {
            if has_raf_flag {
                return Err(Stage5KernelError::InvalidProof {
                    driver: eval.name,
                    reason: "duplicate instruction read raf flag eval",
                });
            }
            has_raf_flag = true;
            outputs.push(InstructionReadRafOutputPlan {
                index: eval.index,
                name: eval.name,
                oracle: eval.oracle,
                kind: InstructionReadRafOutputKind::InstructionRafFlag,
            });
        }
    }

    if table_flags.iter().any(|seen| !*seen) {
        return Err(Stage5KernelError::MissingValue {
            symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_",
        });
    }
    if instruction_ra.iter().any(|seen| !*seen) {
        return Err(Stage5KernelError::MissingValue {
            symbol: "stage5.instruction_read_raf.eval.InstructionRa_",
        });
    }
    if !has_raf_flag {
        return Err(Stage5KernelError::MissingValue {
            symbol: "stage5.instruction_read_raf.eval.InstructionRafFlag",
        });
    }
    outputs.sort_by_key(|output| output.index);
    Ok(outputs)
}

fn parse_instruction_read_raf_eval_index(
    suffix: &str,
    prefix: &'static str,
) -> Result<usize, Stage5KernelError> {
    suffix
        .parse::<usize>()
        .map_err(|_| Stage5KernelError::InvalidProof {
            driver: prefix,
            reason: "invalid instruction read raf eval suffix",
        })
}

fn ram_ra_claim_reduction_state<F: Field>(
    claim: &Stage5SumcheckClaimPlan,
    inputs: &Stage5ProverInputs<'_, F>,
    store: &Stage5ValueStore<F>,
    active_scale: F,
) -> Result<DenseStage5State<F>, Stage5KernelError> {
    let witness = inputs.ram_ra.ok_or(Stage5KernelError::MissingKernelInput {
        kernel: "jolt_stage5_batched",
        input: "ram_ra",
    })?;
    let trace_rounds = log2_exact(witness.trace_len, "stage5.ram.trace_len")?;
    let ram_rounds = log2_exact(witness.ram_k, "stage5.ram_k")?;
    require_operand_count(
        "stage5.ram_ra_claim_reduction.input",
        trace_rounds,
        claim.num_rounds,
    )?;

    let ram_raf_point = store.point("stage5.input.stage2.ram_raf.RamRa")?;
    let ram_rw_point = store.point("stage5.input.stage2.ram_read_write.RamRa")?;
    let ram_val_point = store.point("stage5.input.stage4.ram_val_check.RamRa")?;
    for (input, point) in [
        ("stage5.input.stage2.ram_raf.RamRa", ram_raf_point),
        ("stage5.input.stage2.ram_read_write.RamRa", ram_rw_point),
        ("stage5.input.stage4.ram_val_check.RamRa", ram_val_point),
    ] {
        require_operand_count(input, ram_rounds + trace_rounds, point.len())?;
    }
    let (address_point, r_cycle_raf) = ram_raf_point.split_at(ram_rounds);
    let (_, r_cycle_rw) = ram_rw_point.split_at(ram_rounds);
    let (_, r_cycle_val) = ram_val_point.split_at(ram_rounds);
    let address_eq = EqPolynomial::<F>::evals(address_point, None);
    let ram_ra = ram_ra_at_address(witness, &address_eq)?;
    let gamma = store.scalar("stage5.ram_ra_claim_reduction.gamma")?;
    let gamma2 = store
        .try_scalar("stage5.ram_ra_claim_reduction.gamma2")
        .unwrap_or_else(|| gamma * gamma);
    let mut eq_combined = EqPolynomial::<F>::evals(r_cycle_raf, None);
    let eq_rw = EqPolynomial::<F>::evals(r_cycle_rw, None);
    let eq_val = EqPolynomial::<F>::evals(r_cycle_val, None);
    require_operand_count(
        "stage5.ram_ra_claim_reduction.eq",
        witness.trace_len,
        eq_combined.len(),
    )?;
    for ((combined, rw), val) in eq_combined.iter_mut().zip(eq_rw).zip(eq_val) {
        *combined += gamma * rw + gamma2 * val;
    }

    Ok(DenseStage5State::new(
        vec![eq_combined, ram_ra],
        vec![DenseTerm {
            coefficient: F::one(),
            factors: vec![0, 1],
        }],
        vec![FactorOutput {
            name: "stage5.ram_ra_claim_reduction.eval.RamRa",
            oracle: "RamRa",
            factor: 1,
        }],
        active_scale,
    ))
}

fn ram_ra_at_address<F: Field>(
    witness: Stage5RamRaWitness<'_, F>,
    address_eq: &[F],
) -> Result<Vec<F>, Stage5KernelError> {
    let expected_len = witness.ram_k.checked_mul(witness.trace_len).ok_or(
        Stage5KernelError::InvalidInputLength {
            input: "stage5.ram_ra_claim_reduction.RamRa",
            expected: usize::MAX,
            actual: witness.ram_k,
        },
    )?;
    if !witness.ram_ra.is_empty() {
        require_operand_count(
            "stage5.ram_ra_claim_reduction.RamRa",
            expected_len,
            witness.ram_ra.len(),
        )?;
        let mut output = vec![F::zero(); witness.trace_len];
        for (address, &weight) in address_eq.iter().enumerate() {
            let base = address * witness.trace_len;
            for (cycle, output) in output.iter_mut().enumerate() {
                *output += weight * witness.ram_ra[base + cycle];
            }
        }
        return Ok(output);
    }

    let Some(remapped_addresses) = witness.remapped_addresses else {
        return Err(Stage5KernelError::MissingKernelInput {
            kernel: "jolt_stage5_batched",
            input: "ram_ra.remapped_addresses",
        });
    };
    require_operand_count(
        "stage5.ram_ra_claim_reduction.remapped_addresses",
        witness.trace_len,
        remapped_addresses.len(),
    )?;
    remapped_addresses
        .iter()
        .map(|address| match address {
            Some(address) => {
                address_eq
                    .get(*address)
                    .copied()
                    .ok_or(Stage5KernelError::InvalidInputLength {
                        input: "stage5.ram_ra_claim_reduction.remapped_addresses",
                        expected: address_eq.len(),
                        actual: address + 1,
                    })
            }
            None => Ok(F::zero()),
        })
        .collect()
}

fn registers_val_evaluation_state<F: Field>(
    claim: &Stage5SumcheckClaimPlan,
    inputs: &Stage5ProverInputs<'_, F>,
    store: &Stage5ValueStore<F>,
    active_scale: F,
) -> Result<DenseStage5State<F>, Stage5KernelError> {
    let witness = inputs
        .registers_val
        .ok_or(Stage5KernelError::MissingKernelInput {
            kernel: "jolt_stage5_batched",
            input: "registers_val",
        })?;
    require_operand_count(
        "stage5.registers_val_evaluation.RdInc",
        witness.trace_len,
        witness.rd_inc.len(),
    )?;
    require_operand_count(
        "stage5.registers_val_evaluation.input",
        log2_exact(witness.trace_len, "stage5.trace_len")?,
        claim.num_rounds,
    )?;

    let registers_val_point = store.point("stage5.input.stage4.registers.RegistersVal")?;
    let register_rounds = log2_exact(witness.register_count, "stage5.register_count")?;
    let trace_rounds = log2_exact(witness.trace_len, "stage5.trace_len")?;
    require_operand_count(
        "stage5.input.stage4.registers.RegistersVal",
        register_rounds + trace_rounds,
        registers_val_point.len(),
    )?;
    let (address_point, cycle_point) = registers_val_point.split_at(register_rounds);
    let address_eq = EqPolynomial::<F>::evals(address_point, None);
    let rd_wa_at_address = rd_wa_at_register_address(witness, &address_eq)?;
    let lt = lt_evals_big_endian(cycle_point);
    require_operand_count(
        "stage5.registers_val_evaluation.lt",
        witness.trace_len,
        lt.len(),
    )?;

    Ok(DenseStage5State::new(
        vec![witness.rd_inc.to_vec(), rd_wa_at_address, lt],
        vec![DenseTerm {
            coefficient: F::one(),
            factors: vec![0, 1, 2],
        }],
        vec![
            FactorOutput {
                name: "stage5.registers_val_evaluation.eval.RdInc",
                oracle: "RdInc",
                factor: 0,
            },
            FactorOutput {
                name: "stage5.registers_val_evaluation.eval.RdWa",
                oracle: "RdWa",
                factor: 1,
            },
        ],
        active_scale,
    ))
}

fn rd_wa_at_register_address<F: Field>(
    witness: Stage5RegistersValWitness<'_, F>,
    address_eq: &[F],
) -> Result<Vec<F>, Stage5KernelError> {
    let expected_len = witness
        .register_count
        .checked_mul(witness.trace_len)
        .ok_or(Stage5KernelError::InvalidInputLength {
            input: "stage5.registers_val_evaluation.RdWa",
            expected: usize::MAX,
            actual: witness.register_count,
        })?;
    if !witness.rd_wa.is_empty() {
        require_operand_count(
            "stage5.registers_val_evaluation.RdWa",
            expected_len,
            witness.rd_wa.len(),
        )?;
        let mut output = vec![F::zero(); witness.trace_len];
        for (address, &weight) in address_eq.iter().enumerate() {
            let base = address * witness.trace_len;
            for (cycle, output) in output.iter_mut().enumerate() {
                *output += weight * witness.rd_wa[base + cycle];
            }
        }
        return Ok(output);
    }

    let Some(rd_write_addresses) = witness.rd_write_addresses else {
        return Err(Stage5KernelError::MissingKernelInput {
            kernel: "jolt_stage5_batched",
            input: "registers_val.rd_wa",
        });
    };
    require_operand_count(
        "stage5.registers_val_evaluation.rd_write_addresses",
        witness.trace_len,
        rd_write_addresses.len(),
    )?;
    rd_write_addresses
        .iter()
        .map(|address| match address {
            Some(address) => {
                address_eq
                    .get(*address)
                    .copied()
                    .ok_or(Stage5KernelError::InvalidInputLength {
                        input: "stage5.registers_val_evaluation.rd_write_addresses",
                        expected: address_eq.len(),
                        actual: address + 1,
                    })
            }
            None => Ok(F::zero()),
        })
        .collect()
}

fn expected_batched_output_claim<F: Field>(
    context: Stage5KernelContext<'_>,
    store: &Stage5ValueStore<F>,
    evals: &[Stage5NamedEval<F>],
    point: &[F],
    batching_coeffs: &[F],
) -> Result<F, Stage5KernelError> {
    let mut expected = F::zero();
    for (claim, &coefficient) in context.batch_claims()?.iter().zip(batching_coeffs) {
        let instance = context
            .program
            .instance_results
            .iter()
            .find(|instance| {
                instance.claim == claim.symbol && instance.source == context.driver.symbol
            })
            .ok_or(Stage5KernelError::MissingClaim {
                batch: context.batch.symbol,
                claim: claim.symbol,
            })?;
        let local_point = point
            .get(instance.round_offset..instance.round_offset + instance.num_rounds)
            .ok_or(Stage5KernelError::InvalidInputLength {
                input: instance.symbol,
                expected: instance.round_offset + instance.num_rounds,
                actual: point.len(),
            })?;
        let relation = claim.relation.unwrap_or(instance.relation);
        let value = match Stage5Relation::from_symbol(relation)
            .ok_or(Stage5KernelError::UnknownRelation { relation })?
        {
            Stage5Relation::InstructionReadRaf => {
                expected_instruction_read_raf(store, evals, local_point)?
            }
            Stage5Relation::RamRaClaimReduction => {
                expected_ram_ra_claim_reduction(store, evals, local_point)?
            }
            Stage5Relation::RegistersValEvaluation => {
                expected_registers_val_evaluation(store, evals, local_point)?
            }
            relation @ Stage5Relation::Batched => {
                return Err(Stage5KernelError::KernelNotImplemented {
                    abi: relation.symbol(),
                })
            }
        };
        expected += coefficient * value;
    }
    Ok(expected)
}

fn expected_instruction_read_raf<F: Field>(
    store: &Stage5ValueStore<F>,
    evals: &[Stage5NamedEval<F>],
    local_point: &[F],
) -> Result<F, Stage5KernelError> {
    const LOG_K: usize = 128;
    const XLEN: usize = 64;

    if local_point.len() < LOG_K {
        return Err(Stage5KernelError::InvalidInputLength {
            input: "stage5.instruction_read_raf.point",
            expected: LOG_K,
            actual: local_point.len(),
        });
    }

    let (r_address_prime, r_cycle) = local_point.split_at(LOG_K);
    let r_cycle_prime = reverse_slice(r_cycle);
    let r_reduction = store.point("stage5.input.stage2.instruction.LookupOutput")?;
    let eq_eval_r_reduction = EqPolynomial::<F>::mle(r_reduction, &r_cycle_prime);

    let left_operand_eval = operand_polynomial_eval(r_address_prime, true);
    let right_operand_eval = operand_polynomial_eval(r_address_prime, false);
    let identity_poly_eval = identity_polynomial_eval(r_address_prime);

    let table_values = LookupTableKind::<XLEN>::all()
        .iter()
        .map(|table| table.evaluate_mle::<F, F>(r_address_prime))
        .collect::<Vec<_>>();
    let table_flag_claims = indexed_evals_by_prefix(
        evals,
        "stage5.instruction_read_raf.eval.LookupTableFlag_",
        table_values.len(),
    )?;
    let val_claim = table_values
        .into_iter()
        .zip(table_flag_claims)
        .map(|(table_value, flag_claim)| table_value * flag_claim)
        .sum::<F>();

    let ra_claim =
        indexed_evals_by_prefix_any(evals, "stage5.instruction_read_raf.eval.InstructionRa_")?
            .into_iter()
            .product::<F>();
    let raf_flag_claim =
        eval_by_name(evals, "stage5.instruction_read_raf.eval.InstructionRafFlag")?;
    let gamma = store.scalar("stage5.instruction_read_raf.gamma")?;

    let raf_claim = (F::one() - raf_flag_claim) * (left_operand_eval + gamma * right_operand_eval)
        + raf_flag_claim * gamma * identity_poly_eval;
    Ok(eq_eval_r_reduction * ra_claim * (val_claim + gamma * raf_claim))
}

pub fn instruction_read_raf_output_evals<F: Field>(
    witness: Stage5InstructionReadRafWitness<'_>,
    address_point: &[F],
    cycle_point: &[F],
) -> Result<Stage5InstructionReadRafEvaluations<F>, Stage5KernelError> {
    const LOG_K: usize = 128;
    const XLEN: usize = 64;

    require_operand_count(
        "stage5.instruction_read_raf.address_point",
        LOG_K,
        address_point.len(),
    )?;
    let trace_len_from_point = 1usize.checked_shl(cycle_point.len() as u32).ok_or(
        Stage5KernelError::InvalidInputLength {
            input: "stage5.instruction_read_raf.cycle_point",
            expected: usize::BITS as usize,
            actual: cycle_point.len(),
        },
    )?;
    require_operand_count(
        "stage5.instruction_read_raf.trace_len",
        trace_len_from_point,
        witness.trace_len,
    )?;
    require_operand_count(
        "stage5.instruction_read_raf.lookup_indices",
        witness.trace_len,
        witness.lookup_indices.len(),
    )?;
    require_operand_count(
        "stage5.instruction_read_raf.lookup_table_indices",
        witness.trace_len,
        witness.lookup_table_indices.len(),
    )?;
    require_operand_count(
        "stage5.instruction_read_raf.is_interleaved_operands",
        witness.trace_len,
        witness.is_interleaved_operands.len(),
    )?;
    if witness.ra_virtual_log_k_chunk == 0 || !LOG_K.is_multiple_of(witness.ra_virtual_log_k_chunk)
    {
        return Err(Stage5KernelError::InvalidInputLength {
            input: "stage5.instruction_read_raf.ra_virtual_log_k_chunk",
            expected: LOG_K,
            actual: witness.ra_virtual_log_k_chunk,
        });
    }

    let table_count = LookupTableKind::<XLEN>::all().len();
    let ra_chunks = LOG_K / witness.ra_virtual_log_k_chunk;
    let cycle_eq = EqPolynomial::<F>::evals(cycle_point, None);
    require_operand_count(
        "stage5.instruction_read_raf.eq_cycle",
        witness.trace_len,
        cycle_eq.len(),
    )?;

    let mut grouped_weights =
        HashMap::<(u128, Option<usize>, bool), F>::with_capacity(witness.trace_len.min(1 << 14));
    for (((&lookup_index, table_index), is_interleaved), &weight) in witness
        .lookup_indices
        .iter()
        .zip(witness.lookup_table_indices.iter())
        .zip(witness.is_interleaved_operands.iter())
        .zip(&cycle_eq)
    {
        *grouped_weights
            .entry((lookup_index, *table_index, *is_interleaved))
            .or_insert_with(F::zero) += weight;
    }

    let mut lookup_table_flags = vec![F::zero(); table_count];
    let mut instruction_raf_flag = F::zero();
    for ((_, table_index, is_interleaved), &weight) in &grouped_weights {
        if let Some(table_index) = table_index {
            let Some(flag) = lookup_table_flags.get_mut(*table_index) else {
                return Err(Stage5KernelError::InvalidInputLength {
                    input: "stage5.instruction_read_raf.lookup_table_indices",
                    expected: table_count,
                    actual: *table_index + 1,
                });
            };
            *flag += weight;
        }
        if !*is_interleaved {
            instruction_raf_flag += weight;
        }
    }

    let chunk_bits = witness.ra_virtual_log_k_chunk;
    let chunk_mask = if chunk_bits == 128 {
        u128::MAX
    } else {
        (1u128 << chunk_bits) - 1
    };
    let instruction_ra = (0..ra_chunks)
        .map(|chunk| {
            let chunk_point = &address_point[chunk * chunk_bits..(chunk + 1) * chunk_bits];
            let eq_tables = eq_eval_bit_chunk_tables(chunk_point, 8);
            let shift = LOG_K - (chunk + 1) * chunk_bits;
            grouped_weights
                .iter()
                .map(|((lookup_index, _, _), &cycle_weight)| {
                    let chunk_value = (*lookup_index >> shift) & chunk_mask;
                    cycle_weight
                        * eq_eval_at_bits_from_chunk_tables(&eq_tables, chunk_value, chunk_bits, 8)
                })
                .sum()
        })
        .collect();

    Ok(Stage5InstructionReadRafEvaluations {
        lookup_table_flags,
        instruction_ra,
        instruction_raf_flag,
    })
}

fn expected_ram_ra_claim_reduction<F: Field>(
    store: &Stage5ValueStore<F>,
    evals: &[Stage5NamedEval<F>],
    local_point: &[F],
) -> Result<F, Stage5KernelError> {
    let r_cycle_reduced = reverse_slice(local_point);
    let r_cycle_raf = suffix_point(
        store.point("stage5.input.stage2.ram_raf.RamRa")?,
        r_cycle_reduced.len(),
        "stage5.input.stage2.ram_raf.RamRa",
    )?;
    let r_cycle_rw = suffix_point(
        store.point("stage5.input.stage2.ram_read_write.RamRa")?,
        r_cycle_reduced.len(),
        "stage5.input.stage2.ram_read_write.RamRa",
    )?;
    let r_cycle_val = suffix_point(
        store.point("stage5.input.stage4.ram_val_check.RamRa")?,
        r_cycle_reduced.len(),
        "stage5.input.stage4.ram_val_check.RamRa",
    )?;
    let gamma = store.scalar("stage5.ram_ra_claim_reduction.gamma")?;
    let eq_combined = EqPolynomial::<F>::mle(r_cycle_raf, &r_cycle_reduced)
        + gamma * EqPolynomial::<F>::mle(r_cycle_rw, &r_cycle_reduced)
        + gamma.square() * EqPolynomial::<F>::mle(r_cycle_val, &r_cycle_reduced);
    let ram_ra = eval_by_name(evals, "stage5.ram_ra_claim_reduction.eval.RamRa")?;
    Ok(eq_combined * ram_ra)
}

fn expected_registers_val_evaluation<F: Field>(
    store: &Stage5ValueStore<F>,
    evals: &[Stage5NamedEval<F>],
    local_point: &[F],
) -> Result<F, Stage5KernelError> {
    let registers_val_point = store.point("stage5.input.stage4.registers.RegistersVal")?;
    let r_cycle = suffix_point(
        registers_val_point,
        local_point.len(),
        "stage5.input.stage4.registers.RegistersVal",
    )?;
    let r_reduced = reverse_slice(local_point);
    let lt_eval = lt_polynomial_eval(&r_reduced, r_cycle);
    let rd_inc = eval_by_name(evals, "stage5.registers_val_evaluation.eval.RdInc")?;
    let rd_wa = eval_by_name(evals, "stage5.registers_val_evaluation.eval.RdWa")?;
    Ok(rd_inc * rd_wa * lt_eval)
}

fn eval_by_name<F: Field>(
    evals: &[Stage5NamedEval<F>],
    name: &'static str,
) -> Result<F, Stage5KernelError> {
    evals
        .iter()
        .find(|eval| eval.name == name)
        .map(|eval| eval.value)
        .ok_or(Stage5KernelError::MissingValue { symbol: name })
}

fn named_eval<F: Field>(name: &'static str, oracle: &'static str, value: F) -> Stage5NamedEval<F> {
    Stage5NamedEval {
        name,
        oracle,
        value,
    }
}

fn claim_relation(
    program: &'static Stage5CpuProgramPlan,
    claim: &Stage5SumcheckClaimPlan,
) -> Result<Stage5Relation, Stage5KernelError> {
    if let Some(relation) = claim.relation {
        return Stage5Relation::from_symbol(relation)
            .ok_or(Stage5KernelError::UnknownRelation { relation });
    }
    let kernel_symbol = claim.kernel.ok_or(Stage5KernelError::MissingKernel {
        driver: claim.symbol,
        kernel: "<missing>",
    })?;
    let kernel = find_kernel(program, kernel_symbol).ok_or(Stage5KernelError::MissingKernel {
        driver: claim.symbol,
        kernel: kernel_symbol,
    })?;
    Stage5Relation::from_symbol(kernel.relation).ok_or(Stage5KernelError::UnknownRelation {
        relation: kernel.relation,
    })
}

fn instance_round_offset(
    program: &'static Stage5CpuProgramPlan,
    driver: &'static str,
    claim: &'static str,
) -> Result<usize, Stage5KernelError> {
    program
        .instance_results
        .iter()
        .find(|instance| instance.source == driver && instance.claim == claim)
        .map(|instance| instance.round_offset)
        .ok_or(Stage5KernelError::MissingClaim {
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
    UnivariatePoly::new(combined)
}

fn round_poly_from_dense_terms<F: Field>(
    factors: &[Vec<F>],
    terms: &[DenseTerm<F>],
    active_scale: F,
    relation: Stage5Relation,
) -> Result<UnivariatePoly<F>, Stage5KernelError> {
    let half = factors.first().map_or(0, |factor| factor.len() / 2);
    for term in terms {
        if term.factors.len() > 3 {
            return Err(Stage5KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "stage5 dense term exceeds degree bound",
            });
        }
        if term.factors.iter().any(|factor| *factor >= factors.len()) {
            return Err(Stage5KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "stage5 dense term references missing factor",
            });
        }
    }

    let accumulators = if half >= DENSE_BIND_PAR_THRESHOLD {
        (0..half)
            .into_par_iter()
            .map(|row| dense_row_coefficients(factors, terms, row))
            .reduce(
                || [F::Accumulator::default(); 4],
                |mut left, right| {
                    for index in 0..left.len() {
                        left[index].merge(right[index]);
                    }
                    left
                },
            )
    } else {
        (0..half).fold([F::Accumulator::default(); 4], |mut total, row| {
            let row_coefficients = dense_row_coefficients(factors, terms, row);
            for index in 0..total.len() {
                total[index].merge(row_coefficients[index]);
            }
            total
        })
    };

    Ok(UnivariatePoly::new(
        accumulators
            .into_iter()
            .map(AdditiveAccumulator::reduce)
            .map(|coefficient| coefficient * active_scale)
            .collect(),
    ))
}

fn dense_row_coefficients<F: Field>(
    factors: &[Vec<F>],
    terms: &[DenseTerm<F>],
    row: usize,
) -> [F::Accumulator; 4] {
    let mut coefficients = [F::Accumulator::default(); 4];
    for term in terms {
        match term.factors.as_slice() {
            [] => coefficients[0].add(term.coefficient),
            [first] => {
                let (first0, first_delta) = linear_factor_pair(&factors[*first], row);
                coefficients[0].fmadd(term.coefficient, first0);
                coefficients[1].fmadd(term.coefficient, first_delta);
            }
            [first, second] => {
                let (first0, first_delta) = linear_factor_pair(&factors[*first], row);
                let (second0, second_delta) = linear_factor_pair(&factors[*second], row);
                accumulate_quadratic_coefficients(
                    &mut coefficients,
                    term.coefficient,
                    first0,
                    first_delta,
                    second0,
                    second_delta,
                );
            }
            [first, second, third] => {
                let (first0, first_delta) = linear_factor_pair(&factors[*first], row);
                let (second0, second_delta) = linear_factor_pair(&factors[*second], row);
                let (third0, third_delta) = linear_factor_pair(&factors[*third], row);
                accumulate_cubic_coefficients(
                    &mut coefficients,
                    term.coefficient,
                    first0,
                    first_delta,
                    second0,
                    second_delta,
                    third0,
                    third_delta,
                );
            }
            _ => unreachable!("dense terms are validated before evaluation"),
        }
    }
    coefficients
}

#[inline]
fn linear_factor_pair<F: Field>(factor: &[F], row: usize) -> (F, F) {
    let low = factor[2 * row];
    (low, factor[2 * row + 1] - low)
}

#[inline]
fn accumulate_quadratic_coefficients<F: Field>(
    coefficients: &mut [F::Accumulator; 4],
    scale: F,
    first0: F,
    first_delta: F,
    second0: F,
    second_delta: F,
) {
    coefficients[0].fmadd(scale * first0, second0);
    coefficients[1].fmadd(scale * first_delta, second0);
    coefficients[1].fmadd(scale * first0, second_delta);
    coefficients[2].fmadd(scale * first_delta, second_delta);
}

#[inline]
fn accumulate_cubic_coefficients<F: Field>(
    coefficients: &mut [F::Accumulator; 4],
    scale: F,
    first0: F,
    first_delta: F,
    second0: F,
    second_delta: F,
    third0: F,
    third_delta: F,
) {
    let second0_third0 = second0 * third0;
    let second_delta_third0 = second_delta * third0;
    let second0_third_delta = second0 * third_delta;
    let second_delta_third_delta = second_delta * third_delta;
    let scaled_first0 = scale * first0;
    let scaled_first_delta = scale * first_delta;

    coefficients[0].fmadd(scaled_first0, second0_third0);
    coefficients[1].fmadd(scaled_first_delta, second0_third0);
    coefficients[1].fmadd(scaled_first0, second_delta_third0);
    coefficients[1].fmadd(scaled_first0, second0_third_delta);
    coefficients[2].fmadd(scaled_first_delta, second_delta_third0);
    coefficients[2].fmadd(scaled_first_delta, second0_third_delta);
    coefficients[2].fmadd(scaled_first0, second_delta_third_delta);
    coefficients[3].fmadd(scaled_first_delta, second_delta_third_delta);
}

fn indexed_evals_by_prefix<F: Field>(
    evals: &[Stage5NamedEval<F>],
    prefix: &'static str,
    count: usize,
) -> Result<Vec<F>, Stage5KernelError> {
    let mut values = vec![None; count];
    for eval in evals {
        let Some(suffix) = eval.name.strip_prefix(prefix) else {
            continue;
        };
        let index = suffix
            .parse::<usize>()
            .map_err(|_| Stage5KernelError::InvalidProof {
                driver: prefix,
                reason: "invalid indexed eval suffix",
            })?;
        if index >= count || values[index].is_some() {
            return Err(Stage5KernelError::InvalidProof {
                driver: prefix,
                reason: "invalid indexed eval",
            });
        }
        values[index] = Some(eval.value);
    }
    values
        .into_iter()
        .map(|value| value.ok_or(Stage5KernelError::MissingValue { symbol: prefix }))
        .collect()
}

fn indexed_evals_by_prefix_any<F: Field>(
    evals: &[Stage5NamedEval<F>],
    prefix: &'static str,
) -> Result<Vec<F>, Stage5KernelError> {
    let mut indexed_values = Vec::new();
    for eval in evals {
        let Some(suffix) = eval.name.strip_prefix(prefix) else {
            continue;
        };
        let index = suffix
            .parse::<usize>()
            .map_err(|_| Stage5KernelError::InvalidProof {
                driver: prefix,
                reason: "invalid indexed eval suffix",
            })?;
        if indexed_values
            .iter()
            .any(|(existing_index, _)| *existing_index == index)
        {
            return Err(Stage5KernelError::InvalidProof {
                driver: prefix,
                reason: "duplicate indexed eval",
            });
        }
        indexed_values.push((index, eval.value));
    }
    if indexed_values.is_empty() {
        return Err(Stage5KernelError::MissingValue { symbol: prefix });
    }
    indexed_values.sort_by_key(|(index, _)| *index);
    for (expected, (actual, _)) in indexed_values.iter().enumerate() {
        if *actual != expected {
            return Err(Stage5KernelError::InvalidProof {
                driver: prefix,
                reason: "non-contiguous indexed eval",
            });
        }
    }
    Ok(indexed_values.into_iter().map(|(_, value)| value).collect())
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
    program: &'static Stage5CpuProgramPlan,
    store: &mut Stage5ValueStore<F>,
    transcript: &mut T,
    evals: &[Stage5NamedEval<F>],
) -> Result<Vec<Stage5OpeningClaimValue<F>>, Stage5KernelError>
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
                find_opening_claim(program, symbol).ok_or(Stage5KernelError::MissingClaim {
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
            opening_claims.push(Stage5OpeningClaimValue {
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
    program: &'a Stage5CpuProgramPlan,
    symbol: &str,
) -> Option<&'a Stage5OpeningClaimPlan> {
    program
        .opening_claims
        .iter()
        .find(|claim| claim.symbol == symbol)
}

fn polynomial_degree<F: Field>(poly: &UnivariatePoly<F>) -> usize {
    poly.coefficients()
        .iter()
        .rposition(|coefficient| *coefficient != F::zero())
        .unwrap_or(0)
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

fn single_operand<F: Field>(symbol: &'static str, operands: &[F]) -> Result<F, Stage5KernelError> {
    require_operand_count(symbol, 1, operands.len())?;
    Ok(operands[0])
}

fn require_operand_count(
    input: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), Stage5KernelError> {
    if expected == actual {
        Ok(())
    } else {
        Err(Stage5KernelError::InvalidInputLength {
            input,
            expected,
            actual,
        })
    }
}

fn suffix_point<'a, F: Field>(
    point: &'a [F],
    length: usize,
    input: &'static str,
) -> Result<&'a [F], Stage5KernelError> {
    point
        .get(point.len().saturating_sub(length)..)
        .filter(|suffix| suffix.len() == length)
        .ok_or(Stage5KernelError::InvalidInputLength {
            input,
            expected: length,
            actual: point.len(),
        })
}

fn reverse_slice<F: Field>(values: &[F]) -> Vec<F> {
    values.iter().rev().copied().collect()
}

fn normalize_instruction_read_raf_point<F: Field>(
    point: &[F],
) -> Result<Vec<F>, Stage5KernelError> {
    const LOG_K: usize = 128;
    if point.len() < LOG_K {
        return Err(Stage5KernelError::InvalidInputLength {
            input: "stage5.instruction_read_raf.point",
            expected: LOG_K,
            actual: point.len(),
        });
    }
    let mut normalized = point.to_vec();
    normalized[LOG_K..].reverse();
    Ok(normalized)
}

fn lt_polynomial_eval<F: Field>(x: &[F], y: &[F]) -> F {
    debug_assert_eq!(x.len(), y.len());
    let mut lt_eval = F::zero();
    let mut eq_term = F::one();
    for (x_i, y_i) in x.iter().zip(y.iter()) {
        lt_eval += (F::one() - *x_i) * *y_i * eq_term;
        eq_term *= F::one() - *x_i - *y_i + *x_i * *y_i + *x_i * *y_i;
    }
    lt_eval
}

fn lt_evals_big_endian<F: Field>(point: &[F]) -> Vec<F> {
    let mut evals = vec![F::zero(); 1usize << point.len()];
    for (index, r) in point.iter().rev().enumerate() {
        let (left, right) = evals.split_at_mut(1usize << index);
        left.iter_mut().zip(right).for_each(|(left, right)| {
            *right = *left * *r;
            *left += *r - *right;
        });
    }
    evals
}

fn operand_polynomial_eval<F: Field>(point: &[F], left: bool) -> F {
    let stride_offset = usize::from(!left);
    let operand_bits = point.len() / 2;
    (0..operand_bits)
        .map(|index| point[2 * index + stride_offset].mul_pow_2(operand_bits - 1 - index))
        .sum()
}

fn eq_eval_at_bits<F: Field>(point: &[F], bits: u128, num_bits: usize) -> F {
    debug_assert_eq!(point.len(), num_bits);
    point
        .iter()
        .enumerate()
        .map(|(index, &challenge)| {
            if ((bits >> (num_bits - 1 - index)) & 1) == 1 {
                challenge
            } else {
                F::one() - challenge
            }
        })
        .product()
}

fn eq_eval_bit_chunk_tables<F: Field>(point: &[F], chunk_bits: usize) -> Vec<Vec<F>> {
    point
        .chunks(chunk_bits)
        .map(|chunk| {
            let len = chunk.len();
            (0..(1usize << len))
                .map(|bits| eq_eval_at_bits(chunk, bits as u128, len))
                .collect()
        })
        .collect()
}

fn eq_eval_at_bits_from_chunk_tables<F: Field>(
    tables: &[Vec<F>],
    bits: u128,
    num_bits: usize,
    chunk_bits: usize,
) -> F {
    tables
        .iter()
        .enumerate()
        .map(|(chunk_index, table)| {
            let start = chunk_index * chunk_bits;
            let len = table.len().ilog2() as usize;
            let shift = num_bits - start - len;
            let mask = (1u128 << len) - 1;
            table[((bits >> shift) & mask) as usize]
        })
        .product()
}

#[cfg(test)]
#[inline]
fn lookup_bit(lookup_index: u128, index: usize, total_bits: usize) -> bool {
    ((lookup_index >> (total_bits - 1 - index)) & 1) == 1
}

fn identity_polynomial_eval<F: Field>(point: &[F]) -> F {
    point
        .iter()
        .enumerate()
        .map(|(index, value)| value.mul_pow_2(point.len() - 1 - index))
        .sum()
}

fn log2_exact(value: usize, input: &'static str) -> Result<usize, Stage5KernelError> {
    if value.is_power_of_two() {
        Ok(value.ilog2() as usize)
    } else {
        Err(Stage5KernelError::InvalidInputLength {
            input,
            expected: value.next_power_of_two(),
            actual: value,
        })
    }
}

pub fn execute_stage5_program<F, T, E>(
    program: &'static Stage5CpuProgramPlan,
    mode: Stage5ExecutionMode,
    executor: &mut E,
    transcript: &mut T,
) -> Result<Stage5ExecutionArtifacts<F>, Stage5KernelError>
where
    F: Field,
    T: Transcript<Challenge = F>,
    E: Stage5KernelExecutor<F>,
{
    let mut artifacts = Stage5ExecutionArtifacts::default();
    for step in program.steps {
        match step.kind {
            "transcript_squeeze" => {
                let squeeze =
                    find_squeeze(program, step.symbol).ok_or(Stage5KernelError::MissingValue {
                        symbol: step.symbol,
                    })?;
                let values = transcript.challenge_vector(squeeze.count);
                executor.observe_challenge_vector(squeeze, &values)?;
                artifacts.challenge_vectors.push(Stage5ChallengeVector {
                    symbol: squeeze.symbol,
                    values,
                });
            }
            "transcript_absorb_bytes" => {
                let absorb = find_absorb_bytes(program, step.symbol).ok_or(
                    Stage5KernelError::MissingValue {
                        symbol: step.symbol,
                    },
                )?;
                absorb_stage5_bytes(absorb, transcript);
            }
            "sumcheck_driver" => {
                let driver =
                    find_driver(program, step.symbol).ok_or(Stage5KernelError::MissingDriver {
                        driver: step.symbol,
                    })?;
                let kernel_symbol = driver.kernel.ok_or(Stage5KernelError::MissingKernel {
                    driver: driver.symbol,
                    kernel: "<missing>",
                })?;
                let kernel = find_kernel(program, kernel_symbol).ok_or(
                    Stage5KernelError::MissingKernel {
                        driver: driver.symbol,
                        kernel: kernel_symbol,
                    },
                )?;
                let batch =
                    find_batch(program, driver.batch).ok_or(Stage5KernelError::MissingBatch {
                        driver: driver.symbol,
                        batch: driver.batch,
                    })?;
                let context = Stage5KernelContext {
                    mode,
                    program,
                    kernel,
                    batch,
                    driver,
                };
                let output = match mode {
                    Stage5ExecutionMode::Prover => executor.prove_sumcheck(context, transcript)?,
                    Stage5ExecutionMode::Verifier => {
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
                return Err(Stage5KernelError::InvalidProgramStep {
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

fn absorb_stage5_bytes<T>(absorb: &'static Stage5TranscriptAbsorbBytesPlan, transcript: &mut T)
where
    T: Transcript,
{
    transcript.append(&LabelWithCount(
        absorb.label.as_bytes(),
        absorb.payload.len() as u64,
    ));
    transcript.append_bytes(absorb.payload.as_bytes());
}

fn find_squeeze<'a>(
    program: &'a Stage5CpuProgramPlan,
    symbol: &str,
) -> Option<&'a Stage5TranscriptSqueezePlan> {
    program
        .transcript_squeezes
        .iter()
        .find(|plan| plan.symbol == symbol)
}

fn find_absorb_bytes<'a>(
    program: &'a Stage5CpuProgramPlan,
    symbol: &str,
) -> Option<&'a Stage5TranscriptAbsorbBytesPlan> {
    program
        .transcript_absorb_bytes
        .iter()
        .find(|plan| plan.symbol == symbol)
}

fn find_driver<'a>(
    program: &'a Stage5CpuProgramPlan,
    symbol: &str,
) -> Option<&'a Stage5SumcheckDriverPlan> {
    program
        .drivers
        .iter()
        .find(|driver| driver.symbol == symbol)
}

fn find_kernel<'a>(
    program: &'a Stage5CpuProgramPlan,
    symbol: &str,
) -> Option<&'a Stage5KernelPlan> {
    program
        .kernels
        .iter()
        .find(|kernel| kernel.symbol == symbol)
}

fn find_batch<'a>(
    program: &'a Stage5CpuProgramPlan,
    symbol: &str,
) -> Option<&'a Stage5SumcheckBatchPlan> {
    program.batches.iter().find(|batch| batch.symbol == symbol)
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "tests use expect to keep failure context concise"
)]
mod tests {
    use super::*;
    use jolt_field::{Fr, RingCore};
    use jolt_sumcheck::SumcheckProof;
    use jolt_transcript::Blake2bTranscript;

    const PARAMS: Stage5Params = Stage5Params {
        field: "bn254_fr",
        pcs: "dory",
        transcript: "blake2b_transcript",
    };
    const STEPS: &[Stage5ProgramStepPlan] = &[
        Stage5ProgramStepPlan {
            kind: "transcript_squeeze",
            symbol: "stage5.gamma",
        },
        Stage5ProgramStepPlan {
            kind: "sumcheck_driver",
            symbol: "stage5.sumcheck",
        },
    ];
    const SQUEEZES: &[Stage5TranscriptSqueezePlan] = &[Stage5TranscriptSqueezePlan {
        symbol: "stage5.gamma",
        label: "stage5_gamma",
        kind: "challenge_scalar",
        count: 1,
    }];
    const KERNELS: &[Stage5KernelPlan] = &[Stage5KernelPlan {
        symbol: "jolt.cpu.stage5.batched",
        relation: "jolt.stage5.batched",
        kind: "sumcheck",
        backend: "cpu",
        abi: "jolt_stage5_batched",
    }];
    const CLAIM_INPUTS: &[&str] = &[];
    const CLAIMS: &[Stage5SumcheckClaimPlan] = &[Stage5SumcheckClaimPlan {
        symbol: "stage5.claim",
        stage: "stage5",
        domain: "jolt.trace_domain",
        num_rounds: 1,
        degree: 1,
        claim: "stage5.claim",
        kernel: Some("jolt.cpu.stage5.batched"),
        relation: Some("jolt.stage5.batched"),
        claim_value: "stage5.gamma",
        input_openings: CLAIM_INPUTS,
    }];
    const ORDERED_CLAIMS: &[&str] = &["stage5.claim"];
    const ROUND_SCHEDULE: &[usize] = &[1];
    const BATCHES: &[Stage5SumcheckBatchPlan] = &[Stage5SumcheckBatchPlan {
        symbol: "stage5.batch",
        stage: "stage5",
        proof_slot: "stage5.sumcheck",
        policy: "test",
        count: 1,
        ordered_claims: ORDERED_CLAIMS,
        claim_operands: ORDERED_CLAIMS,
        claim_label: "sumcheck_claim",
        round_label: "sumcheck_poly",
        round_schedule: ROUND_SCHEDULE,
    }];
    const DRIVERS: &[Stage5SumcheckDriverPlan] = &[Stage5SumcheckDriverPlan {
        symbol: "stage5.sumcheck",
        stage: "stage5",
        proof_slot: "stage5.sumcheck",
        kernel: Some("jolt.cpu.stage5.batched"),
        relation: Some("jolt.stage5.batched"),
        batch: "stage5.batch",
        policy: "test",
        round_schedule: ROUND_SCHEDULE,
        claim_label: "sumcheck_claim",
        round_label: "sumcheck_poly",
        num_rounds: 1,
        degree: 1,
    }];
    const PROGRAM: Stage5CpuProgramPlan = Stage5CpuProgramPlan {
        role: "prover",
        params: PARAMS,
        steps: STEPS,
        transcript_squeezes: SQUEEZES,
        transcript_absorb_bytes: &[],
        opening_inputs: &[],
        field_constants: &[],
        field_exprs: &[],
        kernels: KERNELS,
        claims: CLAIMS,
        batches: BATCHES,
        drivers: DRIVERS,
        instance_results: &[],
        evals: &[],
        point_slices: &[],
        point_concats: &[],
        opening_claims: &[],
        opening_equalities: &[],
        opening_batches: &[],
    };

    const REGISTERS_STEPS: &[Stage5ProgramStepPlan] = &[Stage5ProgramStepPlan {
        kind: "sumcheck_driver",
        symbol: "stage5.registers.sumcheck",
    }];
    const REGISTERS_OPENING_INPUTS: &[Stage5OpeningInputPlan] = &[Stage5OpeningInputPlan {
        symbol: "stage5.input.stage4.registers.RegistersVal",
        source_stage: "stage4",
        source_claim: "stage4.registers_read_write.opening.RegistersVal",
        oracle: "RegistersVal",
        domain: "jolt.stage4_registers_rw_domain",
        point_arity: 3,
        claim_kind: "virtual",
    }];
    const REGISTERS_KERNELS: &[Stage5KernelPlan] = &[
        Stage5KernelPlan {
            symbol: "jolt.cpu.stage5.registers_val_evaluation",
            relation: "jolt.stage5.registers_val_evaluation",
            kind: "sumcheck",
            backend: "cpu",
            abi: "jolt_stage5_registers_val_evaluation",
        },
        Stage5KernelPlan {
            symbol: "jolt.cpu.stage5.batched",
            relation: "jolt.stage5.batched",
            kind: "sumcheck",
            backend: "cpu",
            abi: "jolt_stage5_batched",
        },
    ];
    const REGISTERS_CLAIM_INPUTS: &[&str] = &["stage5.input.stage4.registers.RegistersVal"];
    const REGISTERS_CLAIMS: &[Stage5SumcheckClaimPlan] = &[Stage5SumcheckClaimPlan {
        symbol: "stage5.registers_val_evaluation.input",
        stage: "stage5",
        domain: "jolt.trace_domain",
        num_rounds: 2,
        degree: 3,
        claim: "stage5.registers_val_evaluation.registers_val",
        kernel: Some("jolt.cpu.stage5.registers_val_evaluation"),
        relation: Some("jolt.stage5.registers_val_evaluation"),
        claim_value: "stage5.input.stage4.registers.RegistersVal",
        input_openings: REGISTERS_CLAIM_INPUTS,
    }];
    const REGISTERS_ORDERED_CLAIMS: &[&str] = &["stage5.registers_val_evaluation.input"];
    const REGISTERS_ROUND_SCHEDULE: &[usize] = &[2];
    const REGISTERS_BATCHES: &[Stage5SumcheckBatchPlan] = &[Stage5SumcheckBatchPlan {
        symbol: "stage5.registers.batch",
        stage: "stage5",
        proof_slot: "stage5.registers.sumcheck",
        policy: "test",
        count: 1,
        ordered_claims: REGISTERS_ORDERED_CLAIMS,
        claim_operands: REGISTERS_ORDERED_CLAIMS,
        claim_label: "sumcheck_claim",
        round_label: "sumcheck_poly",
        round_schedule: REGISTERS_ROUND_SCHEDULE,
    }];
    const REGISTERS_DRIVERS: &[Stage5SumcheckDriverPlan] = &[Stage5SumcheckDriverPlan {
        symbol: "stage5.registers.sumcheck",
        stage: "stage5",
        proof_slot: "stage5.registers.sumcheck",
        kernel: Some("jolt.cpu.stage5.batched"),
        relation: Some("jolt.stage5.batched"),
        batch: "stage5.registers.batch",
        policy: "test",
        round_schedule: REGISTERS_ROUND_SCHEDULE,
        claim_label: "sumcheck_claim",
        round_label: "sumcheck_poly",
        num_rounds: 2,
        degree: 3,
    }];
    const REGISTERS_INSTANCES: &[Stage5SumcheckInstanceResultPlan] =
        &[Stage5SumcheckInstanceResultPlan {
            symbol: "stage5.registers_val_evaluation.instance",
            source: "stage5.registers.sumcheck",
            claim: "stage5.registers_val_evaluation.input",
            relation: "jolt.stage5.registers_val_evaluation",
            index: 0,
            point_arity: 2,
            num_rounds: 2,
            round_offset: 0,
            point_order: "reverse",
            degree: 3,
        }];
    const REGISTERS_EVALS: &[Stage5SumcheckEvalPlan] = &[
        Stage5SumcheckEvalPlan {
            symbol: "stage5.registers_val_evaluation.eval.RdInc",
            source: "stage5.registers.sumcheck",
            name: "stage5.registers_val_evaluation.eval.RdInc",
            index: 0,
            oracle: "RdInc",
        },
        Stage5SumcheckEvalPlan {
            symbol: "stage5.registers_val_evaluation.eval.RdWa",
            source: "stage5.registers.sumcheck",
            name: "stage5.registers_val_evaluation.eval.RdWa",
            index: 1,
            oracle: "RdWa",
        },
    ];
    const REGISTERS_PROGRAM: Stage5CpuProgramPlan = Stage5CpuProgramPlan {
        role: "prover",
        params: PARAMS,
        steps: REGISTERS_STEPS,
        transcript_squeezes: &[],
        transcript_absorb_bytes: &[],
        opening_inputs: REGISTERS_OPENING_INPUTS,
        field_constants: &[],
        field_exprs: &[],
        kernels: REGISTERS_KERNELS,
        claims: REGISTERS_CLAIMS,
        batches: REGISTERS_BATCHES,
        drivers: REGISTERS_DRIVERS,
        instance_results: REGISTERS_INSTANCES,
        evals: REGISTERS_EVALS,
        point_slices: &[],
        point_concats: &[],
        opening_claims: &[],
        opening_equalities: &[],
        opening_batches: &[],
    };

    const RAM_RA_STEPS: &[Stage5ProgramStepPlan] = &[Stage5ProgramStepPlan {
        kind: "sumcheck_driver",
        symbol: "stage5.ram_ra.sumcheck",
    }];
    const RAM_RA_OPENING_INPUTS: &[Stage5OpeningInputPlan] = &[
        Stage5OpeningInputPlan {
            symbol: "stage5.input.stage2.ram_raf.RamRa",
            source_stage: "stage2",
            source_claim: "stage2.ram_raf.opening.RamRa",
            oracle: "RamRa",
            domain: "jolt.stage2_ram_rw_domain",
            point_arity: 3,
            claim_kind: "virtual",
        },
        Stage5OpeningInputPlan {
            symbol: "stage5.input.stage2.ram_read_write.RamRa",
            source_stage: "stage2",
            source_claim: "stage2.ram_read_write.opening.RamRa",
            oracle: "RamRa",
            domain: "jolt.stage2_ram_rw_domain",
            point_arity: 3,
            claim_kind: "virtual",
        },
        Stage5OpeningInputPlan {
            symbol: "stage5.input.stage4.ram_val_check.RamRa",
            source_stage: "stage4",
            source_claim: "stage4.ram_val_check.opening.RamRa",
            oracle: "RamRa",
            domain: "jolt.stage2_ram_rw_domain",
            point_arity: 3,
            claim_kind: "virtual",
        },
    ];
    const RAM_RA_FIELD_CONSTANTS: &[Stage5FieldConstantPlan] = &[Stage5FieldConstantPlan {
        symbol: "stage5.ram_ra_claim_reduction.gamma",
        field: "bn254_fr",
        value: 2,
    }];
    const RAM_RA_GAMMA2_OPERANDS: &[&str] = &["stage5.ram_ra_claim_reduction.gamma"];
    const RAM_RA_RW_TERM_OPERANDS: &[&str] = &[
        "stage5.ram_ra_claim_reduction.gamma",
        "stage5.input.stage2.ram_read_write.RamRa",
    ];
    const RAM_RA_VAL_TERM_OPERANDS: &[&str] = &[
        "stage5.ram_ra_claim_reduction.gamma2",
        "stage5.input.stage4.ram_val_check.RamRa",
    ];
    const RAM_RA_PARTIAL_OPERANDS: &[&str] = &[
        "stage5.input.stage2.ram_raf.RamRa",
        "stage5.ram_ra_claim_reduction.term.RamRaReadWrite",
    ];
    const RAM_RA_CLAIM_OPERANDS_EXPR: &[&str] = &[
        "stage5.ram_ra_claim_reduction.partial.RafReadWrite",
        "stage5.ram_ra_claim_reduction.term.RamRaValCheck",
    ];
    const RAM_RA_FIELD_EXPRS: &[Stage5FieldExprPlan] = &[
        Stage5FieldExprPlan {
            symbol: "stage5.ram_ra_claim_reduction.gamma2",
            kind: "op",
            formula: "field.pow:2",
            operand_names: RAM_RA_GAMMA2_OPERANDS,
            operands: RAM_RA_GAMMA2_OPERANDS,
        },
        Stage5FieldExprPlan {
            symbol: "stage5.ram_ra_claim_reduction.term.RamRaReadWrite",
            kind: "op",
            formula: "field.mul",
            operand_names: RAM_RA_RW_TERM_OPERANDS,
            operands: RAM_RA_RW_TERM_OPERANDS,
        },
        Stage5FieldExprPlan {
            symbol: "stage5.ram_ra_claim_reduction.term.RamRaValCheck",
            kind: "op",
            formula: "field.mul",
            operand_names: RAM_RA_VAL_TERM_OPERANDS,
            operands: RAM_RA_VAL_TERM_OPERANDS,
        },
        Stage5FieldExprPlan {
            symbol: "stage5.ram_ra_claim_reduction.partial.RafReadWrite",
            kind: "op",
            formula: "field.add",
            operand_names: RAM_RA_PARTIAL_OPERANDS,
            operands: RAM_RA_PARTIAL_OPERANDS,
        },
        Stage5FieldExprPlan {
            symbol: "stage5.ram_ra_claim_reduction.claim_expr",
            kind: "op",
            formula: "field.add",
            operand_names: RAM_RA_CLAIM_OPERANDS_EXPR,
            operands: RAM_RA_CLAIM_OPERANDS_EXPR,
        },
    ];
    const RAM_RA_KERNELS: &[Stage5KernelPlan] = &[
        Stage5KernelPlan {
            symbol: "jolt.cpu.stage5.ram_ra_claim_reduction",
            relation: "jolt.stage5.ram_ra_claim_reduction",
            kind: "sumcheck",
            backend: "cpu",
            abi: "jolt_stage5_ram_ra_claim_reduction",
        },
        Stage5KernelPlan {
            symbol: "jolt.cpu.stage5.batched",
            relation: "jolt.stage5.batched",
            kind: "sumcheck",
            backend: "cpu",
            abi: "jolt_stage5_batched",
        },
    ];
    const RAM_RA_CLAIM_INPUTS: &[&str] = &[
        "stage5.input.stage2.ram_raf.RamRa",
        "stage5.input.stage2.ram_read_write.RamRa",
        "stage5.input.stage4.ram_val_check.RamRa",
    ];
    const RAM_RA_CLAIMS: &[Stage5SumcheckClaimPlan] = &[Stage5SumcheckClaimPlan {
        symbol: "stage5.ram_ra_claim_reduction.input",
        stage: "stage5",
        domain: "jolt.trace_domain",
        num_rounds: 2,
        degree: 2,
        claim: "stage5.ram_ra_claim_reduction.weighted_ram_ra",
        kernel: Some("jolt.cpu.stage5.ram_ra_claim_reduction"),
        relation: Some("jolt.stage5.ram_ra_claim_reduction"),
        claim_value: "stage5.ram_ra_claim_reduction.claim_expr",
        input_openings: RAM_RA_CLAIM_INPUTS,
    }];
    const RAM_RA_ORDERED_CLAIMS: &[&str] = &["stage5.ram_ra_claim_reduction.input"];
    const RAM_RA_ROUND_SCHEDULE: &[usize] = &[2];
    const RAM_RA_BATCHES: &[Stage5SumcheckBatchPlan] = &[Stage5SumcheckBatchPlan {
        symbol: "stage5.ram_ra.batch",
        stage: "stage5",
        proof_slot: "stage5.ram_ra.sumcheck",
        policy: "test",
        count: 1,
        ordered_claims: RAM_RA_ORDERED_CLAIMS,
        claim_operands: RAM_RA_ORDERED_CLAIMS,
        claim_label: "sumcheck_claim",
        round_label: "sumcheck_poly",
        round_schedule: RAM_RA_ROUND_SCHEDULE,
    }];
    const RAM_RA_DRIVERS: &[Stage5SumcheckDriverPlan] = &[Stage5SumcheckDriverPlan {
        symbol: "stage5.ram_ra.sumcheck",
        stage: "stage5",
        proof_slot: "stage5.ram_ra.sumcheck",
        kernel: Some("jolt.cpu.stage5.batched"),
        relation: Some("jolt.stage5.batched"),
        batch: "stage5.ram_ra.batch",
        policy: "test",
        round_schedule: RAM_RA_ROUND_SCHEDULE,
        claim_label: "sumcheck_claim",
        round_label: "sumcheck_poly",
        num_rounds: 2,
        degree: 2,
    }];
    const RAM_RA_INSTANCES: &[Stage5SumcheckInstanceResultPlan] =
        &[Stage5SumcheckInstanceResultPlan {
            symbol: "stage5.ram_ra_claim_reduction.instance",
            source: "stage5.ram_ra.sumcheck",
            claim: "stage5.ram_ra_claim_reduction.input",
            relation: "jolt.stage5.ram_ra_claim_reduction",
            index: 0,
            point_arity: 2,
            num_rounds: 2,
            round_offset: 0,
            point_order: "reverse",
            degree: 2,
        }];
    const RAM_RA_EVALS: &[Stage5SumcheckEvalPlan] = &[Stage5SumcheckEvalPlan {
        symbol: "stage5.ram_ra_claim_reduction.eval.RamRa",
        source: "stage5.ram_ra.sumcheck",
        name: "stage5.ram_ra_claim_reduction.eval.RamRa",
        index: 0,
        oracle: "RamRa",
    }];
    const RAM_RA_PROGRAM: Stage5CpuProgramPlan = Stage5CpuProgramPlan {
        role: "prover",
        params: PARAMS,
        steps: RAM_RA_STEPS,
        transcript_squeezes: &[],
        transcript_absorb_bytes: &[],
        opening_inputs: RAM_RA_OPENING_INPUTS,
        field_constants: RAM_RA_FIELD_CONSTANTS,
        field_exprs: RAM_RA_FIELD_EXPRS,
        kernels: RAM_RA_KERNELS,
        claims: RAM_RA_CLAIMS,
        batches: RAM_RA_BATCHES,
        drivers: RAM_RA_DRIVERS,
        instance_results: RAM_RA_INSTANCES,
        evals: RAM_RA_EVALS,
        point_slices: &[],
        point_concats: &[],
        opening_claims: &[],
        opening_equalities: &[],
        opening_batches: &[],
    };

    #[derive(Default)]
    struct RecordingExecutor {
        observed_challenges: usize,
        proved: bool,
    }

    impl Stage5KernelExecutor<Fr> for RecordingExecutor {
        fn observe_challenge_vector(
            &mut self,
            plan: &'static Stage5TranscriptSqueezePlan,
            values: &[Fr],
        ) -> Result<(), Stage5KernelError> {
            assert_eq!(plan.symbol, "stage5.gamma");
            assert_eq!(values.len(), 1);
            self.observed_challenges += 1;
            Ok(())
        }

        fn prove_sumcheck<T>(
            &mut self,
            context: Stage5KernelContext<'_>,
            _transcript: &mut T,
        ) -> Result<Stage5SumcheckOutput<Fr>, Stage5KernelError>
        where
            T: Transcript<Challenge = Fr>,
        {
            assert_eq!(context.mode, Stage5ExecutionMode::Prover);
            assert_eq!(context.abi_kind()?, Stage5KernelAbi::Batched);
            assert_eq!(context.relation_kind()?, Stage5Relation::Batched);
            assert_eq!(context.batch_claims()?.len(), 1);
            self.proved = true;
            Ok(Stage5SumcheckOutput {
                driver: context.driver.symbol,
                point: vec![Fr::from_u64(7)],
                evals: Vec::new(),
                opening_claims: Vec::new(),
                proof: SumcheckProof {
                    round_polynomials: Vec::new(),
                },
            })
        }

        fn verify_sumcheck<T>(
            &mut self,
            context: Stage5KernelContext<'_>,
            _transcript: &mut T,
        ) -> Result<Stage5SumcheckOutput<Fr>, Stage5KernelError>
        where
            T: Transcript<Challenge = Fr>,
        {
            Err(Stage5KernelError::WrongExecutorMode {
                driver: context.driver.symbol,
                expected: Stage5ExecutionMode::Prover,
                actual: Stage5ExecutionMode::Verifier,
            })
        }
    }

    #[test]
    fn stage5_program_executes_with_stage5_abi_and_relation_names() {
        let mut transcript = Blake2bTranscript::<Fr>::new(b"Jolt");
        let mut executor = RecordingExecutor::default();

        let artifacts = execute_stage5_program(
            &PROGRAM,
            Stage5ExecutionMode::Prover,
            &mut executor,
            &mut transcript,
        )
        .expect("stage5 program executes");

        assert_eq!(executor.observed_challenges, 1);
        assert!(executor.proved);
        assert_eq!(artifacts.challenge_vectors.len(), 1);
        assert_eq!(artifacts.sumchecks.len(), 1);
        assert_eq!(artifacts.sumchecks[0].driver, "stage5.sumcheck");
    }

    #[test]
    fn stage5_registers_val_prover_produces_verifiable_sumcheck() {
        let registers_point = frs(&[1, 2, 3]);
        let rd_inc = frs(&[2, 3, 4, 5]);
        let rd_write_addresses = [Some(0usize), Some(1usize), None, Some(1usize)];
        let input_eval = registers_val_input_claim(&registers_point, &rd_inc, &rd_write_addresses);
        let opening_inputs = vec![Stage5OpeningInputValue {
            symbol: "stage5.input.stage4.registers.RegistersVal",
            point: registers_point,
            eval: input_eval,
        }];
        let prover_inputs = Stage5ProverInputs::new(&opening_inputs).with_registers_val(
            Stage5RegistersValWitness {
                register_count: 2,
                trace_len: 4,
                rd_inc: &rd_inc,
                rd_wa: &[],
                rd_write_addresses: Some(&rd_write_addresses),
            },
        );
        let mut prover = Stage5ProverKernelExecutor::new(prover_inputs);
        let mut prover_transcript = Blake2bTranscript::<Fr>::new(b"Jolt");
        let artifacts = execute_stage5_program(
            &REGISTERS_PROGRAM,
            Stage5ExecutionMode::Prover,
            &mut prover,
            &mut prover_transcript,
        )
        .expect("registers val prover succeeds");

        assert_eq!(artifacts.sumchecks.len(), 1);
        assert_eq!(artifacts.sumchecks[0].evals.len(), 2);

        let proof = Stage5Proof {
            sumchecks: artifacts.sumchecks.clone(),
        };
        let mut verifier = Stage5ProofCarryingKernelExecutor::new(&proof, &opening_inputs);
        let mut verifier_transcript = Blake2bTranscript::<Fr>::new(b"Jolt");
        let verified = execute_stage5_program(
            &REGISTERS_PROGRAM,
            Stage5ExecutionMode::Verifier,
            &mut verifier,
            &mut verifier_transcript,
        )
        .expect("registers val verifier accepts prover output");

        assert_eq!(artifacts.sumchecks[0].point, verified.sumchecks[0].point);
        assert_eq!(
            named_eval_values(&artifacts.sumchecks[0].evals),
            named_eval_values(&verified.sumchecks[0].evals)
        );
    }

    #[test]
    fn stage5_ram_ra_prover_produces_verifiable_sumcheck() {
        let ram_raf_point = frs(&[1, 2, 3]);
        let ram_rw_point = frs(&[1, 5, 7]);
        let ram_val_point = frs(&[1, 11, 13]);
        let remapped_addresses = [Some(0usize), Some(1usize), None, Some(1usize)];
        let gamma = Fr::from_u64(2);
        let claim_raf = ram_ra_input_claim(&ram_raf_point, &remapped_addresses);
        let claim_rw = ram_ra_input_claim(&ram_rw_point, &remapped_addresses);
        let claim_val = ram_ra_input_claim(&ram_val_point, &remapped_addresses);
        let opening_inputs = vec![
            Stage5OpeningInputValue {
                symbol: "stage5.input.stage2.ram_raf.RamRa",
                point: ram_raf_point,
                eval: claim_raf,
            },
            Stage5OpeningInputValue {
                symbol: "stage5.input.stage2.ram_read_write.RamRa",
                point: ram_rw_point,
                eval: claim_rw,
            },
            Stage5OpeningInputValue {
                symbol: "stage5.input.stage4.ram_val_check.RamRa",
                point: ram_val_point,
                eval: claim_val,
            },
        ];
        let mut store = Stage5ValueStore::with_opening_inputs(&opening_inputs);
        store.seed_constants(&RAM_RA_PROGRAM);
        let _ = store
            .evaluate_available_field_exprs(&RAM_RA_PROGRAM)
            .expect("field exprs");
        assert_eq!(
            claim_raf + gamma * claim_rw + gamma.square() * claim_val,
            store
                .scalar("stage5.ram_ra_claim_reduction.claim_expr")
                .expect("claim expr")
        );

        let prover_inputs =
            Stage5ProverInputs::new(&opening_inputs).with_ram_ra(Stage5RamRaWitness {
                ram_k: 2,
                trace_len: 4,
                ram_ra: &[],
                remapped_addresses: Some(&remapped_addresses),
            });
        let mut prover = Stage5ProverKernelExecutor::new(prover_inputs);
        let mut prover_transcript = Blake2bTranscript::<Fr>::new(b"Jolt");
        let artifacts = execute_stage5_program(
            &RAM_RA_PROGRAM,
            Stage5ExecutionMode::Prover,
            &mut prover,
            &mut prover_transcript,
        )
        .expect("ram ra prover succeeds");

        assert_eq!(artifacts.sumchecks.len(), 1);
        assert_eq!(artifacts.sumchecks[0].evals.len(), 1);

        let proof = Stage5Proof {
            sumchecks: artifacts.sumchecks.clone(),
        };
        let mut verifier = Stage5ProofCarryingKernelExecutor::new(&proof, &opening_inputs);
        let mut verifier_transcript = Blake2bTranscript::<Fr>::new(b"Jolt");
        let verified = execute_stage5_program(
            &RAM_RA_PROGRAM,
            Stage5ExecutionMode::Verifier,
            &mut verifier,
            &mut verifier_transcript,
        )
        .expect("ram ra verifier accepts prover output");

        assert_eq!(artifacts.sumchecks[0].point, verified.sumchecks[0].point);
        assert_eq!(
            named_eval_values(&artifacts.sumchecks[0].evals),
            named_eval_values(&verified.sumchecks[0].evals)
        );
    }

    #[test]
    fn stage5_instruction_read_raf_prover_produces_verifiable_sumcheck() {
        const CHUNK_BITS: usize = 64;
        let r_reduction = frs(&[3, 5]);
        let lookup_indices = [
            0u128,
            1u128 << 127,
            (0x1234u128 << 64) | 0xABCDu128,
            (0x55AAu128 << 96) | (0xAA55u128 << 32),
        ];
        let table_indices = [Some(0usize), Some(0usize), None, Some(0usize)];
        let is_interleaved = [true, false, true, false];
        let gamma = Fr::from_u64(2);
        let (lookup_output_claim, left_claim, right_claim) = instruction_read_raf_input_claim(
            &r_reduction,
            &lookup_indices,
            &table_indices,
            &is_interleaved,
        );
        let input_claim = lookup_output_claim + gamma * left_claim + gamma.square() * right_claim;
        let opening_inputs = vec![
            Stage5OpeningInputValue {
                symbol: "stage5.input.stage2.instruction.LookupOutput",
                point: r_reduction.clone(),
                eval: lookup_output_claim,
            },
            Stage5OpeningInputValue {
                symbol: "stage5.input.stage2.instruction.LeftLookupOperand",
                point: r_reduction.clone(),
                eval: left_claim,
            },
            Stage5OpeningInputValue {
                symbol: "stage5.input.stage2.instruction.RightLookupOperand",
                point: r_reduction,
                eval: right_claim,
            },
            Stage5OpeningInputValue {
                symbol: "stage5.instruction_read_raf.claim_expr",
                point: Vec::new(),
                eval: input_claim,
            },
        ];
        let program = instruction_read_raf_test_program(2, 128 / CHUNK_BITS);
        let prover_inputs = Stage5ProverInputs::new(&opening_inputs).with_instruction_read_raf(
            Stage5InstructionReadRafWitness {
                trace_len: 4,
                lookup_indices: &lookup_indices,
                lookup_table_indices: &table_indices,
                is_interleaved_operands: &is_interleaved,
                ra_virtual_log_k_chunk: CHUNK_BITS,
            },
        );
        let mut prover = Stage5ProverKernelExecutor::new(prover_inputs);
        let mut prover_transcript = Blake2bTranscript::<Fr>::new(b"Jolt");
        let artifacts = execute_stage5_program(
            program,
            Stage5ExecutionMode::Prover,
            &mut prover,
            &mut prover_transcript,
        )
        .expect("instruction read raf prover succeeds");

        assert_eq!(artifacts.sumchecks.len(), 1);
        assert_eq!(
            artifacts.sumchecks[0].evals.len(),
            LookupTableKind::<64>::all().len() + 128 / CHUNK_BITS + 1
        );

        let proof = Stage5Proof {
            sumchecks: artifacts.sumchecks.clone(),
        };
        let mut verifier = Stage5ProofCarryingKernelExecutor::new(&proof, &opening_inputs);
        let mut verifier_transcript = Blake2bTranscript::<Fr>::new(b"Jolt");
        let verified = execute_stage5_program(
            program,
            Stage5ExecutionMode::Verifier,
            &mut verifier,
            &mut verifier_transcript,
        )
        .expect("instruction read raf verifier accepts prover output");

        assert_eq!(artifacts.sumchecks[0].point, verified.sumchecks[0].point);
        assert_eq!(
            named_eval_values(&artifacts.sumchecks[0].evals),
            named_eval_values(&verified.sumchecks[0].evals)
        );
    }

    #[test]
    fn instruction_read_raf_output_evals_follow_trace_flags_and_ra_chunks() {
        const CHUNK_BITS: usize = 16;
        let lookup_indices = [
            0u128,
            1u128 << 112,
            0xABCDu128 << 96,
            (0x12u128 << 112) | 0x34u128,
        ];
        let table_indices = [Some(0usize), Some(2usize), None, Some(0usize)];
        let is_interleaved = [true, false, true, false];
        let address_point = (0..128)
            .map(|index| Fr::from_u64(index as u64 + 2))
            .collect::<Vec<_>>();
        let cycle_point = frs(&[3, 5]);
        let witness = Stage5InstructionReadRafWitness {
            trace_len: 4,
            lookup_indices: &lookup_indices,
            lookup_table_indices: &table_indices,
            is_interleaved_operands: &is_interleaved,
            ra_virtual_log_k_chunk: CHUNK_BITS,
        };

        let output = instruction_read_raf_output_evals(witness, &address_point, &cycle_point)
            .expect("instruction read raf evals");
        let cycle_eq = EqPolynomial::<Fr>::evals(&cycle_point, None);
        assert_eq!(output.lookup_table_flags[0], cycle_eq[0] + cycle_eq[3]);
        assert_eq!(output.lookup_table_flags[2], cycle_eq[1]);
        assert_eq!(output.instruction_raf_flag, cycle_eq[1] + cycle_eq[3]);
        assert_eq!(output.instruction_ra.len(), 8);

        for chunk in 0..output.instruction_ra.len() {
            let chunk_point = &address_point[chunk * CHUNK_BITS..(chunk + 1) * CHUNK_BITS];
            let shift = 128 - (chunk + 1) * CHUNK_BITS;
            let expected = lookup_indices
                .iter()
                .zip(&cycle_eq)
                .map(|(&lookup_index, &cycle_weight)| {
                    let chunk_value = (lookup_index >> shift) & ((1u128 << CHUNK_BITS) - 1);
                    cycle_weight * manual_eq_bits(chunk_point, chunk_value, CHUNK_BITS)
                })
                .sum::<Fr>();
            assert_eq!(output.instruction_ra[chunk], expected, "chunk={chunk}");
        }
    }

    fn frs(values: &[u64]) -> Vec<Fr> {
        values.iter().map(|value| Fr::from_u64(*value)).collect()
    }

    fn manual_eq_bits(point: &[Fr], bits: u128, num_bits: usize) -> Fr {
        point
            .iter()
            .enumerate()
            .map(|(index, &challenge)| {
                if ((bits >> (num_bits - 1 - index)) & 1) == 1 {
                    challenge
                } else {
                    Fr::from_u64(1) - challenge
                }
            })
            .product()
    }

    fn registers_val_input_claim(
        registers_point: &[Fr],
        rd_inc: &[Fr],
        rd_write_addresses: &[Option<usize>],
    ) -> Fr {
        let (address_point, cycle_point) = registers_point.split_at(1);
        let address_eq = EqPolynomial::<Fr>::evals(address_point, None);
        let lt = lt_evals_big_endian(cycle_point);
        rd_inc
            .iter()
            .zip(rd_write_addresses)
            .zip(lt)
            .map(|((&inc, address), lt)| {
                let wa = address
                    .and_then(|address| address_eq.get(address))
                    .copied()
                    .unwrap_or_else(|| Fr::from_u64(0));
                inc * wa * lt
            })
            .sum()
    }

    fn ram_ra_input_claim(ram_ra_point: &[Fr], remapped_addresses: &[Option<usize>]) -> Fr {
        let (address_point, cycle_point) = ram_ra_point.split_at(1);
        let address_eq = EqPolynomial::<Fr>::evals(address_point, None);
        let eq_cycle = EqPolynomial::<Fr>::evals(cycle_point, None);
        remapped_addresses
            .iter()
            .zip(eq_cycle)
            .map(|(address, eq)| {
                let ra = address
                    .and_then(|address| address_eq.get(address))
                    .copied()
                    .unwrap_or_else(|| Fr::from_u64(0));
                ra * eq
            })
            .sum()
    }

    fn instruction_read_raf_input_claim(
        r_reduction: &[Fr],
        lookup_indices: &[u128],
        table_indices: &[Option<usize>],
        is_interleaved: &[bool],
    ) -> (Fr, Fr, Fr) {
        let tables = LookupTableKind::<64>::all();
        let eq_cycle = EqPolynomial::<Fr>::evals(r_reduction, None);
        let mut lookup_output_claim = Fr::from_u64(0);
        let mut left_claim = Fr::from_u64(0);
        let mut right_claim = Fr::from_u64(0);
        for (((&lookup_index, table_index), &is_interleaved), &cycle_weight) in lookup_indices
            .iter()
            .zip(table_indices)
            .zip(is_interleaved)
            .zip(&eq_cycle)
        {
            let address_point = field_bit_point_128(lookup_index);
            if let Some(table_index) = table_index {
                lookup_output_claim +=
                    cycle_weight * tables[*table_index].evaluate_mle::<Fr, Fr>(&address_point);
            }
            if is_interleaved {
                left_claim += cycle_weight * operand_polynomial_eval(&address_point, true);
                right_claim += cycle_weight * operand_polynomial_eval(&address_point, false);
            } else {
                right_claim += cycle_weight * identity_polynomial_eval(&address_point);
            }
        }
        (lookup_output_claim, left_claim, right_claim)
    }

    fn field_bit_point_128(value: u128) -> Vec<Fr> {
        (0..128)
            .map(|index| Fr::from_bool(lookup_bit(value, index, 128)))
            .collect()
    }

    fn instruction_read_raf_test_program(
        trace_rounds: usize,
        ra_chunks: usize,
    ) -> &'static Stage5CpuProgramPlan {
        let table_count = LookupTableKind::<64>::all().len();
        let driver_symbol = "stage5.instruction_read_raf.sumcheck";
        let claim_symbol = "stage5.instruction_read_raf.input";
        let ordered_claims = Box::leak(vec![claim_symbol].into_boxed_slice());
        let round_schedule = Box::leak(vec![128, trace_rounds].into_boxed_slice());
        let claim_inputs = Box::leak(
            vec![
                "stage5.input.stage2.instruction.LookupOutput",
                "stage5.input.stage2.instruction.LeftLookupOperand",
                "stage5.input.stage2.instruction.RightLookupOperand",
            ]
            .into_boxed_slice(),
        );
        let mut evals = Vec::with_capacity(table_count + ra_chunks + 1);
        for index in 0..table_count {
            let name = leak_test_str(format!(
                "stage5.instruction_read_raf.eval.LookupTableFlag_{index}"
            ));
            let oracle = leak_test_str(format!("LookupTableFlag_{index}"));
            evals.push(Stage5SumcheckEvalPlan {
                symbol: name,
                source: driver_symbol,
                name,
                index,
                oracle,
            });
        }
        for index in 0..ra_chunks {
            let name = leak_test_str(format!(
                "stage5.instruction_read_raf.eval.InstructionRa_{index}"
            ));
            let oracle = leak_test_str(format!("InstructionRa_{index}"));
            evals.push(Stage5SumcheckEvalPlan {
                symbol: name,
                source: driver_symbol,
                name,
                index: table_count + index,
                oracle,
            });
        }
        evals.push(Stage5SumcheckEvalPlan {
            symbol: "stage5.instruction_read_raf.eval.InstructionRafFlag",
            source: driver_symbol,
            name: "stage5.instruction_read_raf.eval.InstructionRafFlag",
            index: table_count + ra_chunks,
            oracle: "InstructionRafFlag",
        });

        Box::leak(Box::new(Stage5CpuProgramPlan {
            role: "prover",
            params: PARAMS,
            steps: Box::leak(
                vec![Stage5ProgramStepPlan {
                    kind: "sumcheck_driver",
                    symbol: driver_symbol,
                }]
                .into_boxed_slice(),
            ),
            transcript_squeezes: &[],
            transcript_absorb_bytes: &[],
            opening_inputs: Box::leak(
                vec![
                    Stage5OpeningInputPlan {
                        symbol: "stage5.input.stage2.instruction.LookupOutput",
                        source_stage: "stage2",
                        source_claim:
                            "stage2.instruction_lookup.claim_reduction.opening.LookupOutput",
                        oracle: "LookupOutput",
                        domain: "jolt.trace_domain",
                        point_arity: trace_rounds,
                        claim_kind: "virtual",
                    },
                    Stage5OpeningInputPlan {
                        symbol: "stage5.input.stage2.instruction.LeftLookupOperand",
                        source_stage: "stage2",
                        source_claim:
                            "stage2.instruction_lookup.claim_reduction.opening.LeftLookupOperand",
                        oracle: "LeftLookupOperand",
                        domain: "jolt.trace_domain",
                        point_arity: trace_rounds,
                        claim_kind: "virtual",
                    },
                    Stage5OpeningInputPlan {
                        symbol: "stage5.input.stage2.instruction.RightLookupOperand",
                        source_stage: "stage2",
                        source_claim:
                            "stage2.instruction_lookup.claim_reduction.opening.RightLookupOperand",
                        oracle: "RightLookupOperand",
                        domain: "jolt.trace_domain",
                        point_arity: trace_rounds,
                        claim_kind: "virtual",
                    },
                ]
                .into_boxed_slice(),
            ),
            field_constants: Box::leak(
                vec![Stage5FieldConstantPlan {
                    symbol: "stage5.instruction_read_raf.gamma",
                    field: "bn254_fr",
                    value: 2,
                }]
                .into_boxed_slice(),
            ),
            field_exprs: &[],
            kernels: Box::leak(
                vec![
                    Stage5KernelPlan {
                        symbol: "jolt.cpu.stage5.instruction_read_raf",
                        relation: "jolt.stage5.instruction_read_raf",
                        kind: "sumcheck",
                        backend: "cpu",
                        abi: "jolt_stage5_instruction_read_raf",
                    },
                    Stage5KernelPlan {
                        symbol: "jolt.cpu.stage5.batched",
                        relation: "jolt.stage5.batched",
                        kind: "sumcheck",
                        backend: "cpu",
                        abi: "jolt_stage5_batched",
                    },
                ]
                .into_boxed_slice(),
            ),
            claims: Box::leak(
                vec![Stage5SumcheckClaimPlan {
                    symbol: claim_symbol,
                    stage: "stage5",
                    domain: "jolt.stage5_instruction_read_raf_domain",
                    num_rounds: 128 + trace_rounds,
                    degree: ra_chunks + 2,
                    claim: "stage5.instruction_read_raf.weighted_lookup_values",
                    kernel: Some("jolt.cpu.stage5.instruction_read_raf"),
                    relation: Some("jolt.stage5.instruction_read_raf"),
                    claim_value: "stage5.instruction_read_raf.claim_expr",
                    input_openings: claim_inputs,
                }]
                .into_boxed_slice(),
            ),
            batches: Box::leak(
                vec![Stage5SumcheckBatchPlan {
                    symbol: "stage5.instruction_read_raf.batch",
                    stage: "stage5",
                    proof_slot: driver_symbol,
                    policy: "test",
                    count: 1,
                    ordered_claims,
                    claim_operands: ordered_claims,
                    claim_label: "sumcheck_claim",
                    round_label: "sumcheck_poly",
                    round_schedule,
                }]
                .into_boxed_slice(),
            ),
            drivers: Box::leak(
                vec![Stage5SumcheckDriverPlan {
                    symbol: driver_symbol,
                    stage: "stage5",
                    proof_slot: driver_symbol,
                    kernel: Some("jolt.cpu.stage5.batched"),
                    relation: Some("jolt.stage5.batched"),
                    batch: "stage5.instruction_read_raf.batch",
                    policy: "test",
                    round_schedule,
                    claim_label: "sumcheck_claim",
                    round_label: "sumcheck_poly",
                    num_rounds: 128 + trace_rounds,
                    degree: ra_chunks + 2,
                }]
                .into_boxed_slice(),
            ),
            instance_results: Box::leak(
                vec![Stage5SumcheckInstanceResultPlan {
                    symbol: "stage5.instruction_read_raf.instance",
                    source: driver_symbol,
                    claim: claim_symbol,
                    relation: "jolt.stage5.instruction_read_raf",
                    index: 0,
                    point_arity: 128 + trace_rounds,
                    num_rounds: 128 + trace_rounds,
                    round_offset: 0,
                    point_order: "instruction_read_raf",
                    degree: ra_chunks + 2,
                }]
                .into_boxed_slice(),
            ),
            evals: Box::leak(evals.into_boxed_slice()),
            point_slices: &[],
            point_concats: &[],
            opening_claims: &[],
            opening_equalities: &[],
            opening_batches: &[],
        }))
    }

    fn leak_test_str(value: String) -> &'static str {
        Box::leak(value.into_boxed_str())
    }

    fn named_eval_values(evals: &[Stage5NamedEval<Fr>]) -> Vec<(&'static str, &'static str, Fr)> {
        evals
            .iter()
            .map(|eval| (eval.name, eval.oracle, eval.value))
            .collect()
    }
}
