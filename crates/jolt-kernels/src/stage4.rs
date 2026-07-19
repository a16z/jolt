//! Stage 4 coarse-kernel ABI used by Bolt-generated Jolt prover code.

#![expect(
    clippy::large_enum_variant,
    reason = "kernel states stay inline to avoid boxing hot prover state"
)]
#![expect(
    clippy::too_many_arguments,
    reason = "kernel constructors mirror generated staged protocol inputs"
)]

#[cfg(feature = "cuda")]
mod cuda;

use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::mem::MaybeUninit;

use crate::dense::{bind_dense_evals_reuse, DENSE_BIND_PAR_THRESHOLD};
use crate::split_eq::SplitEqState;
use crate::stage2::Stage2RamAccess;
use jolt_field::{Field, FieldAccumulator};
#[cfg(feature = "cuda")]
use jolt_field::Fr;
use jolt_poly::{EqPolynomial, UnivariatePoly};
use jolt_sumcheck::SumcheckProof;
use jolt_transcript::{Label, LabelWithCount, Transcript};
use jolt_witness::{stage4_5_sparse_trace_witness, Stage45SparseTraceWitness};
use rayon::prelude::*;

fn trace_stage4_inner_spans() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("JOLT_STAGE4_TRACE_INSTANCES").is_some())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage4ExecutionMode {
    Prover,
    Verifier,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage4Relation {
    RegistersReadWrite,
    RamValCheck,
    Batched,
}

impl Stage4Relation {
    pub fn from_symbol(symbol: &str) -> Option<Self> {
        match symbol {
            "jolt.stage4.registers_read_write" => Some(Self::RegistersReadWrite),
            "jolt.stage4.ram_val_check" => Some(Self::RamValCheck),
            "jolt.stage4.batched" => Some(Self::Batched),
            _ => None,
        }
    }

    pub fn symbol(self) -> &'static str {
        match self {
            Self::RegistersReadWrite => "jolt.stage4.registers_read_write",
            Self::RamValCheck => "jolt.stage4.ram_val_check",
            Self::Batched => "jolt.stage4.batched",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage4KernelAbi {
    RegistersReadWrite,
    RamValCheck,
    Batched,
}

impl Stage4KernelAbi {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "jolt_stage4_registers_read_write" => Some(Self::RegistersReadWrite),
            "jolt_stage4_ram_val_check" => Some(Self::RamValCheck),
            "jolt_stage4_batched" => Some(Self::Batched),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::RegistersReadWrite => "jolt_stage4_registers_read_write",
            Self::RamValCheck => "jolt_stage4_ram_val_check",
            Self::Batched => "jolt_stage4_batched",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage4Params {
    pub field: &'static str,
    pub pcs: &'static str,
    pub transcript: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage4KernelPlan {
    pub symbol: &'static str,
    pub relation: &'static str,
    pub kind: &'static str,
    pub backend: &'static str,
    pub abi: &'static str,
}

impl Stage4KernelPlan {
    pub fn relation_kind(&self) -> Result<Stage4Relation, Stage4KernelError> {
        Stage4Relation::from_symbol(self.relation).ok_or(Stage4KernelError::UnknownRelation {
            relation: self.relation,
        })
    }

    pub fn abi_kind(&self) -> Result<Stage4KernelAbi, Stage4KernelError> {
        Stage4KernelAbi::from_name(self.abi)
            .ok_or(Stage4KernelError::UnknownKernelAbi { abi: self.abi })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage4TranscriptSqueezePlan {
    pub symbol: &'static str,
    pub label: &'static str,
    pub kind: &'static str,
    pub count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage4TranscriptAbsorbBytesPlan {
    pub symbol: &'static str,
    pub label: &'static str,
    pub payload: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage4ProgramStepPlan {
    pub kind: &'static str,
    pub symbol: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage4OpeningInputPlan {
    pub symbol: &'static str,
    pub source_stage: &'static str,
    pub source_claim: &'static str,
    pub oracle: &'static str,
    pub domain: &'static str,
    pub point_arity: usize,
    pub claim_kind: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage4FieldConstantPlan {
    pub symbol: &'static str,
    pub field: &'static str,
    pub value: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage4FieldExprPlan {
    pub symbol: &'static str,
    pub kind: &'static str,
    pub formula: &'static str,
    pub operand_names: &'static [&'static str],
    pub operands: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage4SumcheckClaimPlan {
    pub symbol: &'static str,
    pub stage: &'static str,
    pub domain: &'static str,
    pub num_rounds: usize,
    pub degree: usize,
    pub claim: &'static str,
    pub kernel: Option<&'static str>,
    pub relation: Option<&'static str>,
    pub claim_value: &'static str,
    pub input_openings: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage4SumcheckBatchPlan {
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
pub struct Stage4SumcheckDriverPlan {
    pub symbol: &'static str,
    pub stage: &'static str,
    pub proof_slot: &'static str,
    pub kernel: Option<&'static str>,
    pub relation: Option<&'static str>,
    pub batch: &'static str,
    pub policy: &'static str,
    pub round_schedule: &'static [usize],
    pub claim_label: &'static str,
    pub round_label: &'static str,
    pub num_rounds: usize,
    pub degree: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage4SumcheckInstanceResultPlan {
    pub symbol: &'static str,
    pub source: &'static str,
    pub claim: &'static str,
    pub relation: &'static str,
    pub index: usize,
    pub point_arity: usize,
    pub num_rounds: usize,
    pub round_offset: usize,
    pub point_order: &'static str,
    pub degree: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage4SumcheckEvalPlan {
    pub symbol: &'static str,
    pub source: &'static str,
    pub name: &'static str,
    pub index: usize,
    pub oracle: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage4PointSlicePlan {
    pub symbol: &'static str,
    pub source: &'static str,
    pub offset: usize,
    pub length: usize,
    pub input: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage4PointConcatPlan {
    pub symbol: &'static str,
    pub layout: &'static str,
    pub arity: usize,
    pub inputs: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage4OpeningClaimPlan {
    pub symbol: &'static str,
    pub oracle: &'static str,
    pub domain: &'static str,
    pub point_arity: usize,
    pub claim_kind: &'static str,
    pub point_source: &'static str,
    pub eval_source: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage4OpeningClaimEqualityPlan {
    pub symbol: &'static str,
    pub mode: &'static str,
    pub lhs: &'static str,
    pub rhs: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage4OpeningBatchPlan {
    pub symbol: &'static str,
    pub stage: &'static str,
    pub proof_slot: &'static str,
    pub policy: &'static str,
    pub count: usize,
    pub ordered_claims: &'static [&'static str],
    pub claim_operands: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage4CpuProgramPlan {
    pub role: &'static str,
    pub params: Stage4Params,
    pub steps: &'static [Stage4ProgramStepPlan],
    pub transcript_squeezes: &'static [Stage4TranscriptSqueezePlan],
    pub transcript_absorb_bytes: &'static [Stage4TranscriptAbsorbBytesPlan],
    pub opening_inputs: &'static [Stage4OpeningInputPlan],
    pub field_constants: &'static [Stage4FieldConstantPlan],
    pub field_exprs: &'static [Stage4FieldExprPlan],
    pub kernels: &'static [Stage4KernelPlan],
    pub claims: &'static [Stage4SumcheckClaimPlan],
    pub batches: &'static [Stage4SumcheckBatchPlan],
    pub drivers: &'static [Stage4SumcheckDriverPlan],
    pub instance_results: &'static [Stage4SumcheckInstanceResultPlan],
    pub evals: &'static [Stage4SumcheckEvalPlan],
    pub point_slices: &'static [Stage4PointSlicePlan],
    pub point_concats: &'static [Stage4PointConcatPlan],
    pub opening_claims: &'static [Stage4OpeningClaimPlan],
    pub opening_equalities: &'static [Stage4OpeningClaimEqualityPlan],
    pub opening_batches: &'static [Stage4OpeningBatchPlan],
}

impl Stage4CpuProgramPlan {
    pub fn claim(&self, symbol: &str) -> Option<&Stage4SumcheckClaimPlan> {
        self.claims.iter().find(|claim| claim.symbol == symbol)
    }

    pub fn instance_results_for_driver(
        &self,
        driver: &'static str,
    ) -> impl Iterator<Item = &Stage4SumcheckInstanceResultPlan> {
        self.instance_results
            .iter()
            .filter(move |instance| instance.source == driver)
    }

    pub fn evals_for_driver(
        &self,
        driver: &'static str,
    ) -> impl Iterator<Item = &Stage4SumcheckEvalPlan> {
        self.evals.iter().filter(move |eval| eval.source == driver)
    }
}

#[derive(Clone, Debug)]
pub struct Stage4NamedEval<F: Field> {
    pub name: &'static str,
    pub oracle: &'static str,
    pub value: F,
}

#[derive(Clone, Debug)]
pub struct Stage4SumcheckOutput<F: Field> {
    pub driver: &'static str,
    pub point: Vec<F>,
    pub evals: Vec<Stage4NamedEval<F>>,
    pub opening_claims: Vec<Stage4OpeningClaimValue<F>>,
    pub proof: SumcheckProof<F>,
}

#[derive(Clone, Debug)]
pub struct Stage4ChallengeVector<F: Field> {
    pub symbol: &'static str,
    pub values: Vec<F>,
}

#[derive(Clone, Debug)]
pub struct Stage4OpeningClaimValue<F: Field> {
    pub symbol: &'static str,
    pub oracle: &'static str,
    pub domain: &'static str,
    pub claim_kind: &'static str,
    pub point: Vec<F>,
    pub eval: F,
}

#[derive(Clone, Debug)]
pub struct Stage4ExecutionArtifacts<F: Field> {
    pub challenge_vectors: Vec<Stage4ChallengeVector<F>>,
    pub sumchecks: Vec<Stage4SumcheckOutput<F>>,
    pub opening_claims: Vec<Stage4OpeningClaimValue<F>>,
    pub opening_batches: Vec<&'static Stage4OpeningBatchPlan>,
}

impl<F: Field> Default for Stage4ExecutionArtifacts<F> {
    fn default() -> Self {
        Self {
            challenge_vectors: Vec::new(),
            sumchecks: Vec::new(),
            opening_claims: Vec::new(),
            opening_batches: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Stage4Proof<F: Field> {
    pub sumchecks: Vec<Stage4SumcheckOutput<F>>,
}

impl<F: Field> From<Stage4ExecutionArtifacts<F>> for Stage4Proof<F> {
    fn from(artifacts: Stage4ExecutionArtifacts<F>) -> Self {
        Self {
            sumchecks: artifacts.sumchecks,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Stage4OpeningInputValue<F: Field> {
    pub symbol: &'static str,
    pub point: Vec<F>,
    pub eval: F,
}

#[derive(Clone, Copy)]
pub struct Stage4RegistersWitness<'a, F: Field> {
    pub register_count: usize,
    pub trace_len: usize,
    pub registers_val: &'a [F],
    pub rs1_ra: &'a [F],
    pub rs2_ra: &'a [F],
    pub rd_wa: &'a [F],
    pub accesses: Option<&'a [Stage4RegisterAccess]>,
    pub rd_inc: &'a [F],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage4RegisterRead {
    pub address: usize,
    pub value: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage4RegisterWrite {
    pub address: usize,
    pub pre_value: u64,
    pub post_value: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stage4RegisterAccess {
    pub rs1: Option<Stage4RegisterRead>,
    pub rs2: Option<Stage4RegisterRead>,
    pub rd: Option<Stage4RegisterWrite>,
}

pub fn stage4_5_sparse_trace_witness_from_accesses<F: Field>(
    register_accesses: &[Stage4RegisterAccess],
    ram_accesses: &[Stage2RamAccess],
) -> Stage45SparseTraceWitness<F> {
    stage4_5_sparse_trace_witness(
        register_accesses.iter().map(|access| {
            access
                .rd
                .map(|rd| (rd.address, rd.pre_value, rd.post_value))
        }),
        ram_accesses.iter().map(|access| {
            (
                access.remapped_address,
                access.read_value,
                access.write_value,
            )
        }),
    )
}

#[derive(Clone, Copy)]
pub struct Stage4RamWitness<'a, F: Field> {
    pub ram_k: usize,
    pub trace_len: usize,
    pub ram_ra: &'a [F],
    pub write_address_indices: Option<&'a [Option<usize>]>,
    pub ram_inc: &'a [F],
}

#[derive(Clone, Copy)]
pub struct Stage4ProverInputs<'a, F: Field> {
    pub opening_inputs: &'a [Stage4OpeningInputValue<F>],
    pub registers: Option<Stage4RegistersWitness<'a, F>>,
    pub ram: Option<Stage4RamWitness<'a, F>>,
}

impl<'a, F: Field> Stage4ProverInputs<'a, F> {
    pub fn new(opening_inputs: &'a [Stage4OpeningInputValue<F>]) -> Self {
        Self {
            opening_inputs,
            registers: None,
            ram: None,
        }
    }

    pub fn empty() -> Self {
        Self {
            opening_inputs: &[],
            registers: None,
            ram: None,
        }
    }

    pub fn with_registers(mut self, registers: Stage4RegistersWitness<'a, F>) -> Self {
        self.registers = Some(registers);
        self
    }

    pub fn with_ram(mut self, ram: Stage4RamWitness<'a, F>) -> Self {
        self.ram = Some(ram);
        self
    }

    pub fn with_sparse_trace_witness(
        self,
        register_count: usize,
        trace_len: usize,
        ram_k: usize,
        register_accesses: &'a [Stage4RegisterAccess],
        rd_inc: &'a [F],
        write_address_indices: &'a [Option<usize>],
        ram_inc: &'a [F],
    ) -> Self {
        self.with_registers(Stage4RegistersWitness {
            register_count,
            trace_len,
            registers_val: &[],
            rs1_ra: &[],
            rs2_ra: &[],
            rd_wa: &[],
            accesses: Some(register_accesses),
            rd_inc,
        })
        .with_ram(Stage4RamWitness {
            ram_k,
            trace_len,
            ram_ra: &[],
            write_address_indices: Some(write_address_indices),
            ram_inc,
        })
    }

    pub fn with_stage45_sparse_trace_witness(
        self,
        register_count: usize,
        trace_len: usize,
        ram_k: usize,
        register_accesses: &'a [Stage4RegisterAccess],
        witness: &'a Stage45SparseTraceWitness<F>,
    ) -> Self {
        self.with_sparse_trace_witness(
            register_count,
            trace_len,
            ram_k,
            register_accesses,
            &witness.rd_inc,
            &witness.ram_addresses,
            &witness.ram_inc,
        )
    }
}

#[derive(Clone, Debug)]
pub struct Stage4ScalarValue<F: Field> {
    pub symbol: &'static str,
    pub value: F,
}

#[derive(Clone, Debug)]
pub struct Stage4PointValue<F: Field> {
    pub symbol: &'static str,
    pub point: Vec<F>,
}

#[derive(Clone, Debug, Default)]
pub struct Stage4ValueStore<F: Field> {
    scalars: Vec<Stage4ScalarValue<F>>,
    points: Vec<Stage4PointValue<F>>,
}

impl<F: Field> Stage4ValueStore<F> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_opening_inputs(inputs: &[Stage4OpeningInputValue<F>]) -> Self {
        let mut store = Self::new();
        store.insert_opening_inputs(inputs);
        store
    }

    pub fn insert_opening_inputs(&mut self, inputs: &[Stage4OpeningInputValue<F>]) {
        for input in inputs {
            self.insert_scalar(input.symbol, input.eval);
            self.insert_point(input.symbol, input.point.clone());
        }
    }

    pub fn insert_scalar(&mut self, symbol: &'static str, value: F) {
        if let Some(existing) = self
            .scalars
            .iter_mut()
            .find(|existing| existing.symbol == symbol)
        {
            existing.value = value;
        } else {
            self.scalars.push(Stage4ScalarValue { symbol, value });
        }
    }

    pub fn insert_point(&mut self, symbol: &'static str, point: Vec<F>) {
        if let Some(existing) = self
            .points
            .iter_mut()
            .find(|existing| existing.symbol == symbol)
        {
            existing.point = point;
        } else {
            self.points.push(Stage4PointValue { symbol, point });
        }
    }

    pub fn try_scalar(&self, symbol: &str) -> Option<F> {
        self.scalars
            .iter()
            .find(|value| value.symbol == symbol)
            .map(|value| value.value)
    }

    pub fn scalar(&self, symbol: &'static str) -> Result<F, Stage4KernelError> {
        self.try_scalar(symbol)
            .ok_or(Stage4KernelError::MissingValue { symbol })
    }

    pub fn try_point(&self, symbol: &str) -> Option<&[F]> {
        self.points
            .iter()
            .find(|value| value.symbol == symbol)
            .map(|value| value.point.as_slice())
    }

    pub fn point(&self, symbol: &'static str) -> Result<&[F], Stage4KernelError> {
        self.try_point(symbol)
            .ok_or(Stage4KernelError::MissingValue { symbol })
    }

    pub fn seed_constants(
        &mut self,
        program: &'static Stage4CpuProgramPlan,
    ) -> Result<(), Stage4KernelError> {
        for constant in program.field_constants {
            self.insert_scalar(constant.symbol, F::from_u64(constant.value as u64));
        }
        Ok(())
    }

    pub fn observe_challenge_vector(
        &mut self,
        plan: &'static Stage4TranscriptSqueezePlan,
        values: &[F],
    ) -> Result<(), Stage4KernelError> {
        if matches!(plan.kind, "challenge_scalar" | "scalar") {
            require_operand_count(plan.symbol, 1, values.len())?;
            self.insert_scalar(plan.symbol, values[0]);
        }
        self.insert_point(plan.symbol, values.to_vec());
        let _ = values;
        Ok(())
    }

    pub fn observe_sumcheck_output(
        &mut self,
        program: &'static Stage4CpuProgramPlan,
        output: &Stage4SumcheckOutput<F>,
    ) -> Result<(), Stage4KernelError> {
        self.observe_sumcheck_values(program, output.driver, &output.point, &output.evals)
    }

    pub fn observe_sumcheck_values(
        &mut self,
        program: &'static Stage4CpuProgramPlan,
        driver: &'static str,
        point: &[F],
        evals: &[Stage4NamedEval<F>],
    ) -> Result<(), Stage4KernelError> {
        self.insert_point(driver, point.to_vec());
        for instance in program.instance_results_for_driver(driver) {
            let end = instance.round_offset + instance.point_arity;
            let mut instance_point = point
                .get(instance.round_offset..end)
                .ok_or(Stage4KernelError::InvalidInputLength {
                    input: instance.symbol,
                    expected: end,
                    actual: point.len(),
                })?
                .to_vec();
            match instance.point_order {
                "as_is" => {}
                "reverse" => instance_point.reverse(),
                "stage4_registers_rw" => {
                    instance_point =
                        normalize_stage4_registers_rw_point(program, driver, &instance_point)?;
                }
                _ => {
                    return Err(Stage4KernelError::InvalidProof {
                        driver,
                        reason: "unsupported point order",
                    });
                }
            }
            self.insert_point(instance.symbol, instance_point);
        }
        for eval in program.evals_for_driver(driver) {
            let value = evals
                .iter()
                .find(|value| value.name == eval.name)
                .or_else(|| evals.get(eval.index))
                .ok_or(Stage4KernelError::MissingValue {
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

    pub fn evaluate_available_points(
        &mut self,
        program: &'static Stage4CpuProgramPlan,
    ) -> Result<usize, Stage4KernelError> {
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
                    .ok_or(Stage4KernelError::InvalidInputLength {
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
                verify_count(concat.symbol, concat.arity, point.len())?;
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
        program: &'static Stage4CpuProgramPlan,
    ) -> Result<usize, Stage4KernelError> {
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
                self.insert_scalar(expr.symbol, evaluate_stage4_field_expr(expr, &operands)?);
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
        program: &'static Stage4CpuProgramPlan,
    ) -> Result<(), Stage4KernelError> {
        for equality in program.opening_equalities {
            match equality.mode {
                "point_and_eval" => {
                    if self.point(equality.lhs)? != self.point(equality.rhs)?
                        || self.scalar(equality.lhs)? != self.scalar(equality.rhs)?
                    {
                        return Err(Stage4KernelError::InvalidProof {
                            driver: equality.symbol,
                            reason: "opening claim equality failed",
                        });
                    }
                }
                _ => {
                    return Err(Stage4KernelError::InvalidProof {
                        driver: equality.symbol,
                        reason: "unsupported opening equality mode",
                    });
                }
            }
        }
        Ok(())
    }

    pub fn claim_value(
        &mut self,
        program: &'static Stage4CpuProgramPlan,
        claim: &Stage4SumcheckClaimPlan,
    ) -> Result<F, Stage4KernelError> {
        let _ = self.evaluate_available_field_exprs(program)?;
        self.scalar(claim.claim_value)
    }

    pub fn batch_claim_values(
        &mut self,
        program: &'static Stage4CpuProgramPlan,
        batch: &Stage4SumcheckBatchPlan,
    ) -> Result<Vec<F>, Stage4KernelError> {
        batch
            .claim_operands
            .iter()
            .map(|symbol| {
                let claim = program
                    .claim(symbol)
                    .ok_or(Stage4KernelError::MissingClaim {
                        batch: batch.symbol,
                        claim: symbol,
                    })?;
                self.claim_value(program, claim)
            })
            .collect()
    }

    fn try_expr_operands(&self, expr: &Stage4FieldExprPlan) -> Option<Vec<F>> {
        expr.operands
            .iter()
            .map(|operand| self.try_scalar(operand))
            .collect()
    }

    fn try_concat_point(&self, concat: &Stage4PointConcatPlan) -> Option<Vec<F>> {
        let mut point = Vec::with_capacity(concat.arity);
        for input in concat.inputs {
            point.extend_from_slice(self.try_point(input)?);
        }
        Some(point)
    }
}

pub fn evaluate_stage4_field_expr<F: Field>(
    expr: &Stage4FieldExprPlan,
    operands: &[F],
) -> Result<F, Stage4KernelError> {
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
        _ => {
            if let Some(exponent) = expr.formula.strip_prefix("field.pow:") {
                require_operand_count(expr.symbol, 1, operands.len())?;
                let exponent = exponent.parse::<usize>().map_err(|_| {
                    Stage4KernelError::UnsupportedFieldExpr {
                        symbol: expr.symbol,
                        formula: expr.formula,
                    }
                })?;
                return Ok(pow_field(operands[0], exponent));
            }
            Err(Stage4KernelError::UnsupportedFieldExpr {
                symbol: expr.symbol,
                formula: expr.formula,
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

fn single_operand<F: Field>(symbol: &'static str, operands: &[F]) -> Result<F, Stage4KernelError> {
    require_operand_count(symbol, 1, operands.len())?;
    Ok(operands[0])
}

fn require_operand_count(
    input: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), Stage4KernelError> {
    if expected == actual {
        Ok(())
    } else {
        Err(Stage4KernelError::InvalidInputLength {
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
) -> Result<(), Stage4KernelError> {
    #[cfg(debug_assertions)]
    {
        if poly.evaluate(F::zero()) + poly.evaluate(F::one()) != previous_claim {
            return Err(Stage4KernelError::InvalidProof { driver, reason });
        }
    }
    #[cfg(not(debug_assertions))]
    {
        let _ = (poly, previous_claim, driver, reason);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
pub struct Stage4KernelContext<'a> {
    pub mode: Stage4ExecutionMode,
    pub program: &'static Stage4CpuProgramPlan,
    pub kernel: &'a Stage4KernelPlan,
    pub batch: &'a Stage4SumcheckBatchPlan,
    pub driver: &'a Stage4SumcheckDriverPlan,
}

impl Stage4KernelContext<'_> {
    pub fn relation_kind(&self) -> Result<Stage4Relation, Stage4KernelError> {
        self.kernel.relation_kind()
    }

    pub fn abi_kind(&self) -> Result<Stage4KernelAbi, Stage4KernelError> {
        self.kernel.abi_kind()
    }

    pub fn batch_claims(&self) -> Result<Vec<&'static Stage4SumcheckClaimPlan>, Stage4KernelError> {
        self.batch
            .claim_operands
            .iter()
            .map(|symbol| {
                self.program
                    .claim(symbol)
                    .ok_or(Stage4KernelError::MissingClaim {
                        batch: self.batch.symbol,
                        claim: symbol,
                    })
            })
            .collect()
    }
}

pub trait Stage4KernelExecutor<F: Field> {
    fn observe_challenge_vector(
        &mut self,
        _plan: &'static Stage4TranscriptSqueezePlan,
        _values: &[F],
    ) -> Result<(), Stage4KernelError> {
        Ok(())
    }

    fn observe_sumcheck_output(
        &mut self,
        _output: &Stage4SumcheckOutput<F>,
    ) -> Result<(), Stage4KernelError> {
        Ok(())
    }

    fn prove_sumcheck<T>(
        &mut self,
        context: Stage4KernelContext<'_>,
        transcript: &mut T,
    ) -> Result<Stage4SumcheckOutput<F>, Stage4KernelError>
    where
        T: Transcript<Challenge = F>;

    fn verify_sumcheck<T>(
        &mut self,
        context: Stage4KernelContext<'_>,
        transcript: &mut T,
    ) -> Result<Stage4SumcheckOutput<F>, Stage4KernelError>
    where
        T: Transcript<Challenge = F>;
}

#[derive(Clone, Debug, Default)]
pub struct UnsupportedStage4KernelExecutor;

impl<F: Field> Stage4KernelExecutor<F> for UnsupportedStage4KernelExecutor {
    fn prove_sumcheck<T>(
        &mut self,
        context: Stage4KernelContext<'_>,
        _transcript: &mut T,
    ) -> Result<Stage4SumcheckOutput<F>, Stage4KernelError>
    where
        T: Transcript<Challenge = F>,
    {
        Err(Stage4KernelError::KernelNotImplemented {
            abi: context.kernel.abi,
        })
    }

    fn verify_sumcheck<T>(
        &mut self,
        context: Stage4KernelContext<'_>,
        _transcript: &mut T,
    ) -> Result<Stage4SumcheckOutput<F>, Stage4KernelError>
    where
        T: Transcript<Challenge = F>,
    {
        Err(Stage4KernelError::KernelNotImplemented {
            abi: context.kernel.abi,
        })
    }
}

#[derive(Clone)]
pub struct Stage4ProverKernelExecutor<'a, F: Field> {
    pub inputs: Stage4ProverInputs<'a, F>,
    challenge_vectors: Vec<Stage4ChallengeVector<F>>,
    completed_sumchecks: Vec<Stage4SumcheckOutput<F>>,
}

impl<'a, F: Field> Stage4ProverKernelExecutor<'a, F> {
    pub fn new(inputs: Stage4ProverInputs<'a, F>) -> Self {
        Self {
            inputs,
            challenge_vectors: Vec::new(),
            completed_sumchecks: Vec::new(),
        }
    }

    fn value_store(
        &self,
        program: &'static Stage4CpuProgramPlan,
    ) -> Result<Stage4ValueStore<F>, Stage4KernelError> {
        value_store_from_observations(
            program,
            self.inputs.opening_inputs,
            &self.challenge_vectors,
            &self.completed_sumchecks,
        )
    }
}

impl<F: Field> Stage4KernelExecutor<F> for Stage4ProverKernelExecutor<'_, F> {
    fn observe_challenge_vector(
        &mut self,
        plan: &'static Stage4TranscriptSqueezePlan,
        values: &[F],
    ) -> Result<(), Stage4KernelError> {
        self.challenge_vectors.push(Stage4ChallengeVector {
            symbol: plan.symbol,
            values: values.to_vec(),
        });
        Ok(())
    }

    fn observe_sumcheck_output(
        &mut self,
        output: &Stage4SumcheckOutput<F>,
    ) -> Result<(), Stage4KernelError> {
        self.completed_sumchecks.push(output.clone());
        Ok(())
    }

    fn prove_sumcheck<T>(
        &mut self,
        context: Stage4KernelContext<'_>,
        transcript: &mut T,
    ) -> Result<Stage4SumcheckOutput<F>, Stage4KernelError>
    where
        T: Transcript<Challenge = F>,
    {
        prove_stage4_kernel(
            context,
            &self.inputs,
            self.value_store(context.program)?,
            transcript,
        )
    }

    fn verify_sumcheck<T>(
        &mut self,
        context: Stage4KernelContext<'_>,
        _transcript: &mut T,
    ) -> Result<Stage4SumcheckOutput<F>, Stage4KernelError>
    where
        T: Transcript<Challenge = F>,
    {
        Err(Stage4KernelError::WrongExecutorMode {
            driver: context.driver.symbol,
            expected: Stage4ExecutionMode::Prover,
            actual: Stage4ExecutionMode::Verifier,
        })
    }
}

#[derive(Clone)]
pub struct Stage4VerifierKernelExecutor<'a, F: Field> {
    pub proof: &'a Stage4Proof<F>,
    pub opening_inputs: &'a [Stage4OpeningInputValue<F>],
    pub cursor: usize,
    challenge_vectors: Vec<Stage4ChallengeVector<F>>,
    completed_sumchecks: Vec<Stage4SumcheckOutput<F>>,
}

impl<'a, F: Field> Stage4VerifierKernelExecutor<'a, F> {
    pub fn new(
        proof: &'a Stage4Proof<F>,
        opening_inputs: &'a [Stage4OpeningInputValue<F>],
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
        program: &'static Stage4CpuProgramPlan,
    ) -> Result<Stage4ValueStore<F>, Stage4KernelError> {
        value_store_from_observations(
            program,
            self.opening_inputs,
            &self.challenge_vectors,
            &self.completed_sumchecks,
        )
    }
}

impl<F: Field> Stage4KernelExecutor<F> for Stage4VerifierKernelExecutor<'_, F> {
    fn observe_challenge_vector(
        &mut self,
        plan: &'static Stage4TranscriptSqueezePlan,
        values: &[F],
    ) -> Result<(), Stage4KernelError> {
        self.challenge_vectors.push(Stage4ChallengeVector {
            symbol: plan.symbol,
            values: values.to_vec(),
        });
        Ok(())
    }

    fn observe_sumcheck_output(
        &mut self,
        output: &Stage4SumcheckOutput<F>,
    ) -> Result<(), Stage4KernelError> {
        self.completed_sumchecks.push(output.clone());
        Ok(())
    }

    fn prove_sumcheck<T>(
        &mut self,
        context: Stage4KernelContext<'_>,
        _transcript: &mut T,
    ) -> Result<Stage4SumcheckOutput<F>, Stage4KernelError>
    where
        T: Transcript<Challenge = F>,
    {
        Err(Stage4KernelError::WrongExecutorMode {
            driver: context.driver.symbol,
            expected: Stage4ExecutionMode::Verifier,
            actual: Stage4ExecutionMode::Prover,
        })
    }

    fn verify_sumcheck<T>(
        &mut self,
        context: Stage4KernelContext<'_>,
        transcript: &mut T,
    ) -> Result<Stage4SumcheckOutput<F>, Stage4KernelError>
    where
        T: Transcript<Challenge = F>,
    {
        let proof =
            self.proof
                .sumchecks
                .get(self.cursor)
                .ok_or(Stage4KernelError::MissingProof {
                    driver: context.driver.symbol,
                })?;
        self.cursor += 1;
        verify_stage4_kernel(
            context,
            self.value_store(context.program)?,
            proof,
            transcript,
        )
    }
}

fn value_store_from_observations<F: Field>(
    program: &'static Stage4CpuProgramPlan,
    opening_inputs: &[Stage4OpeningInputValue<F>],
    challenge_vectors: &[Stage4ChallengeVector<F>],
    completed_sumchecks: &[Stage4SumcheckOutput<F>],
) -> Result<Stage4ValueStore<F>, Stage4KernelError> {
    let mut store = Stage4ValueStore::with_opening_inputs(opening_inputs);
    store.seed_constants(program)?;
    for challenge in challenge_vectors {
        let plan = program
            .transcript_squeezes
            .iter()
            .find(|plan| plan.symbol == challenge.symbol)
            .ok_or(Stage4KernelError::MissingValue {
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

pub fn execute_stage4_program<F, T, E>(
    program: &'static Stage4CpuProgramPlan,
    mode: Stage4ExecutionMode,
    executor: &mut E,
    transcript: &mut T,
) -> Result<Stage4ExecutionArtifacts<F>, Stage4KernelError>
where
    F: Field,
    T: Transcript<Challenge = F>,
    E: Stage4KernelExecutor<F>,
{
    let mut artifacts = Stage4ExecutionArtifacts::default();
    for step in program.steps {
        match step.kind {
            "transcript_squeeze" => {
                let squeeze =
                    find_squeeze(program, step.symbol).ok_or(Stage4KernelError::MissingValue {
                        symbol: step.symbol,
                    })?;
                let values = transcript.challenge_vector(squeeze.count);
                executor.observe_challenge_vector(squeeze, &values)?;
                artifacts.challenge_vectors.push(Stage4ChallengeVector {
                    symbol: squeeze.symbol,
                    values,
                });
            }
            "transcript_absorb_bytes" => {
                let absorb = find_absorb_bytes(program, step.symbol).ok_or(
                    Stage4KernelError::MissingValue {
                        symbol: step.symbol,
                    },
                )?;
                absorb_stage4_bytes(absorb, transcript);
            }
            "sumcheck_driver" => {
                let driver =
                    find_driver(program, step.symbol).ok_or(Stage4KernelError::MissingDriver {
                        driver: step.symbol,
                    })?;
                let kernel_symbol = driver.kernel.ok_or(Stage4KernelError::MissingKernel {
                    driver: driver.symbol,
                    kernel: "<missing>",
                })?;
                let kernel = find_kernel(program, kernel_symbol).ok_or(
                    Stage4KernelError::MissingKernel {
                        driver: driver.symbol,
                        kernel: kernel_symbol,
                    },
                )?;
                let batch =
                    find_batch(program, driver.batch).ok_or(Stage4KernelError::MissingBatch {
                        driver: driver.symbol,
                        batch: driver.batch,
                    })?;
                let context = Stage4KernelContext {
                    mode,
                    program,
                    kernel,
                    batch,
                    driver,
                };
                let output = match mode {
                    Stage4ExecutionMode::Prover => executor.prove_sumcheck(context, transcript)?,
                    Stage4ExecutionMode::Verifier => {
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
                return Err(Stage4KernelError::InvalidProgramStep {
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

fn absorb_stage4_bytes<T>(absorb: &'static Stage4TranscriptAbsorbBytesPlan, transcript: &mut T)
where
    T: Transcript,
{
    transcript.append(&LabelWithCount(
        absorb.label.as_bytes(),
        absorb.payload.len() as u64,
    ));
    transcript.append_bytes(absorb.payload.as_bytes());
}

fn prove_stage4_kernel<F, T>(
    context: Stage4KernelContext<'_>,
    inputs: &Stage4ProverInputs<'_, F>,
    store: Stage4ValueStore<F>,
    transcript: &mut T,
) -> Result<Stage4SumcheckOutput<F>, Stage4KernelError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    match context.abi_kind()? {
        Stage4KernelAbi::Batched => prove_batched_stage4(context, inputs, store, transcript),
        abi => Err(Stage4KernelError::KernelNotImplemented { abi: abi.name() }),
    }
}

fn verify_stage4_kernel<F, T>(
    context: Stage4KernelContext<'_>,
    store: Stage4ValueStore<F>,
    proof: &Stage4SumcheckOutput<F>,
    transcript: &mut T,
) -> Result<Stage4SumcheckOutput<F>, Stage4KernelError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    match context.abi_kind()? {
        Stage4KernelAbi::Batched => verify_batched_stage4(context, store, proof, transcript),
        abi => Err(Stage4KernelError::KernelNotImplemented { abi: abi.name() }),
    }
}

#[tracing::instrument(skip_all, name = "Stage4::prove_batched")]
fn prove_batched_stage4<F, T>(
    context: Stage4KernelContext<'_>,
    inputs: &Stage4ProverInputs<'_, F>,
    mut store: Stage4ValueStore<F>,
    transcript: &mut T,
) -> Result<Stage4SumcheckOutput<F>, Stage4KernelError>
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
        .ok_or(Stage4KernelError::InvalidProof {
            driver: context.driver.symbol,
            reason: "field element 2 is not invertible",
        })?;
    let mut instances = Vec::with_capacity(claims.len());
    for (index, claim) in claims.iter().enumerate() {
        let offset = instance_round_offset(context.program, context.driver.symbol, claim.symbol)?;
        let relation = claim_relation(context.program, claim)?;
        if offset + claim.num_rounds > max_rounds {
            return Err(Stage4KernelError::InvalidInputLength {
                input: claim.symbol,
                expected: max_rounds,
                actual: offset + claim.num_rounds,
            });
        }
        let active_scale = F::one().mul_pow_2(max_rounds - offset - claim.num_rounds);
        let _span = trace_stage4_inner_spans().then(|| {
            tracing::info_span!(
                "Stage4::instance.init",
                relation = relation.symbol(),
                claim = claim.symbol
            )
            .entered()
        });
        instances.push(Stage4BatchedInstance {
            claim,
            relation,
            offset,
            previous_claim: input_claims[index].mul_pow_2(max_rounds - claim.num_rounds),
            state: Stage4ProverInstanceState::new(
                context.program,
                claim,
                inputs,
                &store,
                active_scale,
                context.kernel.backend,
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
                let _span = trace_stage4_inner_spans().then(|| {
                    tracing::info_span!(
                        "Stage4::instance.round_poly",
                        relation = instance.relation.symbol(),
                        claim = instance.claim.symbol,
                        round
                    )
                    .entered()
                });
                instance
                    .state
                    .round_poly(instance.previous_claim, instance.relation)?
            } else {
                UnivariatePoly::new(vec![instance.previous_claim * two_inv])
            };
            #[cfg(debug_assertions)]
            {
                if poly.evaluate(F::zero()) + poly.evaluate(F::one()) != instance.previous_claim {
                    return Err(Stage4KernelError::InvalidProof {
                        driver: context.driver.symbol,
                        reason: "batched instance round claim mismatch",
                    });
                }
            }
            individual_polys.push(poly);
        }
        let batched_poly = combine_univariate_polys(&individual_polys, &batching_coeffs);
        #[cfg(debug_assertions)]
        {
            if batched_poly.evaluate(F::zero()) + batched_poly.evaluate(F::one()) != batched_claim {
                return Err(Stage4KernelError::InvalidProof {
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
                let _span = trace_stage4_inner_spans().then(|| {
                    tracing::info_span!(
                        "Stage4::instance.bind",
                        relation = instance.relation.symbol(),
                        claim = instance.claim.symbol,
                        round
                    )
                    .entered()
                });
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
        return Err(Stage4KernelError::InvalidProof {
            driver: context.driver.symbol,
            reason: "batched output claim mismatch",
        });
    }
    store.observe_sumcheck_values(context.program, context.driver.symbol, &point, &evals)?;
    let opening_claims = append_opening_claims(context.program, &mut store, transcript, &evals)?;
    Ok(Stage4SumcheckOutput {
        driver: context.driver.symbol,
        point,
        evals,
        opening_claims,
        proof: SumcheckProof { round_polynomials },
    })
}

fn verify_batched_stage4<F, T>(
    context: Stage4KernelContext<'_>,
    mut store: Stage4ValueStore<F>,
    proof: &Stage4SumcheckOutput<F>,
    transcript: &mut T,
) -> Result<Stage4SumcheckOutput<F>, Stage4KernelError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    if proof.driver != context.driver.symbol {
        return Err(Stage4KernelError::InvalidProof {
            driver: context.driver.symbol,
            reason: "driver symbol mismatch",
        });
    }
    if proof.proof.round_polynomials.len() != context.driver.num_rounds {
        return Err(Stage4KernelError::InvalidProof {
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
            return Err(Stage4KernelError::InvalidProof {
                driver: context.driver.symbol,
                reason: "batched polynomial exceeds degree bound",
            });
        }
        if poly.evaluate(F::zero()) + poly.evaluate(F::one()) != running_claim {
            return Err(Stage4KernelError::InvalidProof {
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
        return Err(Stage4KernelError::InvalidProof {
            driver: context.driver.symbol,
            reason: "batched point mismatch",
        });
    }
    let expected =
        expected_batched_output_claim(context, &store, &proof.evals, &point, &batching_coeffs)?;
    if running_claim != expected {
        return Err(Stage4KernelError::InvalidProof {
            driver: context.driver.symbol,
            reason: "batched output claim mismatch",
        });
    }
    let output = Stage4SumcheckOutput {
        driver: context.driver.symbol,
        point,
        evals: proof.evals.clone(),
        opening_claims: Vec::new(),
        proof: proof.proof.clone(),
    };
    store.observe_sumcheck_output(context.program, &output)?;
    let opening_claims =
        append_opening_claims(context.program, &mut store, transcript, &output.evals)?;
    let output = Stage4SumcheckOutput {
        opening_claims,
        ..output
    };
    Ok(output)
}

struct Stage4BatchedInstance<'a, F: Field> {
    claim: &'a Stage4SumcheckClaimPlan,
    relation: Stage4Relation,
    offset: usize,
    previous_claim: F,
    state: Stage4ProverInstanceState<F>,
}

impl<F: Field> Stage4BatchedInstance<'_, F> {
    fn is_active(&self, round: usize) -> bool {
        round >= self.offset && round < self.offset + self.claim.num_rounds
    }
}

enum Stage4ProverInstanceState<F: Field> {
    Dense(DenseStage4State<F>),
    SparseRegisters(SparseRegistersState<F>),
}

impl<F: Field> Stage4ProverInstanceState<F> {
    fn new(
        program: &'static Stage4CpuProgramPlan,
        claim: &Stage4SumcheckClaimPlan,
        inputs: &Stage4ProverInputs<'_, F>,
        store: &Stage4ValueStore<F>,
        active_scale: F,
        backend: &'static str,
    ) -> Result<Self, Stage4KernelError> {
        match claim_relation(program, claim)? {
            Stage4Relation::RegistersReadWrite => {
                registers_read_write_state(claim, inputs, store, active_scale, backend)
            }
            Stage4Relation::RamValCheck => {
                ram_val_check_state(claim, inputs, store, active_scale, backend).map(Self::Dense)
            }
            relation @ Stage4Relation::Batched => Err(Stage4KernelError::KernelNotImplemented {
                abi: relation.symbol(),
            }),
        }
    }

    fn round_poly(
        &mut self,
        previous_claim: F,
        relation: Stage4Relation,
    ) -> Result<UnivariatePoly<F>, Stage4KernelError> {
        match self {
            Self::Dense(state) => state.round_poly(previous_claim, relation),
            Self::SparseRegisters(state) => state.round_poly(previous_claim, relation),
        }
    }

    fn ingest_challenge(&mut self, challenge: F) {
        match self {
            Self::Dense(state) => state.bind(challenge),
            Self::SparseRegisters(state) => state.bind(challenge),
        }
    }

    fn final_evals(
        &self,
        relation: Stage4Relation,
    ) -> Result<Vec<Stage4NamedEval<F>>, Stage4KernelError> {
        match self {
            Self::Dense(state) => state.final_evals(relation),
            Self::SparseRegisters(state) => state.final_evals(relation),
        }
    }
}

struct DenseStage4State<F: Field> {
    factors: Vec<Vec<F>>,
    factor_scratch: Vec<Vec<F>>,
    terms: Vec<DenseTerm<F>>,
    outputs: Vec<FactorOutput>,
    active_scale: F,
    #[cfg(feature = "cuda")]
    cuda: Option<cuda::CudaDenseState>,
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

#[cfg(feature = "cuda")]
fn fr_poly_into<F: Field>(poly: UnivariatePoly<Fr>) -> Option<UnivariatePoly<F>> {
    (Box::new(poly) as Box<dyn std::any::Any>)
        .downcast::<UnivariatePoly<F>>()
        .ok()
        .map(|boxed| *boxed)
}

#[cfg(feature = "cuda")]
fn build_cuda_sparse_registers<F: Field>(
    entries: &[SparseRegisterEntry<F>],
    rd_inc: &[F],
    trace_point: &[F],
    trace_rounds: usize,
) -> Option<cuda::CudaSparseRegistersState> {
    let rows: Vec<usize> = entries.par_iter().map(|entry| entry.row).collect();
    let cols: Vec<u8> = entries.par_iter().map(|entry| entry.col).collect();
    let val: Vec<F> = entries.par_iter().map(|entry| entry.val).collect();
    let read_ra: Vec<F> = entries.par_iter().map(|entry| entry.read_ra).collect();
    let rd_wa: Vec<F> = entries.par_iter().map(|entry| entry.rd_wa).collect();
    let prev_val: Vec<u64> = entries.par_iter().map(|entry| entry.prev_val).collect();
    let next_val: Vec<u64> = entries.par_iter().map(|entry| entry.next_val).collect();
    cuda::CudaSparseRegistersState::new(
        &rows, &cols, &val, &read_ra, &rd_wa, &prev_val, &next_val, rd_inc, trace_point,
        trace_rounds,
    )
}

#[cfg(feature = "cuda")]
struct SparseRegisterRawColumns {
    rows: Vec<usize>,
    cols: Vec<u8>,
    prev_val: Vec<u64>,
    next_val: Vec<u64>,
    rs1_flag: Vec<u8>,
    rs2_flag: Vec<u8>,
    rd_flag: Vec<u8>,
}

#[cfg(feature = "cuda")]
fn sparse_register_raw_columns(
    register_count: usize,
    accesses: &[Stage4RegisterAccess],
) -> Result<SparseRegisterRawColumns, Stage4KernelError> {
    let mut out = SparseRegisterRawColumns {
        rows: Vec::with_capacity(accesses.len() * 3),
        cols: Vec::with_capacity(accesses.len() * 3),
        prev_val: Vec::with_capacity(accesses.len() * 3),
        next_val: Vec::with_capacity(accesses.len() * 3),
        rs1_flag: Vec::with_capacity(accesses.len() * 3),
        rs2_flag: Vec::with_capacity(accesses.len() * 3),
        rd_flag: Vec::with_capacity(accesses.len() * 3),
    };
    for (row, access) in accesses.iter().enumerate() {
        let start = out.cols.len();
        let find = |cols: &[u8], col: u8| cols[start..].iter().position(|&c| c == col).map(|p| start + p);

        if let Some(rs1) = access.rs1 {
            validate_register_address(register_count, rs1.address)?;
            let col = sparse_register_col(rs1.address)?;
            out.rows.push(row);
            out.cols.push(col);
            out.prev_val.push(rs1.value);
            out.next_val.push(rs1.value);
            out.rs1_flag.push(1);
            out.rs2_flag.push(0);
            out.rd_flag.push(0);
        }
        if let Some(rs2) = access.rs2 {
            validate_register_address(register_count, rs2.address)?;
            let col = sparse_register_col(rs2.address)?;
            if let Some(idx) = find(&out.cols, col) {
                out.rs2_flag[idx] = 1;
            } else {
                out.rows.push(row);
                out.cols.push(col);
                out.prev_val.push(rs2.value);
                out.next_val.push(rs2.value);
                out.rs1_flag.push(0);
                out.rs2_flag.push(1);
                out.rd_flag.push(0);
            }
        }
        if let Some(rd) = access.rd {
            validate_register_address(register_count, rd.address)?;
            let col = sparse_register_col(rd.address)?;
            if let Some(idx) = find(&out.cols, col) {
                out.rd_flag[idx] = 1;
                out.next_val[idx] = rd.post_value;
            } else {
                out.rows.push(row);
                out.cols.push(col);
                out.prev_val.push(rd.pre_value);
                out.next_val.push(rd.post_value);
                out.rs1_flag.push(0);
                out.rs2_flag.push(0);
                out.rd_flag.push(1);
            }
        }

        let mut order: Vec<usize> = (start..out.cols.len()).collect();
        order.sort_by_key(|&i| out.cols[i]);
        apply_permutation(&mut out.rows, start, &order);
        apply_permutation(&mut out.cols, start, &order);
        apply_permutation(&mut out.prev_val, start, &order);
        apply_permutation(&mut out.next_val, start, &order);
        apply_permutation(&mut out.rs1_flag, start, &order);
        apply_permutation(&mut out.rs2_flag, start, &order);
        apply_permutation(&mut out.rd_flag, start, &order);
    }
    Ok(out)
}

#[cfg(feature = "cuda")]
fn apply_permutation<T: Copy>(values: &mut [T], start: usize, order: &[usize]) {
    let reordered: Vec<T> = order.iter().map(|&i| values[i]).collect();
    values[start..].copy_from_slice(&reordered);
}

#[cfg(feature = "cuda")]
fn cuda_sparse_registers_from_raw<F: Field>(
    register_count: usize,
    accesses: &[Stage4RegisterAccess],
    rd_inc: &[F],
    trace_point: &[F],
    gamma: F,
    gamma2: F,
    trace_rounds: usize,
) -> Result<Option<cuda::CudaSparseRegistersState>, Stage4KernelError> {
    let raw = sparse_register_raw_columns(register_count, accesses)?;
    Ok(cuda::CudaSparseRegistersState::new_from_raw(
        &raw.rows,
        &raw.cols,
        &raw.prev_val,
        &raw.next_val,
        &raw.rs1_flag,
        &raw.rs2_flag,
        &raw.rd_flag,
        rd_inc,
        trace_point,
        gamma,
        gamma2,
        trace_rounds,
    ))
}

#[cfg(feature = "cuda")]
fn build_cuda_dense_state<F: Field>(
    factors: &[Vec<F>],
    terms: &[DenseTerm<F>],
    active_scale: F,
) -> Option<cuda::CudaDenseState> {
    let degree = terms.iter().map(|term| term.factors.len()).max()?;
    if degree == 0 {
        return None;
    }
    let fr_factors: Vec<&[Fr]> = factors
        .iter()
        .map(|factor| crate::cuda::as_fr_slice(factor))
        .collect::<Option<_>>()?;
    let mut term_coeffs = Vec::with_capacity(terms.len());
    let mut term_factor_offsets = vec![0u32];
    let mut term_factor_indices = Vec::new();
    for term in terms {
        term_coeffs.push(crate::cuda::into_fr(term.coefficient)?);
        for &factor in &term.factors {
            term_factor_indices.push(factor as u32);
        }
        term_factor_offsets.push(term_factor_indices.len() as u32);
    }
    let active_scale = crate::cuda::into_fr(active_scale)?;
    cuda::CudaDenseState::new(
        &fr_factors,
        term_coeffs,
        term_factor_offsets,
        term_factor_indices,
        degree,
        active_scale,
    )
}

#[cfg(feature = "cuda")]
fn build_cuda_ram_val_check_state<F: Field>(
    cycle_point: &[F],
    gamma: F,
    ram_ra_at_address: &[F],
    ram_inc: &[F],
    active_scale: F,
) -> Option<cuda::CudaDenseState> {
    let ctx = crate::cuda::shared_ctx()?;
    let cycle_point_fr = crate::cuda::as_fr_slice(cycle_point)?;
    let mut lt_plus_gamma = ctx.lt_evals(cycle_point_fr).ok()?;
    ctx.add_scalar(&mut lt_plus_gamma, crate::cuda::into_fr(gamma)?).ok()?;
    let ram_ra_dev = ctx.upload(crate::cuda::as_fr_slice(ram_ra_at_address)?).ok()?;
    let ram_inc_dev = ctx.resident_committed_clone(crate::cuda::as_fr_slice(ram_inc)?).ok()?;
    cuda::CudaDenseState::from_device_factors(
        vec![lt_plus_gamma, ram_ra_dev, ram_inc_dev],
        vec![crate::cuda::into_fr(F::one())?],
        vec![0, 3],
        vec![0, 1, 2],
        3,
        crate::cuda::into_fr(active_scale)?,
    )
}

#[cfg(feature = "cuda")]
fn cuda_ram_val_check_from_resident_addr<F: Field>(
    address_point: &[F],
    cycle_point: &[F],
    gamma: F,
    ram_inc: &[F],
    active_scale: F,
    trace_len: usize,
) -> Option<cuda::CudaDenseState> {
    let ctx = crate::cuda::shared_ctx()?;
    let resident = ctx.resident_ram_addresses().filter(|s| s.len == trace_len)?;
    let address_eq = ctx.eq_evals(crate::cuda::as_fr_slice(address_point)?, None).ok()?;
    let ram_ra_dev = ctx.ram_ra_gather(&address_eq, &resident.addr, trace_len).ok()?;
    let cycle_point_fr = crate::cuda::as_fr_slice(cycle_point)?;
    let mut lt_plus_gamma = ctx.lt_evals(cycle_point_fr).ok()?;
    ctx.add_scalar(&mut lt_plus_gamma, crate::cuda::into_fr(gamma)?).ok()?;
    let ram_inc_dev = ctx.resident_committed_clone(crate::cuda::as_fr_slice(ram_inc)?).ok()?;
    cuda::CudaDenseState::from_device_factors(
        vec![lt_plus_gamma, ram_ra_dev, ram_inc_dev],
        vec![crate::cuda::into_fr(F::one())?],
        vec![0, 3],
        vec![0, 1, 2],
        3,
        crate::cuda::into_fr(active_scale)?,
    )
}

impl<F: Field> DenseStage4State<F> {
    fn new(
        factors: Vec<Vec<F>>,
        terms: Vec<DenseTerm<F>>,
        outputs: Vec<FactorOutput>,
        active_scale: F,
    ) -> Self {
        Self::new_with_backend(factors, terms, outputs, active_scale, "cpu")
    }

    fn new_with_backend(
        factors: Vec<Vec<F>>,
        terms: Vec<DenseTerm<F>>,
        outputs: Vec<FactorOutput>,
        active_scale: F,
        backend: &'static str,
    ) -> Self {
        let factor_scratch = (0..factors.len()).map(|_| Vec::new()).collect();
        #[cfg(feature = "cuda")]
        let cuda = if backend == "cuda" {
            build_cuda_dense_state(&factors, &terms, active_scale)
        } else {
            None
        };
        #[cfg(not(feature = "cuda"))]
        let _ = backend;
        Self {
            factors,
            factor_scratch,
            terms,
            outputs,
            active_scale,
            #[cfg(feature = "cuda")]
            cuda,
        }
    }

    #[cfg(feature = "cuda")]
    fn from_host_and_cuda(
        factors: Vec<Vec<F>>,
        terms: Vec<DenseTerm<F>>,
        outputs: Vec<FactorOutput>,
        active_scale: F,
        cuda: Option<cuda::CudaDenseState>,
    ) -> Self {
        let factor_scratch = (0..factors.len()).map(|_| Vec::new()).collect();
        Self {
            factors,
            factor_scratch,
            terms,
            outputs,
            active_scale,
            cuda,
        }
    }

    fn round_poly(
        &self,
        previous_claim: F,
        relation: Stage4Relation,
    ) -> Result<UnivariatePoly<F>, Stage4KernelError> {
        #[cfg(feature = "cuda")]
        if let Some(cuda) = &self.cuda {
            if let Ok(poly) = cuda.round_poly() {
                if let Some(poly) = fr_poly_into::<F>(poly) {
                    check_round_claim(
                        &poly,
                        previous_claim,
                        relation.symbol(),
                        "stage4 relation input claim mismatch",
                    )?;
                    return Ok(poly);
                }
            }
        }
        let first_len = self.factors.first().map_or(0, Vec::len);
        if first_len == 0 || !first_len.is_power_of_two() {
            return Err(Stage4KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "stage4 dense factor has invalid length",
            });
        }
        if self.factors.iter().any(|factor| factor.len() != first_len) {
            return Err(Stage4KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "stage4 dense factors have inconsistent lengths",
            });
        }
        let poly =
            round_poly_from_dense_terms(&self.factors, &self.terms, self.active_scale, relation)?;
        check_round_claim(
            &poly,
            previous_claim,
            relation.symbol(),
            "stage4 relation input claim mismatch",
        )?;
        Ok(poly)
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

    fn factor_eval(&self, index: usize, relation: Stage4Relation) -> Result<F, Stage4KernelError> {
        #[cfg(feature = "cuda")]
        if let Some(cuda) = &self.cuda {
            if let Some(Ok(value)) = cuda.factor_eval(index) {
                if let Some(value) = crate::cuda::fr_into::<F>(value) {
                    return Ok(value);
                }
            }
        }
        self.factors
            .get(index)
            .and_then(|values| values.first())
            .copied()
            .ok_or(Stage4KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "empty stage4 factor",
            })
    }

    fn final_evals(
        &self,
        relation: Stage4Relation,
    ) -> Result<Vec<Stage4NamedEval<F>>, Stage4KernelError> {
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

struct SparseRegistersState<F: Field> {
    register_count: usize,
    trace_len: usize,
    current_trace_len: usize,
    entries: Vec<SparseRegisterEntry<F>>,
    entry_scratch: Vec<SparseRegisterEntry<F>>,
    rs2_reads: Vec<(usize, usize)>,
    eq_cycle: SplitEqState<F>,
    rd_inc: Vec<F>,
    rd_inc_scratch: Vec<F>,
    gamma: F,
    gamma2: F,
    active_scale: F,
    bound_point: Vec<F>,
    dense: Option<DenseStage4State<F>>,
    backend: &'static str,
    #[cfg(feature = "cuda")]
    cuda: Option<cuda::CudaSparseRegistersState>,
}

#[derive(Clone, Copy, Debug)]
struct SparseRegisterEntry<F: Field> {
    row: usize,
    col: u8,
    val: F,
    prev_val: u64,
    next_val: u64,
    read_ra: F,
    rd_wa: F,
}

impl<F: Field> SparseRegistersState<F> {
    fn new(
        register_count: usize,
        trace_len: usize,
        accesses: &[Stage4RegisterAccess],
        rd_inc: &[F],
        trace_point: &[F],
        gamma: F,
        gamma2: F,
        active_scale: F,
        backend: &'static str,
    ) -> Result<Self, Stage4KernelError> {
        require_operand_count("stage4.registers.accesses", trace_len, accesses.len())?;
        require_operand_count("stage4.registers.RdInc", trace_len, rd_inc.len())?;

        #[cfg(feature = "cuda")]
        if backend == "cuda" && trace_len > 1 {
            if let Some(cuda) = cuda_sparse_registers_from_raw(
                register_count,
                accesses,
                rd_inc,
                trace_point,
                gamma,
                gamma2,
                log2_exact(trace_len, "stage4.trace_len")?,
            )? {
                let mut rs2_reads = Vec::new();
                for (row, access) in accesses.iter().enumerate() {
                    if let Some(rs2) = access.rs2 {
                        rs2_reads.push((row, rs2.address));
                    }
                }
                return Ok(Self {
                    register_count,
                    trace_len,
                    current_trace_len: trace_len,
                    entries: Vec::new(),
                    entry_scratch: Vec::new(),
                    rs2_reads,
                    eq_cycle: SplitEqState::new_low_to_high(trace_point, None),
                    rd_inc: Vec::new(),
                    rd_inc_scratch: Vec::new(),
                    gamma,
                    gamma2,
                    active_scale,
                    bound_point: Vec::with_capacity(
                        log2_exact(register_count, "stage4.register_count")?
                            + log2_exact(trace_len, "stage4.trace_len")?,
                    ),
                    dense: None,
                    backend,
                    cuda: Some(cuda),
                });
            }
        }

        let mut entries = Vec::with_capacity(accesses.len().saturating_mul(3));
        let mut rs2_reads = Vec::with_capacity(accesses.len());
        for (row, access) in accesses.iter().enumerate() {
            append_sparse_register_entries(
                register_count,
                row,
                *access,
                gamma,
                gamma2,
                &mut entries,
            )?;
            if let Some(rs2) = access.rs2 {
                rs2_reads.push((row, rs2.address));
            }
        }
        let eq_cycle = SplitEqState::new_low_to_high(trace_point, None);
        #[cfg(feature = "cuda")]
        let cuda = if backend == "cuda" && trace_len > 1 {
            build_cuda_sparse_registers(
                &entries,
                rd_inc,
                trace_point,
                log2_exact(trace_len, "stage4.trace_len")?,
            )
        } else {
            None
        };
        #[cfg(not(feature = "cuda"))]
        let _ = backend;
        let mut state = Self {
            register_count,
            trace_len,
            current_trace_len: trace_len,
            entries,
            entry_scratch: Vec::new(),
            rs2_reads,
            eq_cycle,
            rd_inc: rd_inc.to_vec(),
            rd_inc_scratch: Vec::new(),
            gamma,
            gamma2,
            active_scale,
            bound_point: Vec::with_capacity(
                log2_exact(register_count, "stage4.register_count")?
                    + log2_exact(trace_len, "stage4.trace_len")?,
            ),
            dense: None,
            backend,
            #[cfg(feature = "cuda")]
            cuda,
        };
        if trace_len == 1 {
            state.materialize_dense()?;
        }
        Ok(state)
    }

    fn round_poly(
        &mut self,
        previous_claim: F,
        relation: Stage4Relation,
    ) -> Result<UnivariatePoly<F>, Stage4KernelError> {
        if let Some(dense) = &self.dense {
            return dense.round_poly(previous_claim, relation);
        }
        if self.current_trace_len <= 1 {
            return Err(Stage4KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "stage4 sparse registers state was not materialized",
            });
        }
        #[cfg(feature = "cuda")]
        if let Some(cuda) = &self.cuda {
            if let Some(q) = cuda.round_poly_q() {
                if let (Some(q_constant), Some(q_quadratic)) =
                    (crate::cuda::fr_into::<F>(q[0]), crate::cuda::fr_into::<F>(q[1]))
                {
                    let poly = gruen_cubic_poly(
                        self.eq_cycle.current_target(),
                        q_constant * self.active_scale,
                        q_quadratic * self.active_scale,
                        previous_claim,
                    );
                    check_round_claim(
                        &poly,
                        previous_claim,
                        relation.symbol(),
                        "stage4 sparse registers input claim mismatch",
                    )?;
                    return Ok(poly);
                }
            }
        }
        let (mut q_constant, mut q_quadratic) = sparse_register_split_round_coefficients(
            &self.entries,
            &self.eq_cycle,
            &self.rd_inc,
            self.current_trace_len,
        )?;
        q_constant *= self.active_scale;
        q_quadratic *= self.active_scale;
        let poly = gruen_cubic_poly(
            self.eq_cycle.current_target(),
            q_constant,
            q_quadratic,
            previous_claim,
        );
        check_round_claim(
            &poly,
            previous_claim,
            relation.symbol(),
            "stage4 sparse registers input claim mismatch",
        )?;
        Ok(poly)
    }

    fn bind(&mut self, challenge: F) {
        self.bound_point.push(challenge);
        if let Some(dense) = &mut self.dense {
            dense.bind(challenge);
            return;
        }
        #[cfg(feature = "cuda")]
        if let Some(cuda) = &mut self.cuda {
            if let Some(challenge_fr) = crate::cuda::into_fr(challenge) {
                if cuda.bind(challenge_fr).is_ok() {
                    self.eq_cycle.bind(challenge);
                    self.current_trace_len /= 2;
                    if self.current_trace_len == 1 {
                        let _ = self.materialize_dense();
                    }
                    return;
                }
            }
        }
        bind_sparse_register_entries_into(
            &self.entries,
            self.current_trace_len,
            challenge,
            &mut self.entry_scratch,
        );
        std::mem::swap(&mut self.entries, &mut self.entry_scratch);
        self.entry_scratch.clear();
        self.eq_cycle.bind(challenge);
        bind_dense_evals_reuse(&mut self.rd_inc, &mut self.rd_inc_scratch, challenge);
        self.current_trace_len /= 2;
        if self.current_trace_len == 1 {
            let _ = self.materialize_dense();
        }
    }

    fn final_evals(
        &self,
        relation: Stage4Relation,
    ) -> Result<Vec<Stage4NamedEval<F>>, Stage4KernelError> {
        let dense = self.dense.as_ref().ok_or(Stage4KernelError::InvalidProof {
            driver: relation.symbol(),
            reason: "stage4 sparse registers state was not materialized",
        })?;
        let registers_val = dense.factor_eval(1, relation)?;
        let combined_read_ra = dense.factor_eval(2, relation)?;
        let rd_wa = dense.factor_eval(3, relation)?;
        let rd_inc = dense.factor_eval(4, relation)?;
        let rs2_ra = self.final_rs2_read_eval(relation)?;
        let gamma_inverse = self
            .gamma
            .inverse()
            .ok_or(Stage4KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "stage4 registers challenge is not invertible",
            })?;
        let rs1_ra = (combined_read_ra - self.gamma2 * rs2_ra) * gamma_inverse;
        #[cfg(debug_assertions)]
        {
            let expected = self.gamma * rs1_ra + self.gamma2 * rs2_ra;
            if combined_read_ra != expected {
                return Err(Stage4KernelError::InvalidProof {
                    driver: relation.symbol(),
                    reason: "stage4 sparse registers final read claim mismatch",
                });
            }
        }
        Ok(vec![
            named_eval(
                "stage4.registers_read_write.eval.RegistersVal",
                "RegistersVal",
                registers_val,
            ),
            named_eval("stage4.registers_read_write.eval.Rs1Ra", "Rs1Ra", rs1_ra),
            named_eval("stage4.registers_read_write.eval.Rs2Ra", "Rs2Ra", rs2_ra),
            named_eval("stage4.registers_read_write.eval.RdWa", "RdWa", rd_wa),
            named_eval("stage4.registers_read_write.eval.RdInc", "RdInc", rd_inc),
        ])
    }

    fn materialize_dense(&mut self) -> Result<(), Stage4KernelError> {
        #[cfg(feature = "cuda")]
        let cuda_materialized = self
            .cuda
            .as_ref()
            .and_then(|cuda| cuda.materialize::<F>(self.register_count));
        #[cfg(not(feature = "cuda"))]
        let cuda_materialized: Option<(Vec<F>, Vec<F>, Vec<F>)> = None;

        let (registers_val, read_ra, rd_wa) = if let Some(materialized) = cuda_materialized {
            materialized
        } else {
            let mut registers_val = vec![F::zero(); self.register_count];
            let mut read_ra = vec![F::zero(); self.register_count];
            let mut rd_wa = vec![F::zero(); self.register_count];
            for entry in &self.entries {
                let col = usize::from(entry.col);
                if entry.row != 0 || col >= self.register_count {
                    return Err(Stage4KernelError::InvalidInputLength {
                        input: "stage4.registers.accesses",
                        expected: self.register_count,
                        actual: col + 1,
                    });
                }
                registers_val[col] = entry.val;
                read_ra[col] = entry.read_ra;
                rd_wa[col] = entry.rd_wa;
            }
            (registers_val, read_ra, rd_wa)
        };
        let eq_eval = self.eq_cycle.eval();
        #[cfg(feature = "cuda")]
        let cuda_rd_inc = self
            .cuda
            .as_ref()
            .and_then(|cuda| crate::cuda::fr_into::<F>(cuda.rd_inc_first().ok()?));
        #[cfg(not(feature = "cuda"))]
        let cuda_rd_inc: Option<F> = None;
        let rd_inc_eval = if let Some(rd_inc_eval) = cuda_rd_inc {
            rd_inc_eval
        } else {
            self.rd_inc
                .first()
                .copied()
                .ok_or(Stage4KernelError::InvalidInputLength {
                    input: "stage4.registers.RdInc",
                    expected: 1,
                    actual: 0,
                })?
        };
        self.dense = Some(registers_combined_dense_state(
            vec![eq_eval; self.register_count],
            registers_val,
            read_ra,
            rd_wa,
            vec![rd_inc_eval; self.register_count],
            self.active_scale,
            self.backend,
        ));
        Ok(())
    }

    fn final_rs2_read_eval(&self, relation: Stage4Relation) -> Result<F, Stage4KernelError> {
        let trace_rounds = log2_exact(self.trace_len, "stage4.trace_len")?;
        let register_rounds = log2_exact(self.register_count, "stage4.register_count")?;
        if self.bound_point.len() != trace_rounds + register_rounds {
            return Err(Stage4KernelError::InvalidInputLength {
                input: "stage4.registers_read_write.instance",
                expected: trace_rounds + register_rounds,
                actual: self.bound_point.len(),
            });
        }
        let (cycle_point, address_point) = self.bound_point.split_at(trace_rounds);
        let r_cycle = reverse_slice(cycle_point);
        let r_address = reverse_slice(address_point);
        let (cycle_eq, address_eq) = rayon::join(
            || EqPolynomial::<F>::evals(&r_cycle, None),
            || EqPolynomial::<F>::evals(&r_address, None),
        );
        if cycle_eq.len() != self.trace_len || address_eq.len() != self.register_count {
            return Err(Stage4KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "stage4 sparse registers final read point has invalid shape",
            });
        }
        Ok(sparse_register_read_eval(
            &self.rs2_reads,
            &cycle_eq,
            &address_eq,
        ))
    }
}

fn registers_read_write_state<F: Field>(
    claim: &Stage4SumcheckClaimPlan,
    inputs: &Stage4ProverInputs<'_, F>,
    store: &Stage4ValueStore<F>,
    active_scale: F,
    backend: &'static str,
) -> Result<Stage4ProverInstanceState<F>, Stage4KernelError> {
    let witness = inputs
        .registers
        .ok_or(Stage4KernelError::MissingKernelInput {
            kernel: "jolt_stage4_batched",
            input: "registers",
        })?;
    let expected_len = witness
        .register_count
        .checked_mul(witness.trace_len)
        .ok_or(Stage4KernelError::InvalidInputLength {
            input: "stage4.registers",
            expected: usize::MAX,
            actual: witness.register_count,
        })?;

    let trace_point = store.point("stage4.input.stage3.registers.RdWriteValue")?;
    let register_rounds = log2_exact(witness.register_count, "stage4.register_count")?;
    let trace_rounds = log2_exact(witness.trace_len, "stage4.trace_len")?;
    require_operand_count(
        "stage4.registers.trace_point",
        trace_rounds,
        trace_point.len(),
    )?;
    require_operand_count(
        claim.symbol,
        register_rounds + trace_rounds,
        claim.num_rounds,
    )?;

    let gamma = store.scalar("stage4.registers_read_write.gamma")?;
    let gamma2 = store
        .try_scalar("stage4.registers_read_write.gamma2")
        .unwrap_or_else(|| gamma * gamma);

    if let Some(accesses) = witness.accesses {
        return SparseRegistersState::new(
            witness.register_count,
            witness.trace_len,
            accesses,
            witness.rd_inc,
            trace_point,
            gamma,
            gamma2,
            active_scale,
            backend,
        )
        .map(Stage4ProverInstanceState::SparseRegisters);
    }

    require_operand_count(
        "stage4.registers.RegistersVal",
        expected_len,
        witness.registers_val.len(),
    )?;
    require_operand_count("stage4.registers.Rs1Ra", expected_len, witness.rs1_ra.len())?;
    require_operand_count("stage4.registers.Rs2Ra", expected_len, witness.rs2_ra.len())?;
    require_operand_count("stage4.registers.RdWa", expected_len, witness.rd_wa.len())?;
    require_operand_count(
        "stage4.registers.RdInc",
        witness.trace_len,
        witness.rd_inc.len(),
    )?;
    let eq_cycle = EqPolynomial::<F>::evals(trace_point, None);
    let mut eq_cycle_expanded = Vec::with_capacity(expected_len);
    let mut rd_inc_expanded = Vec::with_capacity(expected_len);
    for _address in 0..witness.register_count {
        eq_cycle_expanded.extend_from_slice(&eq_cycle);
        rd_inc_expanded.extend_from_slice(witness.rd_inc);
    }

    Ok(Stage4ProverInstanceState::Dense(registers_dense_state(
        eq_cycle_expanded,
        witness.registers_val.to_vec(),
        witness.rs1_ra.to_vec(),
        witness.rs2_ra.to_vec(),
        witness.rd_wa.to_vec(),
        rd_inc_expanded,
        gamma,
        gamma2,
        active_scale,
    )))
}

fn registers_dense_state<F: Field>(
    eq_cycle: Vec<F>,
    registers_val: Vec<F>,
    rs1_ra: Vec<F>,
    rs2_ra: Vec<F>,
    rd_wa: Vec<F>,
    rd_inc: Vec<F>,
    gamma: F,
    gamma2: F,
    active_scale: F,
) -> DenseStage4State<F> {
    DenseStage4State::new(
        vec![eq_cycle, registers_val, rs1_ra, rs2_ra, rd_wa, rd_inc],
        vec![
            DenseTerm {
                coefficient: F::one(),
                factors: vec![0, 4, 1],
            },
            DenseTerm {
                coefficient: F::one(),
                factors: vec![0, 4, 5],
            },
            DenseTerm {
                coefficient: gamma,
                factors: vec![0, 2, 1],
            },
            DenseTerm {
                coefficient: gamma2,
                factors: vec![0, 3, 1],
            },
        ],
        vec![
            FactorOutput {
                name: "stage4.registers_read_write.eval.RegistersVal",
                oracle: "RegistersVal",
                factor: 1,
            },
            FactorOutput {
                name: "stage4.registers_read_write.eval.Rs1Ra",
                oracle: "Rs1Ra",
                factor: 2,
            },
            FactorOutput {
                name: "stage4.registers_read_write.eval.Rs2Ra",
                oracle: "Rs2Ra",
                factor: 3,
            },
            FactorOutput {
                name: "stage4.registers_read_write.eval.RdWa",
                oracle: "RdWa",
                factor: 4,
            },
            FactorOutput {
                name: "stage4.registers_read_write.eval.RdInc",
                oracle: "RdInc",
                factor: 5,
            },
        ],
        active_scale,
    )
}

fn registers_combined_dense_state<F: Field>(
    eq_cycle: Vec<F>,
    registers_val: Vec<F>,
    read_ra: Vec<F>,
    rd_wa: Vec<F>,
    rd_inc: Vec<F>,
    active_scale: F,
    backend: &'static str,
) -> DenseStage4State<F> {
    DenseStage4State::new_with_backend(
        vec![eq_cycle, registers_val, read_ra, rd_wa, rd_inc],
        vec![
            DenseTerm {
                coefficient: F::one(),
                factors: vec![0, 3, 1],
            },
            DenseTerm {
                coefficient: F::one(),
                factors: vec![0, 3, 4],
            },
            DenseTerm {
                coefficient: F::one(),
                factors: vec![0, 2, 1],
            },
        ],
        Vec::new(),
        active_scale,
        backend,
    )
}

fn append_sparse_register_entries<F: Field>(
    register_count: usize,
    row: usize,
    access: Stage4RegisterAccess,
    gamma: F,
    gamma2: F,
    entries: &mut Vec<SparseRegisterEntry<F>>,
) -> Result<(), Stage4KernelError> {
    let start = entries.len();
    if let Some(rs1) = access.rs1 {
        validate_register_address(register_count, rs1.address)?;
        let col = sparse_register_col(rs1.address)?;
        entries.push(SparseRegisterEntry {
            row,
            col,
            val: F::from_u64(rs1.value),
            prev_val: rs1.value,
            next_val: rs1.value,
            read_ra: gamma,
            rd_wa: F::zero(),
        });
    }
    if let Some(rs2) = access.rs2 {
        validate_register_address(register_count, rs2.address)?;
        let col = sparse_register_col(rs2.address)?;
        if let Some(entry) = entries[start..].iter_mut().find(|entry| entry.col == col) {
            entry.read_ra += gamma2;
        } else {
            entries.push(SparseRegisterEntry {
                row,
                col,
                val: F::from_u64(rs2.value),
                prev_val: rs2.value,
                next_val: rs2.value,
                read_ra: gamma2,
                rd_wa: F::zero(),
            });
        }
    }
    if let Some(rd) = access.rd {
        validate_register_address(register_count, rd.address)?;
        let col = sparse_register_col(rd.address)?;
        if let Some(entry) = entries[start..].iter_mut().find(|entry| entry.col == col) {
            entry.rd_wa = F::one();
            entry.next_val = rd.post_value;
        } else {
            entries.push(SparseRegisterEntry {
                row,
                col,
                val: F::from_u64(rd.pre_value),
                prev_val: rd.pre_value,
                next_val: rd.post_value,
                read_ra: F::zero(),
                rd_wa: F::one(),
            });
        }
    }
    entries[start..].sort_by_key(|entry| entry.col);
    Ok(())
}

fn sparse_register_col(address: usize) -> Result<u8, Stage4KernelError> {
    u8::try_from(address).map_err(|_| Stage4KernelError::InvalidInputLength {
        input: "stage4.registers.accesses",
        expected: usize::from(u8::MAX) + 1,
        actual: address + 1,
    })
}

fn validate_register_address(
    register_count: usize,
    address: usize,
) -> Result<(), Stage4KernelError> {
    if address < register_count {
        Ok(())
    } else {
        Err(Stage4KernelError::InvalidInputLength {
            input: "stage4.registers.accesses",
            expected: register_count,
            actual: address + 1,
        })
    }
}

fn sparse_register_split_round_coefficients<F: Field>(
    entries: &[SparseRegisterEntry<F>],
    eq_cycle: &SplitEqState<F>,
    rd_inc: &[F],
    current_trace_len: usize,
) -> Result<(F, F), Stage4KernelError> {
    if let Some(entry) = entries.last() {
        if entry.row >= current_trace_len {
            return Err(Stage4KernelError::InvalidInputLength {
                input: "stage4.registers.accesses",
                expected: current_trace_len,
                actual: entry.row + 1,
            });
        }
    }
    let e_in = eq_cycle.e_in();
    let e_out = eq_cycle.e_out();
    if e_in.len() > 1 {
        sparse_register_low_round_coefficients(entries, rd_inc, e_in, e_out)
    } else {
        sparse_register_high_round_coefficients(entries, rd_inc, e_in[0], e_out)
    }
}

fn sparse_register_low_round_coefficients<F: Field>(
    entries: &[SparseRegisterEntry<F>],
    rd_inc: &[F],
    e_in: &[F],
    e_out: &[F],
) -> Result<(F, F), Stage4KernelError> {
    let in_pairs = e_in.len() / 2;
    if entries.len() >= DENSE_BIND_PAR_THRESHOLD {
        let accumulators = entries
            .par_chunk_by(|left, right| (left.row / 2) / in_pairs == (right.row / 2) / in_pairs)
            .map(|entries| {
                let mut local = [F::Accumulator::default(); 2];
                accumulate_sparse_register_low_outer_chunk(
                    &mut local, entries, rd_inc, e_in, e_out, in_pairs,
                );
                local
            })
            .reduce(
                || [F::Accumulator::default(); 2],
                |mut left, right| {
                    for (left, right) in left.iter_mut().zip(right) {
                        left.merge(right);
                    }
                    left
                },
            );
        return Ok((accumulators[0].reduce(), accumulators[1].reduce()));
    }

    let mut accumulators = [F::Accumulator::default(); 2];
    let mut cursor = 0usize;
    while cursor < entries.len() {
        let pair = entries[cursor].row / 2;
        let x_out = pair / in_pairs;
        let x_in = pair % in_pairs;
        let weight = e_out[x_out] * (e_in[2 * x_in] + e_in[2 * x_in + 1]);
        let even_row = 2 * pair;
        let odd_row = even_row + 1;
        let even_start = cursor;
        while cursor < entries.len() && entries[cursor].row == even_row {
            cursor += 1;
        }
        let even = &entries[even_start..cursor];
        let odd_start = cursor;
        while cursor < entries.len() && entries[cursor].row == odd_row {
            cursor += 1;
        }
        let odd = &entries[odd_start..cursor];
        accumulate_sparse_register_row_pair_body_coefficients(
            &mut accumulators,
            even,
            odd,
            rd_inc[even_row],
            rd_inc[odd_row] - rd_inc[even_row],
            weight,
        );
    }
    Ok((accumulators[0].reduce(), accumulators[1].reduce()))
}

fn sparse_register_high_round_coefficients<F: Field>(
    entries: &[SparseRegisterEntry<F>],
    rd_inc: &[F],
    in_weight: F,
    e_out: &[F],
) -> Result<(F, F), Stage4KernelError> {
    if entries.len() >= DENSE_BIND_PAR_THRESHOLD {
        let accumulators = entries
            .par_chunk_by(|left, right| left.row / 2 == right.row / 2)
            .map(|entries| {
                let mut local = [F::Accumulator::default(); 2];
                let pair = entries[0].row / 2;
                let weight = in_weight * (e_out[2 * pair] + e_out[2 * pair + 1]);
                accumulate_sparse_register_row_pair_chunk(&mut local, entries, rd_inc, weight);
                local
            })
            .reduce(
                || [F::Accumulator::default(); 2],
                |mut left, right| {
                    for (left, right) in left.iter_mut().zip(right) {
                        left.merge(right);
                    }
                    left
                },
            );
        return Ok((accumulators[0].reduce(), accumulators[1].reduce()));
    }

    let mut accumulators = [F::Accumulator::default(); 2];
    let mut cursor = 0usize;
    while cursor < entries.len() {
        let pair = entries[cursor].row / 2;
        let weight = in_weight * (e_out[2 * pair] + e_out[2 * pair + 1]);
        let start = cursor;
        let even_row = 2 * pair;
        while cursor < entries.len()
            && (entries[cursor].row == even_row || entries[cursor].row == even_row + 1)
        {
            cursor += 1;
        }
        accumulate_sparse_register_row_pair_chunk(
            &mut accumulators,
            &entries[start..cursor],
            rd_inc,
            weight,
        );
    }
    Ok((accumulators[0].reduce(), accumulators[1].reduce()))
}

fn accumulate_sparse_register_low_outer_chunk<F: Field>(
    accumulators: &mut [F::Accumulator; 2],
    entries: &[SparseRegisterEntry<F>],
    rd_inc: &[F],
    e_in: &[F],
    e_out: &[F],
    in_pairs: usize,
) {
    let x_out = (entries[0].row / 2) / in_pairs;
    let out_weight = e_out[x_out];
    for entries in entries.chunk_by(|left, right| left.row / 2 == right.row / 2) {
        let pair = entries[0].row / 2;
        let x_in = pair % in_pairs;
        let weight = out_weight * (e_in[2 * x_in] + e_in[2 * x_in + 1]);
        accumulate_sparse_register_row_pair_chunk(accumulators, entries, rd_inc, weight);
    }
}

fn accumulate_sparse_register_row_pair_chunk<F: Field>(
    accumulators: &mut [F::Accumulator; 2],
    entries: &[SparseRegisterEntry<F>],
    rd_inc: &[F],
    weight: F,
) {
    let pair = entries[0].row / 2;
    let even_row = 2 * pair;
    let odd_start = entries.partition_point(|entry| entry.row == even_row);
    let (even, odd) = entries.split_at(odd_start);
    accumulate_sparse_register_row_pair_body_coefficients(
        accumulators,
        even,
        odd,
        rd_inc[even_row],
        rd_inc[even_row + 1] - rd_inc[even_row],
        weight,
    );
}

#[derive(Clone)]
struct SparseRegisterPairRange {
    pair: usize,
    even: std::ops::Range<usize>,
    odd: std::ops::Range<usize>,
}

fn sparse_register_pair_ranges_into<F: Field>(
    entries: &[SparseRegisterEntry<F>],
    current_trace_len: usize,
    ranges: &mut Vec<SparseRegisterPairRange>,
) -> Result<(), Stage4KernelError> {
    let half = current_trace_len / 2;
    ranges.clear();
    let mut cursor = 0usize;
    while cursor < entries.len() {
        let pair = entries[cursor].row / 2;
        if pair >= half {
            return Err(Stage4KernelError::InvalidInputLength {
                input: "stage4.registers.accesses",
                expected: current_trace_len,
                actual: entries[cursor].row + 1,
            });
        }
        let even_row = 2 * pair;
        let odd_row = even_row + 1;
        let even_start = cursor;
        while cursor < entries.len() && entries[cursor].row == even_row {
            cursor += 1;
        }
        let even = even_start..cursor;
        let odd_start = cursor;
        while cursor < entries.len() && entries[cursor].row == odd_row {
            cursor += 1;
        }
        let odd = odd_start..cursor;
        ranges.push(SparseRegisterPairRange { pair, even, odd });
    }
    Ok(())
}

fn sparse_register_pair_ranges<F: Field>(
    entries: &[SparseRegisterEntry<F>],
    current_trace_len: usize,
) -> Result<Vec<SparseRegisterPairRange>, Stage4KernelError> {
    let mut ranges = Vec::new();
    sparse_register_pair_ranges_into(entries, current_trace_len, &mut ranges)?;
    Ok(ranges)
}

fn accumulate_sparse_register_row_pair_body_coefficients<F: Field>(
    accumulators: &mut [F::Accumulator; 2],
    even: &[SparseRegisterEntry<F>],
    odd: &[SparseRegisterEntry<F>],
    inc0: F,
    inc_delta: F,
    weight: F,
) {
    let mut i = 0usize;
    let mut j = 0usize;
    while i < even.len() || j < odd.len() {
        let (even_entry, odd_entry) =
            if j >= odd.len() || (i < even.len() && even[i].col < odd[j].col) {
                let pair = (Some(&even[i]), None);
                i += 1;
                pair
            } else if i >= even.len() || odd[j].col < even[i].col {
                let pair = (None, Some(&odd[j]));
                j += 1;
                pair
            } else {
                let pair = (Some(&even[i]), Some(&odd[j]));
                i += 1;
                j += 1;
                pair
            };
        accumulate_sparse_register_entry_pair_body_coefficients(
            accumulators,
            even_entry,
            odd_entry,
            inc0,
            inc_delta,
            weight,
        );
    }
}

#[derive(Clone, Copy)]
struct SparseRegisterEval<F: Field> {
    val: F,
    read_ra: F,
    rd_wa: F,
}

#[derive(Clone, Copy)]
struct SparseRegisterLinear<F: Field> {
    val0: F,
    val_delta: F,
    read_ra0: F,
    read_ra_delta: F,
    rd_wa0: F,
    rd_wa_delta: F,
}

fn accumulate_sparse_register_entry_pair_body_coefficients<F: Field>(
    accumulators: &mut [F::Accumulator; 2],
    even: Option<&SparseRegisterEntry<F>>,
    odd: Option<&SparseRegisterEntry<F>>,
    inc0: F,
    inc_delta: F,
    weight: F,
) {
    let linear = sparse_register_entry_linear(even, odd);
    let val_inc0 = linear.val0 + inc0;
    let val_inc_delta = linear.val_delta + inc_delta;
    let body0 = linear.rd_wa0 * val_inc0 + linear.read_ra0 * linear.val0;
    let body2 = linear.rd_wa_delta * val_inc_delta + linear.read_ra_delta * linear.val_delta;

    accumulators[0].fmadd(weight, body0);
    accumulators[1].fmadd(weight, body2);
}

fn sparse_register_entry_linear<F: Field>(
    even: Option<&SparseRegisterEntry<F>>,
    odd: Option<&SparseRegisterEntry<F>>,
) -> SparseRegisterLinear<F> {
    match (even, odd) {
        (Some(even), Some(odd)) => SparseRegisterLinear {
            val0: even.val,
            val_delta: odd.val - even.val,
            read_ra0: even.read_ra,
            read_ra_delta: odd.read_ra - even.read_ra,
            rd_wa0: even.rd_wa,
            rd_wa_delta: odd.rd_wa - even.rd_wa,
        },
        (Some(even), None) => SparseRegisterLinear {
            val0: even.val,
            val_delta: F::from_u64(even.next_val) - even.val,
            read_ra0: even.read_ra,
            read_ra_delta: -even.read_ra,
            rd_wa0: even.rd_wa,
            rd_wa_delta: -even.rd_wa,
        },
        (None, Some(odd)) => SparseRegisterLinear {
            val0: F::from_u64(odd.prev_val),
            val_delta: odd.val - F::from_u64(odd.prev_val),
            read_ra0: F::zero(),
            read_ra_delta: odd.read_ra,
            rd_wa0: F::zero(),
            rd_wa_delta: odd.rd_wa,
        },
        (None, None) => SparseRegisterLinear {
            val0: F::zero(),
            val_delta: F::zero(),
            read_ra0: F::zero(),
            read_ra_delta: F::zero(),
            rd_wa0: F::zero(),
            rd_wa_delta: F::zero(),
        },
    }
}

fn sparse_register_entry_eval<F: Field>(
    even: Option<&SparseRegisterEntry<F>>,
    odd: Option<&SparseRegisterEntry<F>>,
    x: F,
) -> SparseRegisterEval<F> {
    match (even, odd) {
        (Some(even), Some(odd)) => SparseRegisterEval {
            val: linear_eval(even.val, odd.val, x),
            read_ra: linear_eval(even.read_ra, odd.read_ra, x),
            rd_wa: linear_eval(even.rd_wa, odd.rd_wa, x),
        },
        (Some(even), None) => SparseRegisterEval {
            val: linear_eval(even.val, F::from_u64(even.next_val), x),
            read_ra: linear_eval(even.read_ra, F::zero(), x),
            rd_wa: linear_eval(even.rd_wa, F::zero(), x),
        },
        (None, Some(odd)) => SparseRegisterEval {
            val: linear_eval(F::from_u64(odd.prev_val), odd.val, x),
            read_ra: linear_eval(F::zero(), odd.read_ra, x),
            rd_wa: linear_eval(F::zero(), odd.rd_wa, x),
        },
        (None, None) => SparseRegisterEval {
            val: F::zero(),
            read_ra: F::zero(),
            rd_wa: F::zero(),
        },
    }
}

fn bind_sparse_register_entries_into<F: Field>(
    entries: &[SparseRegisterEntry<F>],
    current_trace_len: usize,
    challenge: F,
    output: &mut Vec<SparseRegisterEntry<F>>,
) {
    output.clear();
    if entries.len() >= DENSE_BIND_PAR_THRESHOLD {
        if let Ok(ranges) = sparse_register_pair_ranges(entries, current_trace_len) {
            let bound_lengths = ranges
                .par_iter()
                .map(|range| {
                    sparse_register_row_pair_bound_len(
                        &entries[range.even.clone()],
                        &entries[range.odd.clone()],
                    )
                })
                .collect::<Vec<_>>();
            let output_len = bound_lengths.iter().sum();
            output.reserve(output_len);
            let mut spare = output.spare_capacity_mut();
            let mut output_slices = Vec::with_capacity(ranges.len());
            for &bound_len in &bound_lengths {
                let (slice, rest) = spare.split_at_mut(bound_len);
                output_slices.push(slice);
                spare = rest;
            }
            ranges
                .par_iter()
                .zip(output_slices.into_par_iter())
                .for_each(|(range, output)| {
                    bind_sparse_register_row_pair_into(
                        &entries[range.even.clone()],
                        &entries[range.odd.clone()],
                        range.pair,
                        challenge,
                        output,
                    );
                });
            // SAFETY: every slot in `output_slices` was initialized exactly once by
            // `bind_sparse_register_row_pair_into`.
            unsafe {
                output.set_len(output_len);
            }
            return;
        }
    }

    bind_sparse_register_entries_sequential_into(entries, challenge, output);
}

fn bind_sparse_register_entries_sequential_into<F: Field>(
    entries: &[SparseRegisterEntry<F>],
    challenge: F,
    output: &mut Vec<SparseRegisterEntry<F>>,
) {
    output.reserve(entries.len());
    let mut cursor = 0usize;
    while cursor < entries.len() {
        let pair = entries[cursor].row / 2;
        let even_row = 2 * pair;
        let odd_row = even_row + 1;
        let even_start = cursor;
        while cursor < entries.len() && entries[cursor].row == even_row {
            cursor += 1;
        }
        let even = &entries[even_start..cursor];
        let odd_start = cursor;
        while cursor < entries.len() && entries[cursor].row == odd_row {
            cursor += 1;
        }
        let odd = &entries[odd_start..cursor];
        bind_sparse_register_row_pair(even, odd, pair, challenge, output);
    }
}

fn sparse_register_row_pair_bound_len<F: Field>(
    even: &[SparseRegisterEntry<F>],
    odd: &[SparseRegisterEntry<F>],
) -> usize {
    let mut i = 0usize;
    let mut j = 0usize;
    let mut len = 0usize;
    while i < even.len() || j < odd.len() {
        if j >= odd.len() || (i < even.len() && even[i].col < odd[j].col) {
            i += 1;
        } else if i >= even.len() || odd[j].col < even[i].col {
            j += 1;
        } else {
            i += 1;
            j += 1;
        }
        len += 1;
    }
    len
}

fn bind_sparse_register_row_pair_into<F: Field>(
    even: &[SparseRegisterEntry<F>],
    odd: &[SparseRegisterEntry<F>],
    row: usize,
    challenge: F,
    output: &mut [MaybeUninit<SparseRegisterEntry<F>>],
) {
    let mut i = 0usize;
    let mut j = 0usize;
    let mut out = 0usize;
    while i < even.len() || j < odd.len() {
        let (even_entry, odd_entry, col) =
            if j >= odd.len() || (i < even.len() && even[i].col < odd[j].col) {
                let pair = (Some(&even[i]), None, even[i].col);
                i += 1;
                pair
            } else if i >= even.len() || odd[j].col < even[i].col {
                let pair = (None, Some(&odd[j]), odd[j].col);
                j += 1;
                pair
            } else {
                let pair = (Some(&even[i]), Some(&odd[j]), even[i].col);
                i += 1;
                j += 1;
                pair
            };
        output[out] = MaybeUninit::new(bind_sparse_register_entry_pair(
            even_entry, odd_entry, row, col, challenge,
        ));
        out += 1;
    }
    debug_assert_eq!(out, output.len());
}

fn bind_sparse_register_row_pair<F: Field>(
    even: &[SparseRegisterEntry<F>],
    odd: &[SparseRegisterEntry<F>],
    row: usize,
    challenge: F,
    output: &mut Vec<SparseRegisterEntry<F>>,
) {
    let mut i = 0usize;
    let mut j = 0usize;
    while i < even.len() || j < odd.len() {
        let (even_entry, odd_entry, col) =
            if j >= odd.len() || (i < even.len() && even[i].col < odd[j].col) {
                let pair = (Some(&even[i]), None, even[i].col);
                i += 1;
                pair
            } else if i >= even.len() || odd[j].col < even[i].col {
                let pair = (None, Some(&odd[j]), odd[j].col);
                j += 1;
                pair
            } else {
                let pair = (Some(&even[i]), Some(&odd[j]), even[i].col);
                i += 1;
                j += 1;
                pair
            };
        output.push(bind_sparse_register_entry_pair(
            even_entry, odd_entry, row, col, challenge,
        ));
    }
}

fn bind_sparse_register_entry_pair<F: Field>(
    even: Option<&SparseRegisterEntry<F>>,
    odd: Option<&SparseRegisterEntry<F>>,
    row: usize,
    col: u8,
    challenge: F,
) -> SparseRegisterEntry<F> {
    let value = sparse_register_entry_eval(even, odd, challenge);
    let (prev_val, next_val) = match (even, odd) {
        (Some(even), Some(odd)) => (even.prev_val, odd.next_val),
        (Some(even), None) => (even.prev_val, even.next_val),
        (None, Some(odd)) => (odd.prev_val, odd.next_val),
        (None, None) => (0, 0),
    };
    SparseRegisterEntry {
        row,
        col,
        val: value.val,
        prev_val,
        next_val,
        read_ra: value.read_ra,
        rd_wa: value.rd_wa,
    }
}

fn sparse_register_read_eval<F: Field>(
    reads: &[(usize, usize)],
    cycle_eq: &[F],
    address_eq: &[F],
) -> F {
    if reads.len() >= DENSE_BIND_PAR_THRESHOLD {
        reads
            .par_iter()
            .map(|&(row, col)| {
                debug_assert!(row < cycle_eq.len());
                debug_assert!(col < address_eq.len());
                cycle_eq[row] * address_eq[col]
            })
            .sum()
    } else {
        reads
            .iter()
            .map(|&(row, col)| {
                debug_assert!(row < cycle_eq.len());
                debug_assert!(col < address_eq.len());
                cycle_eq[row] * address_eq[col]
            })
            .sum()
    }
}

fn linear_eval<F: Field>(low: F, high: F, x: F) -> F {
    low + x * (high - low)
}

fn gruen_cubic_poly<F: Field>(
    target: F,
    q_constant: F,
    q_quadratic_coeff: F,
    previous_claim: F,
) -> UnivariatePoly<F> {
    let eq_eval_1 = target;
    let eq_eval_0 = F::one() - target;
    let eq_delta = eq_eval_1 - eq_eval_0;
    let eq_eval_2 = eq_eval_1 + eq_delta;
    let eq_eval_3 = eq_eval_2 + eq_delta;
    let cubic_eval_0 = eq_eval_0 * q_constant;
    let cubic_eval_1 = previous_claim - cubic_eval_0;
    let quadratic_eval_1 = cubic_eval_1 / eq_eval_1;
    let e_times_2 = q_quadratic_coeff + q_quadratic_coeff;
    let quadratic_eval_2 = quadratic_eval_1 + quadratic_eval_1 - q_constant + e_times_2;
    let quadratic_eval_3 = quadratic_eval_2 + quadratic_eval_1 - q_constant + e_times_2 + e_times_2;
    UnivariatePoly::from_evals(&[
        cubic_eval_0,
        cubic_eval_1,
        eq_eval_2 * quadratic_eval_2,
        eq_eval_3 * quadratic_eval_3,
    ])
}

#[tracing::instrument(skip_all, name = "ram_val_check_state")]
fn ram_val_check_state<F: Field>(
    claim: &Stage4SumcheckClaimPlan,
    inputs: &Stage4ProverInputs<'_, F>,
    store: &Stage4ValueStore<F>,
    active_scale: F,
    backend: &'static str,
) -> Result<DenseStage4State<F>, Stage4KernelError> {
    let witness = inputs.ram.ok_or(Stage4KernelError::MissingKernelInput {
        kernel: "jolt_stage4_batched",
        input: "ram",
    })?;
    let expected_len = witness.ram_k.checked_mul(witness.trace_len).ok_or(
        Stage4KernelError::InvalidInputLength {
            input: "stage4.ram",
            expected: usize::MAX,
            actual: witness.ram_k,
        },
    )?;
    require_operand_count(
        "stage4.ram.RamInc",
        witness.trace_len,
        witness.ram_inc.len(),
    )?;

    let ram_val_point = store.point("stage4.input.stage2.RamVal")?;
    let trace_rounds = log2_exact(witness.trace_len, "stage4.ram.trace_len")?;
    let ram_rounds = log2_exact(witness.ram_k, "stage4.ram_k")?;
    require_operand_count("stage4.ram_val_check.input", trace_rounds, claim.num_rounds)?;
    require_operand_count(
        "stage4.input.stage2.RamVal",
        ram_rounds + trace_rounds,
        ram_val_point.len(),
    )?;
    let (address_point, cycle_point) = ram_val_point.split_at(ram_rounds);

    let gamma = store.scalar("stage4.ram_val_check.gamma")?;

    let terms = vec![DenseTerm {
        coefficient: F::one(),
        factors: vec![0, 1, 2],
    }];
    let outputs = vec![
        FactorOutput {
            name: "stage4.ram_val_check.eval.RamRa",
            oracle: "RamRa",
            factor: 1,
        },
        FactorOutput {
            name: "stage4.ram_val_check.eval.RamInc",
            oracle: "RamInc",
            factor: 2,
        },
    ];

    #[cfg(feature = "cuda")]
    if backend == "cuda" {
        if let Some(cuda) = cuda_ram_val_check_from_resident_addr(
            address_point,
            cycle_point,
            gamma,
            witness.ram_inc,
            active_scale,
            witness.trace_len,
        ) {
            return Ok(DenseStage4State::from_host_and_cuda(
                vec![Vec::new(), Vec::new(), Vec::new()],
                terms,
                outputs,
                active_scale,
                Some(cuda),
            ));
        }
    }

    let address_eq = EqPolynomial::<F>::evals(address_point, None);
    let ram_ra_at_address = ram_ra_at_address(witness, &address_eq, expected_len)?;

    #[cfg(feature = "cuda")]
    if backend == "cuda" {
        if let Some(cuda) = build_cuda_ram_val_check_state(
            cycle_point,
            gamma,
            &ram_ra_at_address,
            witness.ram_inc,
            active_scale,
        ) {
            return Ok(DenseStage4State::from_host_and_cuda(
                vec![Vec::new(), ram_ra_at_address, witness.ram_inc.to_vec()],
                terms,
                outputs,
                active_scale,
                Some(cuda),
            ));
        }
    }

    let mut lt_plus_gamma = lt_evals_big_endian(cycle_point);
    require_operand_count(
        "stage4.ram_val_check.lt",
        witness.trace_len,
        lt_plus_gamma.len(),
    )?;
    lt_plus_gamma
        .par_iter_mut()
        .for_each(|value| *value += gamma);

    Ok(DenseStage4State::new_with_backend(
        vec![lt_plus_gamma, ram_ra_at_address, witness.ram_inc.to_vec()],
        terms,
        outputs,
        active_scale,
        backend,
    ))
}

fn ram_ra_at_address<F: Field>(
    witness: Stage4RamWitness<'_, F>,
    address_eq: &[F],
    dense_len: usize,
) -> Result<Vec<F>, Stage4KernelError> {
    if !witness.ram_ra.is_empty() {
        require_operand_count("stage4.ram.RamRa", dense_len, witness.ram_ra.len())?;
        let mut output = vec![F::zero(); witness.trace_len];
        for (address, &weight) in address_eq.iter().enumerate() {
            let base = address * witness.trace_len;
            for (cycle, output) in output.iter_mut().enumerate() {
                *output += weight * witness.ram_ra[base + cycle];
            }
        }
        return Ok(output);
    }

    let Some(write_address_indices) = witness.write_address_indices else {
        return Err(Stage4KernelError::MissingKernelInput {
            kernel: "jolt_stage4_batched",
            input: "ram_ra",
        });
    };
    require_operand_count(
        "stage4.ram.write_address_indices",
        witness.trace_len,
        write_address_indices.len(),
    )?;
    write_address_indices
        .iter()
        .map(|address| match address {
            Some(address) => {
                address_eq
                    .get(*address)
                    .copied()
                    .ok_or(Stage4KernelError::InvalidInputLength {
                        input: "stage4.ram.write_address_indices",
                        expected: address_eq.len(),
                        actual: address + 1,
                    })
            }
            None => Ok(F::zero()),
        })
        .collect()
}

fn round_poly_from_dense_terms<F: Field>(
    factors: &[Vec<F>],
    terms: &[DenseTerm<F>],
    active_scale: F,
    relation: Stage4Relation,
) -> Result<UnivariatePoly<F>, Stage4KernelError> {
    let half = factors.first().map_or(0, |factor| factor.len() / 2);
    for term in terms {
        if term.factors.len() > 3 {
            return Err(Stage4KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "stage4 dense term exceeds degree bound",
            });
        }
        if term.factors.iter().any(|factor| *factor >= factors.len()) {
            return Err(Stage4KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "stage4 dense term references missing factor",
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
            .map(FieldAccumulator::reduce)
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
            [] => coefficients[0].acc_add(term.coefficient),
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

fn expected_batched_output_claim<F: Field>(
    context: Stage4KernelContext<'_>,
    store: &Stage4ValueStore<F>,
    evals: &[Stage4NamedEval<F>],
    point: &[F],
    batching_coeffs: &[F],
) -> Result<F, Stage4KernelError> {
    let mut expected = F::zero();
    for (claim, &coefficient) in context.batch_claims()?.iter().zip(batching_coeffs) {
        let instance = context
            .program
            .instance_results
            .iter()
            .find(|instance| {
                instance.claim == claim.symbol && instance.source == context.driver.symbol
            })
            .ok_or(Stage4KernelError::MissingClaim {
                batch: context.batch.symbol,
                claim: claim.symbol,
            })?;
        let local_point = point
            .get(instance.round_offset..instance.round_offset + instance.num_rounds)
            .ok_or(Stage4KernelError::InvalidInputLength {
                input: instance.symbol,
                expected: instance.round_offset + instance.num_rounds,
                actual: point.len(),
            })?;
        let claim_value = match Stage4Relation::from_symbol(instance.relation).ok_or(
            Stage4KernelError::UnknownRelation {
                relation: instance.relation,
            },
        )? {
            Stage4Relation::RegistersReadWrite => {
                expected_registers_read_write(store, evals, local_point)?
            }
            Stage4Relation::RamValCheck => expected_ram_val_check(store, evals, local_point)?,
            relation @ Stage4Relation::Batched => {
                return Err(Stage4KernelError::KernelNotImplemented {
                    abi: relation.symbol(),
                })
            }
        };
        expected += coefficient * claim_value;
    }
    Ok(expected)
}

fn expected_registers_read_write<F: Field>(
    store: &Stage4ValueStore<F>,
    evals: &[Stage4NamedEval<F>],
    local_point: &[F],
) -> Result<F, Stage4KernelError> {
    let trace_point = store.point("stage4.input.stage3.registers.RdWriteValue")?;
    let r_cycle = normalize_stage4_registers_rw_cycle_point(
        local_point,
        trace_point.len(),
        "stage4.registers_read_write.instance",
    )?;
    let eq_eval = EqPolynomial::<F>::mle(&r_cycle, trace_point);
    let registers_val = eval_by_name(evals, "stage4.registers_read_write.eval.RegistersVal")?;
    let rs1_ra = eval_by_name(evals, "stage4.registers_read_write.eval.Rs1Ra")?;
    let rs2_ra = eval_by_name(evals, "stage4.registers_read_write.eval.Rs2Ra")?;
    let rd_wa = eval_by_name(evals, "stage4.registers_read_write.eval.RdWa")?;
    let rd_inc = eval_by_name(evals, "stage4.registers_read_write.eval.RdInc")?;
    let gamma = store.scalar("stage4.registers_read_write.gamma")?;
    Ok(eq_eval
        * (rd_wa * (registers_val + rd_inc)
            + gamma * (rs1_ra * registers_val + gamma * rs2_ra * registers_val)))
}

fn expected_ram_val_check<F: Field>(
    store: &Stage4ValueStore<F>,
    evals: &[Stage4NamedEval<F>],
    local_point: &[F],
) -> Result<F, Stage4KernelError> {
    let ram_val_point = store.point("stage4.input.stage2.RamVal")?;
    let r_cycle_prime = reverse_slice(local_point);
    let r_cycle = suffix_point(
        ram_val_point,
        r_cycle_prime.len(),
        "stage4.input.stage2.RamVal",
    )?;
    let lt_eval = lt_polynomial_eval(&r_cycle_prime, r_cycle);
    let gamma = store.scalar("stage4.ram_val_check.gamma")?;
    let ram_ra = eval_by_name(evals, "stage4.ram_val_check.eval.RamRa")?;
    let ram_inc = eval_by_name(evals, "stage4.ram_val_check.eval.RamInc")?;
    Ok(ram_inc * ram_ra * (lt_eval + gamma))
}

fn eval_by_name<F: Field>(
    evals: &[Stage4NamedEval<F>],
    name: &'static str,
) -> Result<F, Stage4KernelError> {
    evals
        .iter()
        .find(|eval| eval.name == name)
        .map(|eval| eval.value)
        .ok_or(Stage4KernelError::MissingValue { symbol: name })
}

fn named_eval<F: Field>(name: &'static str, oracle: &'static str, value: F) -> Stage4NamedEval<F> {
    Stage4NamedEval {
        name,
        oracle,
        value,
    }
}

fn claim_relation(
    program: &'static Stage4CpuProgramPlan,
    claim: &Stage4SumcheckClaimPlan,
) -> Result<Stage4Relation, Stage4KernelError> {
    if let Some(relation) = claim.relation {
        return Stage4Relation::from_symbol(relation)
            .ok_or(Stage4KernelError::UnknownRelation { relation });
    }
    let kernel_symbol = claim.kernel.ok_or(Stage4KernelError::MissingKernel {
        driver: claim.symbol,
        kernel: "<missing>",
    })?;
    let kernel = find_kernel(program, kernel_symbol).ok_or(Stage4KernelError::MissingKernel {
        driver: claim.symbol,
        kernel: kernel_symbol,
    })?;
    kernel.relation_kind()
}

fn instance_round_offset(
    program: &'static Stage4CpuProgramPlan,
    driver: &'static str,
    claim: &'static str,
) -> Result<usize, Stage4KernelError> {
    program
        .instance_results
        .iter()
        .find(|instance| instance.source == driver && instance.claim == claim)
        .map(|instance| instance.round_offset)
        .ok_or(Stage4KernelError::MissingClaim {
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
    program: &'static Stage4CpuProgramPlan,
    store: &mut Stage4ValueStore<F>,
    transcript: &mut T,
    evals: &[Stage4NamedEval<F>],
) -> Result<Vec<Stage4OpeningClaimValue<F>>, Stage4KernelError>
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
    let mut seen = seed_stage4_opening_aliases(store, program);
    for batch in program.opening_batches {
        for symbol in batch.claim_operands {
            let claim =
                find_opening_claim(program, symbol).ok_or(Stage4KernelError::MissingClaim {
                    batch: batch.symbol,
                    claim: symbol,
                })?;
            let point = store.point(claim.point_source)?.to_vec();
            let value = store.scalar(claim.eval_source)?;
            let duplicate = has_seen_opening(&seen, claim.claim_kind, claim.oracle, &point);
            if !duplicate {
                append_labeled_scalar(transcript, "opening_claim", &value);
                seen.push((claim.claim_kind, claim.oracle, point.clone()));
            }
            opening_claims.push(Stage4OpeningClaimValue {
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

fn seed_stage4_opening_aliases<F: Field>(
    store: &Stage4ValueStore<F>,
    program: &'static Stage4CpuProgramPlan,
) -> Vec<(&'static str, &'static str, Vec<F>)> {
    program
        .opening_inputs
        .iter()
        .filter_map(|input| {
            store
                .try_point(input.symbol)
                .map(|point| (input.claim_kind, input.oracle, point.to_vec()))
        })
        .collect()
}

fn has_seen_opening<F: Field>(
    seen: &[(&'static str, &'static str, Vec<F>)],
    claim_kind: &'static str,
    oracle: &'static str,
    point: &[F],
) -> bool {
    seen.iter().any(|(seen_kind, seen_oracle, seen_point)| {
        *seen_kind == claim_kind && *seen_oracle == oracle && seen_point.as_slice() == point
    })
}

fn find_opening_claim<'a>(
    program: &'a Stage4CpuProgramPlan,
    symbol: &str,
) -> Option<&'a Stage4OpeningClaimPlan> {
    program
        .opening_claims
        .iter()
        .find(|claim| claim.symbol == symbol)
}

fn normalize_stage4_registers_rw_point<F: Field>(
    program: &'static Stage4CpuProgramPlan,
    driver: &'static str,
    point: &[F],
) -> Result<Vec<F>, Stage4KernelError> {
    let driver_plan =
        find_driver(program, driver).ok_or(Stage4KernelError::MissingDriver { driver })?;
    if driver_plan.round_schedule.len() != 2 {
        return Err(Stage4KernelError::InvalidProof {
            driver,
            reason: "stage4 registers point normalization requires [cycle, address] schedule",
        });
    }
    let cycle_rounds = driver_plan.round_schedule[0];
    let address_rounds = driver_plan.round_schedule[1];
    if point.len() != cycle_rounds + address_rounds {
        return Err(Stage4KernelError::InvalidInputLength {
            input: "stage4.registers_read_write.instance",
            expected: cycle_rounds + address_rounds,
            actual: point.len(),
        });
    }
    let (cycle, address) = point.split_at(cycle_rounds);
    Ok(address
        .iter()
        .rev()
        .copied()
        .chain(cycle.iter().rev().copied())
        .collect())
}

fn normalize_stage4_registers_rw_cycle_point<F: Field>(
    point: &[F],
    cycle_rounds: usize,
    input: &'static str,
) -> Result<Vec<F>, Stage4KernelError> {
    let cycle = point
        .get(..cycle_rounds)
        .filter(|cycle| cycle.len() == cycle_rounds)
        .ok_or(Stage4KernelError::InvalidInputLength {
            input,
            expected: cycle_rounds,
            actual: point.len(),
        })?;
    Ok(cycle.iter().rev().copied().collect())
}

fn suffix_point<'a, F: Field>(
    point: &'a [F],
    length: usize,
    input: &'static str,
) -> Result<&'a [F], Stage4KernelError> {
    point
        .get(point.len().saturating_sub(length)..)
        .filter(|suffix| suffix.len() == length)
        .ok_or(Stage4KernelError::InvalidInputLength {
            input,
            expected: length,
            actual: point.len(),
        })
}

fn reverse_slice<F: Field>(values: &[F]) -> Vec<F> {
    values.iter().rev().copied().collect()
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

#[cfg(test)]
fn boolean_point_from_index<F: Field>(index: usize, bits: usize) -> Vec<F> {
    (0..bits)
        .rev()
        .map(|bit| F::from_bool(((index >> bit) & 1) == 1))
        .collect()
}

fn log2_exact(value: usize, input: &'static str) -> Result<usize, Stage4KernelError> {
    if value.is_power_of_two() {
        Ok(value.ilog2() as usize)
    } else {
        Err(Stage4KernelError::InvalidInputLength {
            input,
            expected: value.next_power_of_two(),
            actual: value,
        })
    }
}

fn verify_count(
    artifact: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), Stage4KernelError> {
    if expected == actual {
        Ok(())
    } else {
        Err(Stage4KernelError::PlanCountMismatch {
            artifact,
            expected,
            actual,
        })
    }
}

fn find_squeeze(
    program: &'static Stage4CpuProgramPlan,
    symbol: &str,
) -> Option<&'static Stage4TranscriptSqueezePlan> {
    program
        .transcript_squeezes
        .iter()
        .find(|squeeze| squeeze.symbol == symbol)
}

fn find_absorb_bytes(
    program: &'static Stage4CpuProgramPlan,
    symbol: &str,
) -> Option<&'static Stage4TranscriptAbsorbBytesPlan> {
    program
        .transcript_absorb_bytes
        .iter()
        .find(|absorb| absorb.symbol == symbol)
}

fn find_driver(
    program: &'static Stage4CpuProgramPlan,
    symbol: &str,
) -> Option<&'static Stage4SumcheckDriverPlan> {
    program
        .drivers
        .iter()
        .find(|driver| driver.symbol == symbol)
}

fn find_kernel(
    program: &'static Stage4CpuProgramPlan,
    symbol: &str,
) -> Option<&'static Stage4KernelPlan> {
    program
        .kernels
        .iter()
        .find(|kernel| kernel.symbol == symbol)
}

fn find_batch(
    program: &'static Stage4CpuProgramPlan,
    symbol: &str,
) -> Option<&'static Stage4SumcheckBatchPlan> {
    program.batches.iter().find(|batch| batch.symbol == symbol)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage4KernelError {
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
        expected: Stage4ExecutionMode,
        actual: Stage4ExecutionMode,
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

impl Display for Stage4KernelError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingClaim { batch, claim } => {
                write!(
                    formatter,
                    "stage4 batch @{batch} references missing claim @{claim}"
                )
            }
            Self::MissingValue { symbol } => {
                write!(formatter, "stage4 value @{symbol} is not available")
            }
            Self::MissingDriver { driver } => {
                write!(formatter, "stage4 driver @{driver} is not available")
            }
            Self::MissingKernel { driver, kernel } => {
                write!(
                    formatter,
                    "stage4 driver @{driver} references missing kernel @{kernel}"
                )
            }
            Self::MissingBatch { driver, batch } => {
                write!(
                    formatter,
                    "stage4 driver @{driver} references missing batch @{batch}"
                )
            }
            Self::UnknownRelation { relation } => {
                write!(formatter, "stage4 relation @{relation} is not registered")
            }
            Self::UnknownKernelAbi { abi } => {
                write!(formatter, "stage4 kernel ABI `{abi}` is not registered")
            }
            Self::PlanCountMismatch {
                artifact,
                expected,
                actual,
            } => write!(
                formatter,
                "stage4 plan @{artifact} count mismatch: expected {expected}, got {actual}"
            ),
            Self::InvalidInputLength {
                input,
                expected,
                actual,
            } => write!(
                formatter,
                "stage4 input `{input}` length mismatch: expected {expected}, got {actual}"
            ),
            Self::UnsupportedFieldExpr { symbol, formula } => write!(
                formatter,
                "stage4 field expr @{symbol} uses unsupported formula `{formula}`"
            ),
            Self::KernelNotImplemented { abi } => {
                write!(formatter, "stage4 kernel ABI `{abi}` is not implemented")
            }
            Self::WrongExecutorMode {
                driver,
                expected,
                actual,
            } => write!(
                formatter,
                "stage4 driver @{driver} ran with {actual:?} executor path, expected {expected:?}"
            ),
            Self::MissingProof { driver } => {
                write!(
                    formatter,
                    "stage4 verifier missing proof for driver @{driver}"
                )
            }
            Self::MissingKernelInput { kernel, input } => {
                write!(
                    formatter,
                    "stage4 kernel `{kernel}` missing input `{input}`"
                )
            }
            Self::InvalidProgramStep { symbol, kind } => {
                write!(
                    formatter,
                    "stage4 program step @{symbol} has unsupported kind `{kind}`"
                )
            }
            Self::InvalidProof { driver, reason } => {
                write!(
                    formatter,
                    "stage4 proof for driver @{driver} is invalid: {reason}"
                )
            }
        }
    }
}

impl Error for Stage4KernelError {}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "stage4 kernel tests fail fast")]
mod tests {
    use super::*;
    use jolt_field::Fr;
    use jolt_transcript::Blake2bTranscript;

    #[test]
    fn lt_evals_big_endian_matches_pointwise() {
        let point = frs(&[3, 5, 7]);
        let table = lt_evals_big_endian(&point);

        assert_eq!(table.len(), 8);
        for (index, value) in table.iter().enumerate() {
            let bits = boolean_point_from_index::<Fr>(index, point.len());
            assert_eq!(*value, lt_polynomial_eval(&bits, &point));
        }
    }

    #[test]
    fn stage4_batched_kernel_proves_and_verifies_synthetic_witness() {
        let program = synthetic_stage4_program();
        let data = SyntheticStage4Data::new();
        let opening_inputs = data.opening_inputs();
        let prover_inputs = Stage4ProverInputs::new(&opening_inputs)
            .with_registers(data.registers_witness())
            .with_ram(data.ram_witness());
        let mut prover = Stage4ProverKernelExecutor::new(prover_inputs);
        let mut prover_transcript = Blake2bTranscript::<Fr>::new(b"stage4_test");

        let artifacts = execute_stage4_program(
            program,
            Stage4ExecutionMode::Prover,
            &mut prover,
            &mut prover_transcript,
        )
        .expect("stage4 prover succeeds");

        assert_eq!(artifacts.sumchecks.len(), 1);
        assert_eq!(artifacts.sumchecks[0].proof.round_polynomials.len(), 3);
        let proof = Stage4Proof::from(artifacts);
        let mut verifier = Stage4VerifierKernelExecutor::new(&proof, &opening_inputs);
        let mut verifier_transcript = Blake2bTranscript::<Fr>::new(b"stage4_test");
        let verified = execute_stage4_program(
            program,
            Stage4ExecutionMode::Verifier,
            &mut verifier,
            &mut verifier_transcript,
        )
        .expect("stage4 verifier accepts prover proof");

        assert_eq!(verified.sumchecks.len(), 1);
        assert_eq!(prover_transcript.state(), verifier_transcript.state());
    }

    #[test]
    fn stage4_batched_kernel_proves_with_sparse_ram_addresses() {
        let program = synthetic_stage4_program();
        let data = SyntheticStage4Data::new();
        let opening_inputs = data.opening_inputs();
        let prover_inputs = Stage4ProverInputs::new(&opening_inputs)
            .with_registers(data.registers_witness())
            .with_ram(data.sparse_ram_witness());
        let mut prover = Stage4ProverKernelExecutor::new(prover_inputs);
        let mut prover_transcript = Blake2bTranscript::<Fr>::new(b"stage4_test");

        let artifacts = execute_stage4_program(
            program,
            Stage4ExecutionMode::Prover,
            &mut prover,
            &mut prover_transcript,
        )
        .expect("stage4 prover succeeds with sparse RAM addresses");

        let proof = Stage4Proof::from(artifacts);
        let mut verifier = Stage4VerifierKernelExecutor::new(&proof, &opening_inputs);
        let mut verifier_transcript = Blake2bTranscript::<Fr>::new(b"stage4_test");
        let verified = execute_stage4_program(
            program,
            Stage4ExecutionMode::Verifier,
            &mut verifier,
            &mut verifier_transcript,
        )
        .expect("stage4 verifier accepts sparse RAM proof");

        assert_eq!(verified.sumchecks.len(), 1);
        assert_eq!(prover_transcript.state(), verifier_transcript.state());
    }

    #[test]
    fn stage4_batched_kernel_proves_with_sparse_register_accesses() {
        let program = synthetic_stage4_program();
        let data = SyntheticStage4Data::new();
        let opening_inputs = data.opening_inputs();
        let prover_inputs = Stage4ProverInputs::new(&opening_inputs)
            .with_registers(data.sparse_registers_witness())
            .with_ram(data.sparse_ram_witness());
        let mut prover = Stage4ProverKernelExecutor::new(prover_inputs);
        let mut prover_transcript = Blake2bTranscript::<Fr>::new(b"stage4_test");

        let artifacts = execute_stage4_program(
            program,
            Stage4ExecutionMode::Prover,
            &mut prover,
            &mut prover_transcript,
        )
        .expect("stage4 prover succeeds with sparse register accesses");

        let proof = Stage4Proof::from(artifacts);
        let mut verifier = Stage4VerifierKernelExecutor::new(&proof, &opening_inputs);
        let mut verifier_transcript = Blake2bTranscript::<Fr>::new(b"stage4_test");
        let verified = execute_stage4_program(
            program,
            Stage4ExecutionMode::Verifier,
            &mut verifier,
            &mut verifier_transcript,
        )
        .expect("stage4 verifier accepts sparse register proof");

        assert_eq!(verified.sumchecks.len(), 1);
        assert_eq!(prover_transcript.state(), verifier_transcript.state());
    }

    #[test]
    fn stage4_batched_kernel_rejects_tampered_eval() {
        let program = synthetic_stage4_program();
        let data = SyntheticStage4Data::new();
        let opening_inputs = data.opening_inputs();
        let mut prover = Stage4ProverKernelExecutor::new(
            Stage4ProverInputs::new(&opening_inputs)
                .with_registers(data.registers_witness())
                .with_ram(data.ram_witness()),
        );
        let mut prover_transcript = Blake2bTranscript::<Fr>::new(b"stage4_test");
        let mut proof = Stage4Proof::from(
            execute_stage4_program(
                program,
                Stage4ExecutionMode::Prover,
                &mut prover,
                &mut prover_transcript,
            )
            .expect("stage4 prover succeeds"),
        );
        proof.sumchecks[0].evals[0].value += Fr::from_u64(1);

        let mut verifier = Stage4VerifierKernelExecutor::new(&proof, &opening_inputs);
        let mut verifier_transcript = Blake2bTranscript::<Fr>::new(b"stage4_test");
        let error = execute_stage4_program(
            program,
            Stage4ExecutionMode::Verifier,
            &mut verifier,
            &mut verifier_transcript,
        )
        .expect_err("tampered proof is rejected");

        assert!(matches!(error, Stage4KernelError::InvalidProof { .. }));
    }

    #[test]
    fn sparse_trace_witness_from_accesses_groups_stage4_and_stage2_accesses() {
        let register_accesses = [
            Stage4RegisterAccess {
                rd: Some(Stage4RegisterWrite {
                    address: 2,
                    pre_value: 5,
                    post_value: 9,
                }),
                ..Stage4RegisterAccess::default()
            },
            Stage4RegisterAccess::default(),
        ];
        let ram_accesses = [
            Stage2RamAccess {
                remapped_address: Some(7),
                read_value: 11,
                write_value: 3,
            },
            Stage2RamAccess::noop(),
        ];

        let witness =
            stage4_5_sparse_trace_witness_from_accesses::<Fr>(&register_accesses, &ram_accesses);

        assert_eq!(witness.rd_inc, vec![Fr::from_u64(4), Fr::from_u64(0)]);
        assert_eq!(witness.rd_write_addresses, vec![Some(2), None]);
        assert_eq!(witness.ram_addresses, vec![Some(7), None]);
        assert_eq!(witness.ram_inc, vec![-Fr::from_u64(8), Fr::from_u64(0)]);
    }

    #[derive(Clone)]
    struct SyntheticStage4Data {
        registers_val: Vec<Fr>,
        rs1_ra: Vec<Fr>,
        rs2_ra: Vec<Fr>,
        rd_wa: Vec<Fr>,
        register_accesses: Vec<Stage4RegisterAccess>,
        rd_inc: Vec<Fr>,
        ram_ra: Vec<Fr>,
        ram_write_addresses: Vec<Option<usize>>,
        ram_inc: Vec<Fr>,
    }

    impl SyntheticStage4Data {
        fn new() -> Self {
            Self {
                registers_val: frs(&[10, 12, 12, 15, 20, 20, 21, 21]),
                rs1_ra: frs(&[1, 0, 1, 0, 0, 1, 0, 1]),
                rs2_ra: frs(&[0, 1, 0, 1, 1, 0, 1, 0]),
                rd_wa: frs(&[1, 0, 1, 0, 0, 1, 0, 1]),
                register_accesses: synthetic_register_accesses(),
                rd_inc: frs(&[2, 1, 3, 4]),
                ram_ra: frs(&[1, 0, 1, 0, 0, 1, 0, 1]),
                ram_write_addresses: vec![Some(0), Some(1), Some(0), Some(1)],
                ram_inc: frs(&[5, 7, 0, 2]),
            }
        }

        fn registers_witness(&self) -> Stage4RegistersWitness<'_, Fr> {
            Stage4RegistersWitness {
                register_count: 2,
                trace_len: 4,
                registers_val: &self.registers_val,
                rs1_ra: &self.rs1_ra,
                rs2_ra: &self.rs2_ra,
                rd_wa: &self.rd_wa,
                accesses: None,
                rd_inc: &self.rd_inc,
            }
        }

        fn sparse_registers_witness(&self) -> Stage4RegistersWitness<'_, Fr> {
            Stage4RegistersWitness {
                register_count: 2,
                trace_len: 4,
                registers_val: &[],
                rs1_ra: &[],
                rs2_ra: &[],
                rd_wa: &[],
                accesses: Some(&self.register_accesses),
                rd_inc: &self.rd_inc,
            }
        }

        fn ram_witness(&self) -> Stage4RamWitness<'_, Fr> {
            Stage4RamWitness {
                ram_k: 2,
                trace_len: 4,
                ram_ra: &self.ram_ra,
                write_address_indices: None,
                ram_inc: &self.ram_inc,
            }
        }

        fn sparse_ram_witness(&self) -> Stage4RamWitness<'_, Fr> {
            Stage4RamWitness {
                ram_k: 2,
                trace_len: 4,
                ram_ra: &[],
                write_address_indices: Some(&self.ram_write_addresses),
                ram_inc: &self.ram_inc,
            }
        }

        fn opening_inputs(&self) -> Vec<Stage4OpeningInputValue<Fr>> {
            let trace_point = frs(&[5, 7]);
            let ram_address_point = frs(&[11]);
            let ram_cycle_point = frs(&[13, 17]);
            let ram_val_point = [ram_address_point.as_slice(), ram_cycle_point.as_slice()].concat();
            let rd_write_values = self.rd_write_values();
            let rs1_values = self.read_values(&self.rs1_ra);
            let rs2_values = self.read_values(&self.rs2_ra);
            let ram_init = ram_initial_eval(&ram_address_point);
            let ram_ra_at_address = self.ram_ra_at_address(&ram_address_point);
            let ram_val_delta = ram_val_delta(&ram_ra_at_address, &self.ram_inc, &ram_cycle_point);
            let ram_final_delta = ram_final_delta(&ram_ra_at_address, &self.ram_inc);
            vec![
                opening(
                    "stage4.input.stage3.registers.RdWriteValue",
                    &trace_point,
                    mle_eval(&rd_write_values, &trace_point),
                ),
                opening(
                    "stage4.input.stage3.registers.Rs1Value",
                    &trace_point,
                    mle_eval(&rs1_values, &trace_point),
                ),
                opening(
                    "stage4.input.stage3.registers.Rs2Value",
                    &trace_point,
                    mle_eval(&rs2_values, &trace_point),
                ),
                opening(
                    "stage4.input.stage3.instruction.Rs1Value",
                    &trace_point,
                    mle_eval(&rs1_values, &trace_point),
                ),
                opening(
                    "stage4.input.stage3.instruction.Rs2Value",
                    &trace_point,
                    mle_eval(&rs2_values, &trace_point),
                ),
                opening(
                    "stage4.input.stage2.RamVal",
                    &ram_val_point,
                    ram_init + ram_val_delta,
                ),
                opening(
                    "stage4.input.stage2.RamValFinal",
                    &ram_address_point,
                    ram_init + ram_final_delta,
                ),
                opening(
                    "stage4.input.initial_ram.RamValInit",
                    &ram_address_point,
                    ram_init,
                ),
            ]
        }

        fn rd_write_values(&self) -> Vec<Fr> {
            (0..4)
                .map(|cycle| {
                    (0..2)
                        .map(|address| {
                            let index = address * 4 + cycle;
                            self.rd_wa[index] * (self.registers_val[index] + self.rd_inc[cycle])
                        })
                        .sum()
                })
                .collect()
        }

        fn read_values(&self, read_address: &[Fr]) -> Vec<Fr> {
            (0..4)
                .map(|cycle| {
                    (0..2)
                        .map(|address| {
                            let index = address * 4 + cycle;
                            read_address[index] * self.registers_val[index]
                        })
                        .sum()
                })
                .collect()
        }

        fn ram_ra_at_address(&self, address_point: &[Fr]) -> Vec<Fr> {
            let address_eq = EqPolynomial::<Fr>::evals(address_point, None);
            (0..4)
                .map(|cycle| {
                    (0..2)
                        .map(|address| address_eq[address] * self.ram_ra[address * 4 + cycle])
                        .sum()
                })
                .collect()
        }
    }

    fn synthetic_stage4_program() -> &'static Stage4CpuProgramPlan {
        Box::leak(Box::new(Stage4CpuProgramPlan {
            role: "prover",
            params: Stage4Params {
                field: "bn254_fr",
                pcs: "dory",
                transcript: "blake2b_transcript",
            },
            steps: leak_slice(vec![
                Stage4ProgramStepPlan {
                    kind: "transcript_squeeze",
                    symbol: "stage4.registers_read_write.gamma",
                },
                Stage4ProgramStepPlan {
                    kind: "transcript_absorb_bytes",
                    symbol: "stage4.ram_val_check.domain_separator",
                },
                Stage4ProgramStepPlan {
                    kind: "transcript_squeeze",
                    symbol: "stage4.ram_val_check.gamma",
                },
                Stage4ProgramStepPlan {
                    kind: "sumcheck_driver",
                    symbol: "stage4.sumcheck",
                },
            ]),
            transcript_squeezes: leak_slice(vec![
                Stage4TranscriptSqueezePlan {
                    symbol: "stage4.registers_read_write.gamma",
                    label: "registers_read_write_gamma",
                    kind: "challenge_scalar",
                    count: 1,
                },
                Stage4TranscriptSqueezePlan {
                    symbol: "stage4.ram_val_check.gamma",
                    label: "ram_val_check_gamma",
                    kind: "challenge_scalar",
                    count: 1,
                },
            ]),
            transcript_absorb_bytes: leak_slice(vec![Stage4TranscriptAbsorbBytesPlan {
                symbol: "stage4.ram_val_check.domain_separator",
                label: "ram_val_check_gamma",
                payload: "",
            }]),
            opening_inputs: leak_slice(stage4_opening_input_plans()),
            field_constants: &[],
            field_exprs: leak_slice(stage4_field_exprs()),
            kernels: leak_slice(vec![
                kernel(
                    "jolt.cpu.stage4.registers_read_write",
                    "jolt.stage4.registers_read_write",
                    "jolt_stage4_registers_read_write",
                ),
                kernel(
                    "jolt.cpu.stage4.ram_val_check",
                    "jolt.stage4.ram_val_check",
                    "jolt_stage4_ram_val_check",
                ),
                kernel(
                    "jolt.cpu.stage4.batched",
                    "jolt.stage4.batched",
                    "jolt_stage4_batched",
                ),
            ]),
            claims: leak_slice(vec![
                Stage4SumcheckClaimPlan {
                    symbol: "stage4.registers_read_write.input",
                    stage: "stage4",
                    domain: "jolt.stage4_registers_rw_domain",
                    num_rounds: 3,
                    degree: 3,
                    claim: "stage4.registers_read_write.weighted_values",
                    kernel: Some("jolt.cpu.stage4.registers_read_write"),
                    relation: None,
                    claim_value: "stage4.registers_read_write.claim_expr",
                    input_openings: leak_slice(vec![
                        "stage4.input.stage3.registers.RdWriteValue",
                        "stage4.input.stage3.registers.Rs1Value",
                        "stage4.input.stage3.registers.Rs2Value",
                    ]),
                },
                Stage4SumcheckClaimPlan {
                    symbol: "stage4.ram_val_check.input",
                    stage: "stage4",
                    domain: "jolt.trace_domain",
                    num_rounds: 2,
                    degree: 3,
                    claim: "stage4.ram_val_check.weighted_values",
                    kernel: Some("jolt.cpu.stage4.ram_val_check"),
                    relation: None,
                    claim_value: "stage4.ram_val_check.claim_expr",
                    input_openings: leak_slice(vec![
                        "stage4.input.stage2.RamVal",
                        "stage4.input.stage2.RamValFinal",
                        "stage4.input.initial_ram.RamValInit",
                    ]),
                },
            ]),
            batches: leak_slice(vec![Stage4SumcheckBatchPlan {
                symbol: "stage4.batch",
                stage: "stage4",
                proof_slot: "stage4.sumcheck",
                policy: "jolt_core_stage4_aligned",
                count: 2,
                ordered_claims: leak_slice(vec![
                    "stage4.registers_read_write.input",
                    "stage4.ram_val_check.input",
                ]),
                claim_operands: leak_slice(vec![
                    "stage4.registers_read_write.input",
                    "stage4.ram_val_check.input",
                ]),
                claim_label: "sumcheck_claim",
                round_label: "sumcheck_poly",
                round_schedule: leak_slice(vec![2, 1]),
            }]),
            drivers: leak_slice(vec![Stage4SumcheckDriverPlan {
                symbol: "stage4.sumcheck",
                stage: "stage4",
                proof_slot: "stage4.sumcheck",
                kernel: Some("jolt.cpu.stage4.batched"),
                relation: None,
                batch: "stage4.batch",
                policy: "jolt_core_stage4_aligned",
                round_schedule: leak_slice(vec![2, 1]),
                claim_label: "sumcheck_claim",
                round_label: "sumcheck_poly",
                num_rounds: 3,
                degree: 3,
            }]),
            instance_results: leak_slice(vec![
                Stage4SumcheckInstanceResultPlan {
                    symbol: "stage4.registers_read_write.instance",
                    source: "stage4.sumcheck",
                    claim: "stage4.registers_read_write.input",
                    relation: "jolt.stage4.registers_read_write",
                    index: 0,
                    point_arity: 3,
                    num_rounds: 3,
                    round_offset: 0,
                    point_order: "stage4_registers_rw",
                    degree: 3,
                },
                Stage4SumcheckInstanceResultPlan {
                    symbol: "stage4.ram_val_check.instance",
                    source: "stage4.sumcheck",
                    claim: "stage4.ram_val_check.input",
                    relation: "jolt.stage4.ram_val_check",
                    index: 1,
                    point_arity: 2,
                    num_rounds: 2,
                    round_offset: 1,
                    point_order: "reverse",
                    degree: 3,
                },
            ]),
            evals: leak_slice(stage4_eval_plans()),
            point_slices: leak_slice(vec![
                Stage4PointSlicePlan {
                    symbol: "stage4.registers_read_write.point.RdInc",
                    source: "stage4.registers_read_write.instance",
                    offset: 1,
                    length: 2,
                    input: "stage4.registers_read_write.instance",
                },
                Stage4PointSlicePlan {
                    symbol: "stage4.ram_val_check.point.RamAddress",
                    source: "stage4.input.stage2.RamVal",
                    offset: 0,
                    length: 1,
                    input: "stage4.input.stage2.RamVal",
                },
            ]),
            point_concats: leak_slice(vec![Stage4PointConcatPlan {
                symbol: "stage4.ram_val_check.point.RamRa",
                layout: "address_then_cycle",
                arity: 3,
                inputs: leak_slice(vec![
                    "stage4.ram_val_check.point.RamAddress",
                    "stage4.ram_val_check.instance",
                ]),
            }]),
            opening_claims: leak_slice(stage4_opening_claim_plans()),
            opening_equalities: leak_slice(vec![
                Stage4OpeningClaimEqualityPlan {
                    symbol: "stage4.registers.rs1_claim_consistency",
                    mode: "point_and_eval",
                    lhs: "stage4.input.stage3.registers.Rs1Value",
                    rhs: "stage4.input.stage3.instruction.Rs1Value",
                },
                Stage4OpeningClaimEqualityPlan {
                    symbol: "stage4.registers.rs2_claim_consistency",
                    mode: "point_and_eval",
                    lhs: "stage4.input.stage3.registers.Rs2Value",
                    rhs: "stage4.input.stage3.instruction.Rs2Value",
                },
            ]),
            opening_batches: leak_slice(vec![Stage4OpeningBatchPlan {
                symbol: "stage4.openings",
                stage: "stage4",
                proof_slot: "stage4.openings",
                policy: "jolt_stage4_output_order",
                count: 7,
                ordered_claims: leak_slice(
                    stage4_opening_claim_plans()
                        .iter()
                        .map(|claim| claim.symbol)
                        .collect(),
                ),
                claim_operands: leak_slice(
                    stage4_opening_claim_plans()
                        .iter()
                        .map(|claim| claim.symbol)
                        .collect(),
                ),
            }]),
        }))
    }

    fn stage4_field_exprs() -> Vec<Stage4FieldExprPlan> {
        vec![
            field_expr(
                "stage4.registers_read_write.gamma2",
                "field.pow:2",
                vec!["stage4.registers_read_write.gamma"],
            ),
            field_expr(
                "stage4.registers_read_write.term.Rs1Value",
                "field.mul",
                vec![
                    "stage4.registers_read_write.gamma",
                    "stage4.input.stage3.registers.Rs1Value",
                ],
            ),
            field_expr(
                "stage4.registers_read_write.term.Rs2Value",
                "field.mul",
                vec![
                    "stage4.registers_read_write.gamma2",
                    "stage4.input.stage3.registers.Rs2Value",
                ],
            ),
            field_expr(
                "stage4.registers_read_write.partial.RdWriteValueRs1Value",
                "field.add",
                vec![
                    "stage4.input.stage3.registers.RdWriteValue",
                    "stage4.registers_read_write.term.Rs1Value",
                ],
            ),
            field_expr(
                "stage4.registers_read_write.claim_expr",
                "field.add",
                vec![
                    "stage4.registers_read_write.partial.RdWriteValueRs1Value",
                    "stage4.registers_read_write.term.Rs2Value",
                ],
            ),
            field_expr(
                "stage4.ram_val_check.delta.RamVal",
                "field.sub",
                vec![
                    "stage4.input.stage2.RamVal",
                    "stage4.input.initial_ram.RamValInit",
                ],
            ),
            field_expr(
                "stage4.ram_val_check.delta.RamValFinal",
                "field.sub",
                vec![
                    "stage4.input.stage2.RamValFinal",
                    "stage4.input.initial_ram.RamValInit",
                ],
            ),
            field_expr(
                "stage4.ram_val_check.term.RamValFinal",
                "field.mul",
                vec![
                    "stage4.ram_val_check.gamma",
                    "stage4.ram_val_check.delta.RamValFinal",
                ],
            ),
            field_expr(
                "stage4.ram_val_check.claim_expr",
                "field.add",
                vec![
                    "stage4.ram_val_check.delta.RamVal",
                    "stage4.ram_val_check.term.RamValFinal",
                ],
            ),
        ]
    }

    fn stage4_opening_input_plans() -> Vec<Stage4OpeningInputPlan> {
        vec![
            opening_input_plan(
                "stage4.input.stage3.registers.RdWriteValue",
                "RdWriteValue",
                2,
            ),
            opening_input_plan("stage4.input.stage3.registers.Rs1Value", "Rs1Value", 2),
            opening_input_plan("stage4.input.stage3.registers.Rs2Value", "Rs2Value", 2),
            opening_input_plan("stage4.input.stage3.instruction.Rs1Value", "Rs1Value", 2),
            opening_input_plan("stage4.input.stage3.instruction.Rs2Value", "Rs2Value", 2),
            opening_input_plan("stage4.input.stage2.RamVal", "RamVal", 3),
            opening_input_plan("stage4.input.stage2.RamValFinal", "RamValFinal", 1),
            opening_input_plan("stage4.input.initial_ram.RamValInit", "RamValInit", 1),
        ]
    }

    fn stage4_eval_plans() -> Vec<Stage4SumcheckEvalPlan> {
        vec![
            eval(
                "stage4.registers_read_write.eval.RegistersVal",
                "RegistersVal",
                0,
            ),
            eval("stage4.registers_read_write.eval.Rs1Ra", "Rs1Ra", 1),
            eval("stage4.registers_read_write.eval.Rs2Ra", "Rs2Ra", 2),
            eval("stage4.registers_read_write.eval.RdWa", "RdWa", 3),
            eval("stage4.registers_read_write.eval.RdInc", "RdInc", 4),
            eval("stage4.ram_val_check.eval.RamRa", "RamRa", 0),
            eval("stage4.ram_val_check.eval.RamInc", "RamInc", 1),
        ]
    }

    fn stage4_opening_claim_plans() -> Vec<Stage4OpeningClaimPlan> {
        vec![
            opening_claim(
                "stage4.registers_read_write.opening.RegistersVal",
                "RegistersVal",
                "stage4.registers_read_write.instance",
                "stage4.registers_read_write.eval.RegistersVal",
                3,
                "virtual",
            ),
            opening_claim(
                "stage4.registers_read_write.opening.Rs1Ra",
                "Rs1Ra",
                "stage4.registers_read_write.instance",
                "stage4.registers_read_write.eval.Rs1Ra",
                3,
                "virtual",
            ),
            opening_claim(
                "stage4.registers_read_write.opening.Rs2Ra",
                "Rs2Ra",
                "stage4.registers_read_write.instance",
                "stage4.registers_read_write.eval.Rs2Ra",
                3,
                "virtual",
            ),
            opening_claim(
                "stage4.registers_read_write.opening.RdWa",
                "RdWa",
                "stage4.registers_read_write.instance",
                "stage4.registers_read_write.eval.RdWa",
                3,
                "virtual",
            ),
            opening_claim(
                "stage4.registers_read_write.opening.RdInc",
                "RdInc",
                "stage4.registers_read_write.point.RdInc",
                "stage4.registers_read_write.eval.RdInc",
                2,
                "committed",
            ),
            opening_claim(
                "stage4.ram_val_check.opening.RamRa",
                "RamRa",
                "stage4.ram_val_check.point.RamRa",
                "stage4.ram_val_check.eval.RamRa",
                3,
                "virtual",
            ),
            opening_claim(
                "stage4.ram_val_check.opening.RamInc",
                "RamInc",
                "stage4.ram_val_check.instance",
                "stage4.ram_val_check.eval.RamInc",
                2,
                "committed",
            ),
        ]
    }

    fn field_expr(
        symbol: &'static str,
        formula: &'static str,
        operands: Vec<&'static str>,
    ) -> Stage4FieldExprPlan {
        let operands = leak_slice(operands);
        Stage4FieldExprPlan {
            symbol,
            kind: "op",
            formula,
            operand_names: operands,
            operands,
        }
    }

    fn kernel(symbol: &'static str, relation: &'static str, abi: &'static str) -> Stage4KernelPlan {
        Stage4KernelPlan {
            symbol,
            relation,
            kind: "sumcheck",
            backend: "cpu",
            abi,
        }
    }

    fn opening_input_plan(
        symbol: &'static str,
        oracle: &'static str,
        point_arity: usize,
    ) -> Stage4OpeningInputPlan {
        Stage4OpeningInputPlan {
            symbol,
            source_stage: "synthetic",
            source_claim: symbol,
            oracle,
            domain: "synthetic",
            point_arity,
            claim_kind: "virtual",
        }
    }

    fn eval(symbol: &'static str, oracle: &'static str, index: usize) -> Stage4SumcheckEvalPlan {
        Stage4SumcheckEvalPlan {
            symbol,
            source: "stage4.sumcheck",
            name: symbol,
            index,
            oracle,
        }
    }

    fn opening_claim(
        symbol: &'static str,
        oracle: &'static str,
        point_source: &'static str,
        eval_source: &'static str,
        point_arity: usize,
        claim_kind: &'static str,
    ) -> Stage4OpeningClaimPlan {
        Stage4OpeningClaimPlan {
            symbol,
            oracle,
            domain: "synthetic",
            point_arity,
            claim_kind,
            point_source,
            eval_source,
        }
    }

    fn opening(symbol: &'static str, point: &[Fr], eval: Fr) -> Stage4OpeningInputValue<Fr> {
        Stage4OpeningInputValue {
            symbol,
            point: point.to_vec(),
            eval,
        }
    }

    fn mle_eval(values: &[Fr], point: &[Fr]) -> Fr {
        EqPolynomial::<Fr>::evals(point, None)
            .iter()
            .zip(values)
            .map(|(&weight, &value)| weight * value)
            .sum()
    }

    fn ram_initial_eval(address_point: &[Fr]) -> Fr {
        let initial = frs(&[11, 13]);
        mle_eval(&initial, address_point)
    }

    fn ram_val_delta(ram_ra_at_address: &[Fr], ram_inc: &[Fr], cycle_point: &[Fr]) -> Fr {
        (0..4)
            .map(|cycle| {
                let cycle_bits = boolean_point_from_index::<Fr>(cycle, 2);
                ram_inc[cycle]
                    * ram_ra_at_address[cycle]
                    * lt_polynomial_eval(&cycle_bits, cycle_point)
            })
            .sum()
    }

    fn ram_final_delta(ram_ra_at_address: &[Fr], ram_inc: &[Fr]) -> Fr {
        ram_ra_at_address
            .iter()
            .zip(ram_inc)
            .map(|(&ra, &inc)| ra * inc)
            .sum()
    }

    fn synthetic_register_accesses() -> Vec<Stage4RegisterAccess> {
        vec![
            Stage4RegisterAccess {
                rs1: Some(Stage4RegisterRead {
                    address: 0,
                    value: 10,
                }),
                rs2: Some(Stage4RegisterRead {
                    address: 1,
                    value: 20,
                }),
                rd: Some(Stage4RegisterWrite {
                    address: 0,
                    pre_value: 10,
                    post_value: 12,
                }),
            },
            Stage4RegisterAccess {
                rs1: Some(Stage4RegisterRead {
                    address: 1,
                    value: 20,
                }),
                rs2: Some(Stage4RegisterRead {
                    address: 0,
                    value: 12,
                }),
                rd: Some(Stage4RegisterWrite {
                    address: 1,
                    pre_value: 20,
                    post_value: 21,
                }),
            },
            Stage4RegisterAccess {
                rs1: Some(Stage4RegisterRead {
                    address: 0,
                    value: 12,
                }),
                rs2: Some(Stage4RegisterRead {
                    address: 1,
                    value: 21,
                }),
                rd: Some(Stage4RegisterWrite {
                    address: 0,
                    pre_value: 12,
                    post_value: 15,
                }),
            },
            Stage4RegisterAccess {
                rs1: Some(Stage4RegisterRead {
                    address: 1,
                    value: 21,
                }),
                rs2: Some(Stage4RegisterRead {
                    address: 0,
                    value: 15,
                }),
                rd: Some(Stage4RegisterWrite {
                    address: 1,
                    pre_value: 21,
                    post_value: 25,
                }),
            },
        ]
    }

    fn frs(values: &[u64]) -> Vec<Fr> {
        values.iter().copied().map(Fr::from_u64).collect()
    }

    fn leak_slice<T>(values: Vec<T>) -> &'static [T] {
        Box::leak(values.into_boxed_slice())
    }
}
