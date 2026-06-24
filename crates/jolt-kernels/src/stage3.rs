//! Stage 3 coarse-kernel ABI used by Bolt-generated Jolt prover code.

#![expect(
    clippy::too_many_arguments,
    reason = "kernel constructors mirror generated staged protocol inputs"
)]

use std::error::Error;
use std::fmt::{self, Display, Formatter};

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
use jolt_field::Fr;

use crate::dense::DENSE_BIND_PAR_THRESHOLD;
use crate::split_eq::SplitEqState;
use jolt_field::{Field, FieldAccumulator};
use jolt_poly::{EqPlusOnePolynomial, EqPlusOnePrefixSuffix, EqPolynomial, UnivariatePoly};
use jolt_sumcheck::SumcheckProof;
use jolt_transcript::{Label, LabelWithCount, Transcript};
use rayon::prelude::*;

fn trace_stage3_inner_spans() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("JOLT_STAGE3_TRACE_INSTANCES").is_some())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage3ExecutionMode {
    Prover,
    Verifier,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage3Relation {
    SpartanShift,
    InstructionInput,
    RegistersClaimReduction,
    Batched,
}

impl Stage3Relation {
    pub fn from_symbol(symbol: &str) -> Option<Self> {
        match symbol {
            "jolt.stage3.spartan_shift" => Some(Self::SpartanShift),
            "jolt.stage3.instruction_input" => Some(Self::InstructionInput),
            "jolt.stage3.registers_claim_reduction" => Some(Self::RegistersClaimReduction),
            "jolt.stage3.batched" => Some(Self::Batched),
            _ => None,
        }
    }

    pub fn symbol(self) -> &'static str {
        match self {
            Self::SpartanShift => "jolt.stage3.spartan_shift",
            Self::InstructionInput => "jolt.stage3.instruction_input",
            Self::RegistersClaimReduction => "jolt.stage3.registers_claim_reduction",
            Self::Batched => "jolt.stage3.batched",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage3KernelAbi {
    SpartanShift,
    InstructionInput,
    RegistersClaimReduction,
    Batched,
}

impl Stage3KernelAbi {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "jolt_stage3_spartan_shift" => Some(Self::SpartanShift),
            "jolt_stage3_instruction_input" => Some(Self::InstructionInput),
            "jolt_stage3_registers_claim_reduction" => Some(Self::RegistersClaimReduction),
            "jolt_stage3_batched" => Some(Self::Batched),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::SpartanShift => "jolt_stage3_spartan_shift",
            Self::InstructionInput => "jolt_stage3_instruction_input",
            Self::RegistersClaimReduction => "jolt_stage3_registers_claim_reduction",
            Self::Batched => "jolt_stage3_batched",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage3Params {
    pub field: &'static str,
    pub pcs: &'static str,
    pub transcript: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage3KernelPlan {
    pub symbol: &'static str,
    pub relation: &'static str,
    pub kind: &'static str,
    pub backend: &'static str,
    pub abi: &'static str,
}

impl Stage3KernelPlan {
    pub fn relation_kind(&self) -> Result<Stage3Relation, Stage3KernelError> {
        Stage3Relation::from_symbol(self.relation).ok_or(Stage3KernelError::UnknownRelation {
            relation: self.relation,
        })
    }

    pub fn abi_kind(&self) -> Result<Stage3KernelAbi, Stage3KernelError> {
        Stage3KernelAbi::from_name(self.abi)
            .ok_or(Stage3KernelError::UnknownKernelAbi { abi: self.abi })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage3TranscriptSqueezePlan {
    pub symbol: &'static str,
    pub label: &'static str,
    pub kind: &'static str,
    pub count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage3ProgramStepPlan {
    pub kind: &'static str,
    pub symbol: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage3OpeningInputPlan {
    pub symbol: &'static str,
    pub source_stage: &'static str,
    pub source_claim: &'static str,
    pub oracle: &'static str,
    pub domain: &'static str,
    pub point_arity: usize,
    pub claim_kind: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage3FieldConstantPlan {
    pub symbol: &'static str,
    pub field: &'static str,
    pub value: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage3FieldExprPlan {
    pub symbol: &'static str,
    pub kind: &'static str,
    pub formula: &'static str,
    pub operand_names: &'static [&'static str],
    pub operands: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage3SumcheckClaimPlan {
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
pub struct Stage3SumcheckBatchPlan {
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
pub struct Stage3SumcheckDriverPlan {
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
pub struct Stage3SumcheckInstanceResultPlan {
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
pub struct Stage3SumcheckEvalPlan {
    pub symbol: &'static str,
    pub source: &'static str,
    pub name: &'static str,
    pub index: usize,
    pub oracle: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage3PointSlicePlan {
    pub symbol: &'static str,
    pub source: &'static str,
    pub offset: usize,
    pub length: usize,
    pub input: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage3PointConcatPlan {
    pub symbol: &'static str,
    pub layout: &'static str,
    pub arity: usize,
    pub inputs: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage3OpeningClaimPlan {
    pub symbol: &'static str,
    pub oracle: &'static str,
    pub domain: &'static str,
    pub point_arity: usize,
    pub claim_kind: &'static str,
    pub point_source: &'static str,
    pub eval_source: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage3OpeningClaimEqualityPlan {
    pub symbol: &'static str,
    pub mode: &'static str,
    pub lhs: &'static str,
    pub rhs: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage3OpeningBatchPlan {
    pub symbol: &'static str,
    pub stage: &'static str,
    pub proof_slot: &'static str,
    pub policy: &'static str,
    pub count: usize,
    pub ordered_claims: &'static [&'static str],
    pub claim_operands: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage3CpuProgramPlan {
    pub params: Stage3Params,
    pub steps: &'static [Stage3ProgramStepPlan],
    pub transcript_squeezes: &'static [Stage3TranscriptSqueezePlan],
    pub opening_inputs: &'static [Stage3OpeningInputPlan],
    pub field_constants: &'static [Stage3FieldConstantPlan],
    pub field_exprs: &'static [Stage3FieldExprPlan],
    pub kernels: &'static [Stage3KernelPlan],
    pub claims: &'static [Stage3SumcheckClaimPlan],
    pub batches: &'static [Stage3SumcheckBatchPlan],
    pub drivers: &'static [Stage3SumcheckDriverPlan],
    pub instance_results: &'static [Stage3SumcheckInstanceResultPlan],
    pub evals: &'static [Stage3SumcheckEvalPlan],
    pub point_slices: &'static [Stage3PointSlicePlan],
    pub point_concats: &'static [Stage3PointConcatPlan],
    pub opening_claims: &'static [Stage3OpeningClaimPlan],
    pub opening_equalities: &'static [Stage3OpeningClaimEqualityPlan],
    pub opening_batches: &'static [Stage3OpeningBatchPlan],
}

impl Stage3CpuProgramPlan {
    pub fn kernel(&self, symbol: &str) -> Option<&Stage3KernelPlan> {
        find_kernel(self, symbol)
    }

    pub fn batch(&self, symbol: &str) -> Option<&Stage3SumcheckBatchPlan> {
        find_batch(self, symbol)
    }

    pub fn claim(&self, symbol: &str) -> Option<&Stage3SumcheckClaimPlan> {
        self.claims.iter().find(|claim| claim.symbol == symbol)
    }

    pub fn instance_results_for_driver(
        &self,
        driver: &'static str,
    ) -> impl Iterator<Item = &Stage3SumcheckInstanceResultPlan> {
        self.instance_results
            .iter()
            .filter(move |instance| instance.source == driver)
    }

    pub fn evals_for_driver(
        &self,
        driver: &'static str,
    ) -> impl Iterator<Item = &Stage3SumcheckEvalPlan> {
        self.evals.iter().filter(move |eval| eval.source == driver)
    }
}

#[derive(Clone, Debug)]
pub struct Stage3NamedEval<F: Field> {
    pub name: &'static str,
    pub oracle: &'static str,
    pub value: F,
}

#[derive(Clone, Debug)]
pub struct Stage3SumcheckOutput<F: Field> {
    pub driver: &'static str,
    pub point: Vec<F>,
    pub evals: Vec<Stage3NamedEval<F>>,
    pub opening_claims: Vec<Stage3OpeningClaimValue<F>>,
    pub proof: SumcheckProof<F>,
}

#[derive(Clone, Debug)]
pub struct Stage3ChallengeVector<F: Field> {
    pub symbol: &'static str,
    pub values: Vec<F>,
}

#[derive(Clone, Debug)]
pub struct Stage3OpeningClaimValue<F: Field> {
    pub symbol: &'static str,
    pub oracle: &'static str,
    pub domain: &'static str,
    pub claim_kind: &'static str,
    pub point: Vec<F>,
    pub eval: F,
}

#[derive(Clone, Debug)]
pub struct Stage3ExecutionArtifacts<F: Field> {
    pub challenge_vectors: Vec<Stage3ChallengeVector<F>>,
    pub sumchecks: Vec<Stage3SumcheckOutput<F>>,
    pub opening_claims: Vec<Stage3OpeningClaimValue<F>>,
    pub opening_batches: Vec<&'static Stage3OpeningBatchPlan>,
}

impl<F: Field> Default for Stage3ExecutionArtifacts<F> {
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
pub struct Stage3Proof<F: Field> {
    pub sumchecks: Vec<Stage3SumcheckOutput<F>>,
}

impl<F: Field> From<Stage3ExecutionArtifacts<F>> for Stage3Proof<F> {
    fn from(artifacts: Stage3ExecutionArtifacts<F>) -> Self {
        Self {
            sumchecks: artifacts.sumchecks,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Stage3ScalarValue<F: Field> {
    pub symbol: &'static str,
    pub value: F,
}

#[derive(Clone, Debug)]
pub struct Stage3PointValue<F: Field> {
    pub symbol: &'static str,
    pub point: Vec<F>,
}

#[derive(Clone, Debug)]
pub struct Stage3OpeningInputValue<F: Field> {
    pub symbol: &'static str,
    pub point: Vec<F>,
    pub eval: F,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage3Cycle {
    pub unexpanded_pc: u64,
    pub pc: u64,
    pub is_virtual: bool,
    pub is_first_in_sequence: bool,
    pub is_noop: bool,
    pub left_operand_is_rs1: bool,
    pub rs1_value: u64,
    pub left_operand_is_pc: bool,
    pub right_operand_is_rs2: bool,
    pub rs2_value: u64,
    pub right_operand_is_imm: bool,
    pub imm: i128,
    pub rd_write_value: u64,
}

impl Stage3Cycle {
    pub fn padding() -> Self {
        Self {
            unexpanded_pc: 0,
            pc: 0,
            is_virtual: false,
            is_first_in_sequence: false,
            is_noop: true,
            left_operand_is_rs1: false,
            rs1_value: 0,
            left_operand_is_pc: false,
            right_operand_is_rs2: false,
            rs2_value: 0,
            right_operand_is_imm: false,
            imm: 0,
            rd_write_value: 0,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Stage3ValueStore<F: Field> {
    scalars: Vec<Stage3ScalarValue<F>>,
    points: Vec<Stage3PointValue<F>>,
}

impl<F: Field> Stage3ValueStore<F> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_opening_inputs(inputs: &[Stage3OpeningInputValue<F>]) -> Self {
        let mut store = Self::new();
        store.insert_opening_inputs(inputs);
        store
    }

    pub fn insert_opening_inputs(&mut self, inputs: &[Stage3OpeningInputValue<F>]) {
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
            self.scalars.push(Stage3ScalarValue { symbol, value });
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
            self.points.push(Stage3PointValue { symbol, point });
        }
    }

    pub fn try_scalar(&self, symbol: &str) -> Option<F> {
        self.scalars
            .iter()
            .find(|value| value.symbol == symbol)
            .map(|value| value.value)
    }

    pub fn scalar(&self, symbol: &'static str) -> Result<F, Stage3KernelError> {
        self.try_scalar(symbol)
            .ok_or(Stage3KernelError::MissingValue { symbol })
    }

    pub fn try_point(&self, symbol: &str) -> Option<&[F]> {
        self.points
            .iter()
            .find(|value| value.symbol == symbol)
            .map(|value| value.point.as_slice())
    }

    pub fn point(&self, symbol: &'static str) -> Result<&[F], Stage3KernelError> {
        self.try_point(symbol)
            .ok_or(Stage3KernelError::MissingValue { symbol })
    }

    pub fn seed_constants(
        &mut self,
        program: &'static Stage3CpuProgramPlan,
    ) -> Result<(), Stage3KernelError> {
        for constant in program.field_constants {
            self.insert_scalar(constant.symbol, F::from_u64(constant.value as u64));
        }
        Ok(())
    }

    pub fn observe_challenge_vector(
        &mut self,
        plan: &'static Stage3TranscriptSqueezePlan,
        values: &[F],
    ) -> Result<(), Stage3KernelError> {
        if matches!(plan.kind, "challenge_scalar" | "scalar") {
            require_operand_count(plan.symbol, 1, values.len())?;
            self.insert_scalar(plan.symbol, values[0]);
        }
        self.insert_point(plan.symbol, values.to_vec());
        Ok(())
    }

    pub fn observe_sumcheck_output(
        &mut self,
        program: &'static Stage3CpuProgramPlan,
        output: &Stage3SumcheckOutput<F>,
    ) -> Result<(), Stage3KernelError> {
        self.observe_sumcheck_values(program, output.driver, &output.point, &output.evals)
    }

    pub fn observe_sumcheck_values(
        &mut self,
        program: &'static Stage3CpuProgramPlan,
        driver: &'static str,
        point: &[F],
        evals: &[Stage3NamedEval<F>],
    ) -> Result<(), Stage3KernelError> {
        self.insert_point(driver, point.to_vec());
        for instance in program.instance_results_for_driver(driver) {
            let end = instance.round_offset + instance.point_arity;
            let mut point = point
                .get(instance.round_offset..end)
                .ok_or(Stage3KernelError::InvalidInputLength {
                    input: instance.symbol,
                    expected: end,
                    actual: point.len(),
                })?
                .to_vec();
            match instance.point_order {
                "as_is" => {}
                "reverse" => point.reverse(),
                _ => {
                    return Err(Stage3KernelError::InvalidProof {
                        driver,
                        reason: "unsupported point order",
                    });
                }
            }
            self.insert_point(instance.symbol, point);
        }
        for eval in program.evals_for_driver(driver) {
            let value = evals
                .iter()
                .find(|value| value.name == eval.name)
                .or_else(|| evals.get(eval.index))
                .ok_or(Stage3KernelError::MissingValue {
                    symbol: eval.symbol,
                })?
                .value;
            self.insert_scalar(eval.symbol, value);
            self.insert_scalar(eval.name, value);
        }
        Ok(())
    }

    pub fn evaluate_available_points(
        &mut self,
        program: &'static Stage3CpuProgramPlan,
    ) -> Result<usize, Stage3KernelError> {
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
                    .ok_or(Stage3KernelError::InvalidInputLength {
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
        program: &'static Stage3CpuProgramPlan,
    ) -> Result<usize, Stage3KernelError> {
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
                self.insert_scalar(expr.symbol, evaluate_stage3_field_expr(expr, &operands)?);
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
        program: &'static Stage3CpuProgramPlan,
    ) -> Result<(), Stage3KernelError> {
        for equality in program.opening_equalities {
            match equality.mode {
                "point_and_eval" => {
                    if self.point(equality.lhs)? != self.point(equality.rhs)?
                        || self.scalar(equality.lhs)? != self.scalar(equality.rhs)?
                    {
                        return Err(Stage3KernelError::InvalidProof {
                            driver: equality.symbol,
                            reason: "opening claim equality failed",
                        });
                    }
                }
                _ => {
                    return Err(Stage3KernelError::InvalidProof {
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
        program: &'static Stage3CpuProgramPlan,
        claim: &Stage3SumcheckClaimPlan,
    ) -> Result<F, Stage3KernelError> {
        let _ = self.evaluate_available_field_exprs(program)?;
        self.scalar(claim.claim_value)
    }

    pub fn batch_claim_values(
        &mut self,
        program: &'static Stage3CpuProgramPlan,
        batch: &Stage3SumcheckBatchPlan,
    ) -> Result<Vec<F>, Stage3KernelError> {
        batch
            .claim_operands
            .iter()
            .map(|symbol| {
                let claim = program
                    .claim(symbol)
                    .ok_or(Stage3KernelError::MissingClaim {
                        batch: batch.symbol,
                        claim: symbol,
                    })?;
                self.claim_value(program, claim)
            })
            .collect()
    }

    fn try_expr_operands(&self, expr: &Stage3FieldExprPlan) -> Option<Vec<F>> {
        expr.operands
            .iter()
            .map(|operand| self.try_scalar(operand))
            .collect()
    }

    fn try_concat_point(&self, concat: &Stage3PointConcatPlan) -> Option<Vec<F>> {
        let mut point = Vec::with_capacity(concat.arity);
        for input in concat.inputs {
            point.extend_from_slice(self.try_point(input)?);
        }
        Some(point)
    }
}

pub fn evaluate_stage3_field_expr<F: Field>(
    expr: &Stage3FieldExprPlan,
    operands: &[F],
) -> Result<F, Stage3KernelError> {
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
                    Stage3KernelError::UnsupportedFieldExpr {
                        symbol: expr.symbol,
                        formula: expr.formula,
                    }
                })?;
                return Ok(pow_field(operands[0], exponent));
            }
            Err(Stage3KernelError::UnsupportedFieldExpr {
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

fn single_operand<F: Field>(symbol: &'static str, operands: &[F]) -> Result<F, Stage3KernelError> {
    require_operand_count(symbol, 1, operands.len())?;
    Ok(operands[0])
}

fn require_operand_count(
    input: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), Stage3KernelError> {
    if expected == actual {
        Ok(())
    } else {
        Err(Stage3KernelError::InvalidInputLength {
            input,
            expected,
            actual,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Stage3KernelContext<'a> {
    pub mode: Stage3ExecutionMode,
    pub program: &'static Stage3CpuProgramPlan,
    pub kernel: &'a Stage3KernelPlan,
    pub batch: &'a Stage3SumcheckBatchPlan,
    pub driver: &'a Stage3SumcheckDriverPlan,
}

impl Stage3KernelContext<'_> {
    pub fn relation_kind(&self) -> Result<Stage3Relation, Stage3KernelError> {
        self.kernel.relation_kind()
    }

    pub fn abi_kind(&self) -> Result<Stage3KernelAbi, Stage3KernelError> {
        self.kernel.abi_kind()
    }

    pub fn batch_claims(&self) -> Result<Vec<&'static Stage3SumcheckClaimPlan>, Stage3KernelError> {
        self.batch
            .claim_operands
            .iter()
            .map(|symbol| {
                self.program
                    .claim(symbol)
                    .ok_or(Stage3KernelError::MissingClaim {
                        batch: self.batch.symbol,
                        claim: symbol,
                    })
            })
            .collect()
    }
}

pub trait Stage3KernelExecutor<F: Field> {
    fn observe_challenge_vector(
        &mut self,
        _plan: &'static Stage3TranscriptSqueezePlan,
        _values: &[F],
    ) -> Result<(), Stage3KernelError> {
        Ok(())
    }

    fn observe_sumcheck_output(
        &mut self,
        _output: &Stage3SumcheckOutput<F>,
    ) -> Result<(), Stage3KernelError> {
        Ok(())
    }

    fn prove_sumcheck<T>(
        &mut self,
        context: Stage3KernelContext<'_>,
        transcript: &mut T,
    ) -> Result<Stage3SumcheckOutput<F>, Stage3KernelError>
    where
        T: Transcript<Challenge = F>;

    fn verify_sumcheck<T>(
        &mut self,
        context: Stage3KernelContext<'_>,
        transcript: &mut T,
    ) -> Result<Stage3SumcheckOutput<F>, Stage3KernelError>
    where
        T: Transcript<Challenge = F>;
}

#[derive(Clone, Debug, Default)]
pub struct UnsupportedStage3KernelExecutor;

impl<F: Field> Stage3KernelExecutor<F> for UnsupportedStage3KernelExecutor {
    fn prove_sumcheck<T>(
        &mut self,
        context: Stage3KernelContext<'_>,
        _transcript: &mut T,
    ) -> Result<Stage3SumcheckOutput<F>, Stage3KernelError>
    where
        T: Transcript<Challenge = F>,
    {
        Err(Stage3KernelError::KernelNotImplemented {
            abi: context.kernel.abi,
        })
    }

    fn verify_sumcheck<T>(
        &mut self,
        context: Stage3KernelContext<'_>,
        _transcript: &mut T,
    ) -> Result<Stage3SumcheckOutput<F>, Stage3KernelError>
    where
        T: Transcript<Challenge = F>,
    {
        Err(Stage3KernelError::KernelNotImplemented {
            abi: context.kernel.abi,
        })
    }
}

#[derive(Clone, Copy)]
pub struct Stage3ProverInputs<'a, F: Field> {
    pub opening_inputs: &'a [Stage3OpeningInputValue<F>],
    pub cycles: Option<&'a [Stage3Cycle]>,
}

impl<'a, F: Field> Stage3ProverInputs<'a, F> {
    pub fn new(opening_inputs: &'a [Stage3OpeningInputValue<F>]) -> Self {
        Self {
            opening_inputs,
            cycles: None,
        }
    }

    pub fn empty() -> Self {
        Self {
            opening_inputs: &[],
            cycles: None,
        }
    }

    pub fn with_cycles(mut self, cycles: &'a [Stage3Cycle]) -> Self {
        self.cycles = Some(cycles);
        self
    }
}

#[derive(Clone)]
pub struct Stage3ProverKernelExecutor<'a, F: Field> {
    pub inputs: Stage3ProverInputs<'a, F>,
    challenge_vectors: Vec<Stage3ChallengeVector<F>>,
    completed_sumchecks: Vec<Stage3SumcheckOutput<F>>,
}

impl<'a, F: Field> Stage3ProverKernelExecutor<'a, F> {
    pub fn new(inputs: Stage3ProverInputs<'a, F>) -> Self {
        Self {
            inputs,
            challenge_vectors: Vec::new(),
            completed_sumchecks: Vec::new(),
        }
    }

    fn value_store(
        &self,
        program: &'static Stage3CpuProgramPlan,
    ) -> Result<Stage3ValueStore<F>, Stage3KernelError> {
        value_store_from_observations(
            program,
            self.inputs.opening_inputs,
            &self.challenge_vectors,
            &self.completed_sumchecks,
        )
    }
}

impl<F: Field> Stage3KernelExecutor<F> for Stage3ProverKernelExecutor<'_, F> {
    fn observe_challenge_vector(
        &mut self,
        plan: &'static Stage3TranscriptSqueezePlan,
        values: &[F],
    ) -> Result<(), Stage3KernelError> {
        self.challenge_vectors.push(Stage3ChallengeVector {
            symbol: plan.symbol,
            values: values.to_vec(),
        });
        Ok(())
    }

    fn observe_sumcheck_output(
        &mut self,
        output: &Stage3SumcheckOutput<F>,
    ) -> Result<(), Stage3KernelError> {
        self.completed_sumchecks.push(output.clone());
        Ok(())
    }

    fn prove_sumcheck<T>(
        &mut self,
        context: Stage3KernelContext<'_>,
        transcript: &mut T,
    ) -> Result<Stage3SumcheckOutput<F>, Stage3KernelError>
    where
        T: Transcript<Challenge = F>,
    {
        prove_stage3_kernel(
            context,
            &self.inputs,
            self.value_store(context.program)?,
            transcript,
        )
    }

    fn verify_sumcheck<T>(
        &mut self,
        context: Stage3KernelContext<'_>,
        _transcript: &mut T,
    ) -> Result<Stage3SumcheckOutput<F>, Stage3KernelError>
    where
        T: Transcript<Challenge = F>,
    {
        Err(Stage3KernelError::WrongExecutorMode {
            driver: context.driver.symbol,
            expected: Stage3ExecutionMode::Prover,
            actual: Stage3ExecutionMode::Verifier,
        })
    }
}

#[derive(Clone)]
pub struct Stage3VerifierKernelExecutor<'a, F: Field> {
    pub proof: &'a Stage3Proof<F>,
    pub opening_inputs: &'a [Stage3OpeningInputValue<F>],
    pub cursor: usize,
    challenge_vectors: Vec<Stage3ChallengeVector<F>>,
    completed_sumchecks: Vec<Stage3SumcheckOutput<F>>,
}

impl<'a, F: Field> Stage3VerifierKernelExecutor<'a, F> {
    pub fn new(
        proof: &'a Stage3Proof<F>,
        opening_inputs: &'a [Stage3OpeningInputValue<F>],
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
        program: &'static Stage3CpuProgramPlan,
    ) -> Result<Stage3ValueStore<F>, Stage3KernelError> {
        value_store_from_observations(
            program,
            self.opening_inputs,
            &self.challenge_vectors,
            &self.completed_sumchecks,
        )
    }
}

impl<F: Field> Stage3KernelExecutor<F> for Stage3VerifierKernelExecutor<'_, F> {
    fn observe_challenge_vector(
        &mut self,
        plan: &'static Stage3TranscriptSqueezePlan,
        values: &[F],
    ) -> Result<(), Stage3KernelError> {
        self.challenge_vectors.push(Stage3ChallengeVector {
            symbol: plan.symbol,
            values: values.to_vec(),
        });
        Ok(())
    }

    fn observe_sumcheck_output(
        &mut self,
        output: &Stage3SumcheckOutput<F>,
    ) -> Result<(), Stage3KernelError> {
        self.completed_sumchecks.push(output.clone());
        Ok(())
    }

    fn prove_sumcheck<T>(
        &mut self,
        context: Stage3KernelContext<'_>,
        _transcript: &mut T,
    ) -> Result<Stage3SumcheckOutput<F>, Stage3KernelError>
    where
        T: Transcript<Challenge = F>,
    {
        Err(Stage3KernelError::WrongExecutorMode {
            driver: context.driver.symbol,
            expected: Stage3ExecutionMode::Verifier,
            actual: Stage3ExecutionMode::Prover,
        })
    }

    fn verify_sumcheck<T>(
        &mut self,
        context: Stage3KernelContext<'_>,
        transcript: &mut T,
    ) -> Result<Stage3SumcheckOutput<F>, Stage3KernelError>
    where
        T: Transcript<Challenge = F>,
    {
        let proof =
            self.proof
                .sumchecks
                .get(self.cursor)
                .ok_or(Stage3KernelError::MissingProof {
                    driver: context.driver.symbol,
                })?;
        self.cursor += 1;
        verify_stage3_kernel(
            context,
            self.value_store(context.program)?,
            proof,
            transcript,
        )
    }
}

fn value_store_from_observations<F: Field>(
    program: &'static Stage3CpuProgramPlan,
    opening_inputs: &[Stage3OpeningInputValue<F>],
    challenge_vectors: &[Stage3ChallengeVector<F>],
    completed_sumchecks: &[Stage3SumcheckOutput<F>],
) -> Result<Stage3ValueStore<F>, Stage3KernelError> {
    let mut store = Stage3ValueStore::with_opening_inputs(opening_inputs);
    store.seed_constants(program)?;
    for challenge in challenge_vectors {
        let plan = program
            .transcript_squeezes
            .iter()
            .find(|plan| plan.symbol == challenge.symbol)
            .ok_or(Stage3KernelError::MissingValue {
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage3KernelError {
    MissingKernel {
        driver: &'static str,
        kernel: &'static str,
    },
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
    UnknownRelation {
        relation: &'static str,
    },
    UnknownKernelAbi {
        abi: &'static str,
    },
    KernelNotImplemented {
        abi: &'static str,
    },
    WrongExecutorMode {
        driver: &'static str,
        expected: Stage3ExecutionMode,
        actual: Stage3ExecutionMode,
    },
    MissingProof {
        driver: &'static str,
    },
    MissingKernelInput {
        kernel: &'static str,
        input: &'static str,
    },
    InvalidProof {
        driver: &'static str,
        reason: &'static str,
    },
}

impl Display for Stage3KernelError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingKernel { driver, kernel } => {
                write!(
                    formatter,
                    "stage3 driver @{driver} references missing kernel @{kernel}"
                )
            }
            Self::MissingBatch { driver, batch } => {
                write!(
                    formatter,
                    "stage3 driver @{driver} references missing batch @{batch}"
                )
            }
            Self::MissingClaim { batch, claim } => {
                write!(
                    formatter,
                    "stage3 batch @{batch} references missing claim @{claim}"
                )
            }
            Self::MissingValue { symbol } => {
                write!(formatter, "stage3 value @{symbol} is not available")
            }
            Self::PlanCountMismatch {
                artifact,
                expected,
                actual,
            } => write!(
                formatter,
                "stage3 plan @{artifact} count mismatch: expected {expected}, got {actual}"
            ),
            Self::InvalidInputLength {
                input,
                expected,
                actual,
            } => write!(
                formatter,
                "stage3 input `{input}` length mismatch: expected {expected}, got {actual}"
            ),
            Self::UnsupportedFieldExpr { symbol, formula } => write!(
                formatter,
                "stage3 field expr @{symbol} uses unsupported formula `{formula}`"
            ),
            Self::UnknownRelation { relation } => {
                write!(formatter, "stage3 relation @{relation} is not registered")
            }
            Self::UnknownKernelAbi { abi } => {
                write!(formatter, "stage3 kernel ABI `{abi}` is not registered")
            }
            Self::KernelNotImplemented { abi } => {
                write!(formatter, "stage3 kernel ABI `{abi}` is not implemented")
            }
            Self::WrongExecutorMode {
                driver,
                expected,
                actual,
            } => write!(
                formatter,
                "stage3 driver @{driver} ran with {actual:?} executor path, expected {expected:?}"
            ),
            Self::MissingProof { driver } => {
                write!(
                    formatter,
                    "stage3 verifier missing proof for driver @{driver}"
                )
            }
            Self::MissingKernelInput { kernel, input } => {
                write!(
                    formatter,
                    "stage3 kernel `{kernel}` missing input `{input}`"
                )
            }
            Self::InvalidProof { driver, reason } => {
                write!(
                    formatter,
                    "stage3 proof for driver @{driver} is invalid: {reason}"
                )
            }
        }
    }
}

impl Error for Stage3KernelError {}

fn prove_stage3_kernel<F, T>(
    context: Stage3KernelContext<'_>,
    inputs: &Stage3ProverInputs<'_, F>,
    store: Stage3ValueStore<F>,
    transcript: &mut T,
) -> Result<Stage3SumcheckOutput<F>, Stage3KernelError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    match context.abi_kind()? {
        Stage3KernelAbi::Batched => prove_batched_stage3(context, inputs, store, transcript),
        abi => Err(Stage3KernelError::KernelNotImplemented { abi: abi.name() }),
    }
}

fn verify_stage3_kernel<F, T>(
    context: Stage3KernelContext<'_>,
    store: Stage3ValueStore<F>,
    proof: &Stage3SumcheckOutput<F>,
    transcript: &mut T,
) -> Result<Stage3SumcheckOutput<F>, Stage3KernelError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    match context.abi_kind()? {
        Stage3KernelAbi::Batched => verify_batched_stage3(context, store, proof, transcript),
        abi => Err(Stage3KernelError::KernelNotImplemented { abi: abi.name() }),
    }
}

#[tracing::instrument(skip_all, name = "Stage3::prove_batched")]
fn prove_batched_stage3<F, T>(
    context: Stage3KernelContext<'_>,
    inputs: &Stage3ProverInputs<'_, F>,
    mut store: Stage3ValueStore<F>,
    transcript: &mut T,
) -> Result<Stage3SumcheckOutput<F>, Stage3KernelError>
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
        .ok_or(Stage3KernelError::InvalidProof {
            driver: context.driver.symbol,
            reason: "field element 2 is not invertible",
        })?;
    let mut instances = Vec::with_capacity(claims.len());
    for (index, claim) in claims.iter().enumerate() {
        let relation = claim_relation(context.program, claim)?;
        let _span = trace_stage3_inner_spans().then(|| {
            tracing::info_span!(
                "Stage3::instance.init",
                relation = relation.symbol(),
                claim = claim.symbol
            )
            .entered()
        });
        instances.push(Stage3BatchedInstance {
            claim,
            relation,
            offset: instance_round_offset(context.program, context.driver.symbol, claim.symbol)?,
            previous_claim: input_claims[index].mul_pow_2(max_rounds - claim.num_rounds),
            state: Stage3ProverInstanceState::new(
                context.program,
                claim,
                inputs,
                &store,
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
                let _span = trace_stage3_inner_spans().then(|| {
                    tracing::info_span!(
                        "Stage3::instance.round_poly",
                        relation = instance.relation.symbol(),
                        claim = instance.claim.symbol,
                        round
                    )
                    .entered()
                });
                instance
                    .state
                    .round_poly(round - instance.offset, instance.previous_claim)?
            } else {
                UnivariatePoly::new(vec![instance.previous_claim * two_inv])
            };
            #[cfg(debug_assertions)]
            {
                if poly.evaluate(F::zero()) + poly.evaluate(F::one()) != instance.previous_claim {
                    return Err(Stage3KernelError::InvalidProof {
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
                return Err(Stage3KernelError::InvalidProof {
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
                let _span = trace_stage3_inner_spans().then(|| {
                    tracing::info_span!(
                        "Stage3::instance.bind",
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
        return Err(Stage3KernelError::InvalidProof {
            driver: context.driver.symbol,
            reason: "batched output claim mismatch",
        });
    }
    store.observe_sumcheck_values(context.program, context.driver.symbol, &point, &evals)?;
    let opening_claims = append_opening_claims(context.program, &mut store, transcript, &evals)?;
    Ok(Stage3SumcheckOutput {
        driver: context.driver.symbol,
        point,
        evals,
        opening_claims,
        proof: SumcheckProof { round_polynomials },
    })
}

fn verify_batched_stage3<F, T>(
    context: Stage3KernelContext<'_>,
    mut store: Stage3ValueStore<F>,
    proof: &Stage3SumcheckOutput<F>,
    transcript: &mut T,
) -> Result<Stage3SumcheckOutput<F>, Stage3KernelError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    if proof.driver != context.driver.symbol {
        return Err(Stage3KernelError::InvalidProof {
            driver: context.driver.symbol,
            reason: "driver symbol mismatch",
        });
    }
    if proof.proof.round_polynomials.len() != context.driver.num_rounds {
        return Err(Stage3KernelError::InvalidProof {
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
            return Err(Stage3KernelError::InvalidProof {
                driver: context.driver.symbol,
                reason: "batched polynomial exceeds degree bound",
            });
        }
        if poly.evaluate(F::zero()) + poly.evaluate(F::one()) != running_claim {
            return Err(Stage3KernelError::InvalidProof {
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
        return Err(Stage3KernelError::InvalidProof {
            driver: context.driver.symbol,
            reason: "batched point mismatch",
        });
    }
    let expected =
        expected_batched_output_claim(context, &store, &proof.evals, &point, &batching_coeffs)?;
    if running_claim != expected {
        return Err(Stage3KernelError::InvalidProof {
            driver: context.driver.symbol,
            reason: "batched output claim mismatch",
        });
    }
    store.observe_sumcheck_values(context.program, context.driver.symbol, &point, &proof.evals)?;
    let opening_claims =
        append_opening_claims(context.program, &mut store, transcript, &proof.evals)?;
    Ok(Stage3SumcheckOutput {
        driver: context.driver.symbol,
        point,
        evals: proof.evals.clone(),
        opening_claims,
        proof: proof.proof.clone(),
    })
}

struct Stage3BatchedInstance<'a, F: Field> {
    claim: &'a Stage3SumcheckClaimPlan,
    relation: Stage3Relation,
    offset: usize,
    previous_claim: F,
    state: Stage3ProverInstanceState<F>,
}

impl<F: Field> Stage3BatchedInstance<'_, F> {
    fn is_active(&self, round: usize) -> bool {
        round >= self.offset && round < self.offset + self.claim.num_rounds
    }
}

enum Stage3ProverInstanceState<F: Field> {
    SpartanShift(Box<SpartanShiftState<F>>),
    SumOfProducts(Box<SumOfProductsState<F>>),
}

impl<F: Field> Stage3ProverInstanceState<F> {
    fn new(
        program: &'static Stage3CpuProgramPlan,
        claim: &Stage3SumcheckClaimPlan,
        inputs: &Stage3ProverInputs<'_, F>,
        store: &Stage3ValueStore<F>,
        backend: &'static str,
    ) -> Result<Self, Stage3KernelError> {
        match claim_relation(program, claim)? {
            Stage3Relation::SpartanShift => {
                return spartan_shift_state(claim, inputs, store)
                    .map(|state| Self::SpartanShift(Box::new(state)));
            }
            Stage3Relation::InstructionInput => {
                instruction_input_state(claim, inputs, store, backend)
            }
            Stage3Relation::RegistersClaimReduction => {
                registers_state(claim, inputs, store, backend)
            }
            relation @ Stage3Relation::Batched => Err(Stage3KernelError::KernelNotImplemented {
                abi: relation.symbol(),
            }),
        }
        .map(|state| Self::SumOfProducts(Box::new(state)))
    }

    fn round_poly(
        &self,
        _round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, Stage3KernelError> {
        match self {
            Self::SpartanShift(state) => Ok(state.round_poly(previous_claim)),
            Self::SumOfProducts(state) => Ok(state.round_poly(previous_claim)),
        }
    }

    fn ingest_challenge(&mut self, challenge: F) {
        match self {
            Self::SpartanShift(state) => state.bind(challenge),
            Self::SumOfProducts(state) => state.bind(challenge),
        }
    }

    fn final_evals(
        &self,
        relation: Stage3Relation,
    ) -> Result<Vec<Stage3NamedEval<F>>, Stage3KernelError> {
        match self {
            Self::SpartanShift(state) => state.final_evals(relation),
            Self::SumOfProducts(state) => state.final_evals(relation),
        }
    }
}

#[derive(Clone)]
struct SpartanShiftState<F: Field> {
    phase: SpartanShiftPhase<F>,
    r_outer: Vec<F>,
    r_product: Vec<F>,
    gamma: F,
    gamma2: F,
    gamma3: F,
    gamma4: F,
    point: Vec<F>,
}

#[derive(Clone)]
enum SpartanShiftPhase<F: Field> {
    Phase1(SpartanShiftPhase1<F>),
    Phase2(SpartanShiftPhase2<F>),
}

#[derive(Clone)]
struct SpartanShiftPhase1<F: Field> {
    prefix_suffix_pairs: Vec<(Vec<F>, Vec<F>)>,
    scratch: Vec<(Vec<F>, Vec<F>)>,
    cycles: Vec<Stage3Cycle>,
}

#[derive(Clone)]
struct SpartanShiftPhase2<F: Field> {
    eq_outer: Vec<F>,
    eq_product: Vec<F>,
    weighted_next_values: Vec<F>,
    not_noop: Vec<F>,
    unexpanded_pc: Vec<F>,
    pc: Vec<F>,
    is_virtual: Vec<F>,
    is_first_in_sequence: Vec<F>,
    is_noop: Vec<F>,
    scratch: Vec<Vec<F>>,
}

struct SumOfProductsState<F: Field> {
    kind: SumOfProductsKind,
    factors: Vec<Vec<F>>,
    factor_scratch: Vec<Vec<F>>,
    split_eq: Option<SplitEqState<F>>,
    terms: Vec<ProductTerm<F>>,
    outputs: Vec<FactorOutput>,
    deferred_outputs: Vec<DeferredOutput<F>>,
    point: Vec<F>,
    #[cfg(feature = "cuda")]
    cuda: Option<cuda::CudaSumOfProductsState>,
}

#[derive(Clone, Copy)]
enum SumOfProductsKind {
    InstructionInput,
    Registers,
}

#[derive(Clone)]
struct ProductTerm<F: Field> {
    coefficient: F,
}

#[derive(Clone, Copy)]
struct FactorOutput {
    name: &'static str,
    oracle: &'static str,
    factor: usize,
}

#[derive(Clone)]
struct DeferredOutput<F: Field> {
    name: &'static str,
    oracle: &'static str,
    values: Vec<F>,
}

impl<F: Field> SumOfProductsState<F> {
    #[cfg_attr(not(feature = "cuda"), expect(unused_variables))]
    fn new(
        kind: SumOfProductsKind,
        factors: Vec<Vec<F>>,
        split_eq: Option<SplitEqState<F>>,
        terms: Vec<ProductTerm<F>>,
        outputs: Vec<FactorOutput>,
        deferred_outputs: Vec<DeferredOutput<F>>,
        backend: &'static str,
        split_point: &[F],
    ) -> Self {
        let factor_scratch = (0..factors.len()).map(|_| Vec::new()).collect();
        #[cfg(feature = "cuda")]
        let cuda = if backend == "cuda" {
            cuda_sum_of_products_state(kind, &factors, &terms, split_point)
        } else {
            None
        };
        Self {
            kind,
            factors,
            factor_scratch,
            split_eq,
            terms,
            outputs,
            deferred_outputs,
            point: Vec::new(),
            #[cfg(feature = "cuda")]
            cuda,
        }
    }

    fn round_poly(&self, previous_claim: F) -> UnivariatePoly<F> {
        #[cfg(feature = "cuda")]
        if let Some(cuda) = &self.cuda {
            if let Some(poly) = cuda_round_poly::<F>(cuda, self.kind, previous_claim) {
                return poly;
            }
        }
        let Some(split_eq) = self.split_eq.as_ref() else {
            std::process::abort();
        };
        match self.kind {
            SumOfProductsKind::InstructionInput => round_poly_from_instruction_input(
                &self.factors,
                &self.terms,
                split_eq,
                previous_claim,
            ),
            SumOfProductsKind::Registers => {
                round_poly_from_registers(&self.factors, &self.terms, split_eq, previous_claim)
            }
        }
    }

    fn bind(&mut self, challenge: F) {
        #[cfg(feature = "cuda")]
        if let Some(cuda) = &mut self.cuda {
            if let Some(challenge_fr) = crate::cuda::into_fr(challenge) {
                if cuda.bind(challenge_fr).is_ok() {
                    self.point.push(challenge);
                    return;
                }
            }
        }
        let half = self.factors.first().map_or(0, |factor| factor.len() / 2);
        if half >= DENSE_BIND_PAR_THRESHOLD {
            self.factors
                .par_iter_mut()
                .zip(self.factor_scratch.par_iter_mut())
                .for_each(|(factor, scratch)| {
                    bind_dense_evals_reuse_serial(factor, scratch, challenge);
                });
        } else {
            for (factor, scratch) in self.factors.iter_mut().zip(&mut self.factor_scratch) {
                bind_dense_evals_reuse_serial(factor, scratch, challenge);
            }
        }
        if let Some(split_eq) = &mut self.split_eq {
            split_eq.bind(challenge);
        }
        self.point.push(challenge);
    }

    fn factor_eval(&self, index: usize, relation: Stage3Relation) -> Result<F, Stage3KernelError> {
        #[cfg(feature = "cuda")]
        if let Some(cuda) = &self.cuda {
            if let Ok(value) = cuda.factor_eval(index) {
                if let Some(value) = crate::cuda::fr_into::<F>(value) {
                    return Ok(value);
                }
            }
        }
        self.factors
            .get(index)
            .and_then(|values| values.first())
            .copied()
            .ok_or(Stage3KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "empty stage3 factor",
            })
    }

    fn final_evals(
        &self,
        relation: Stage3Relation,
    ) -> Result<Vec<Stage3NamedEval<F>>, Stage3KernelError> {
        let mut evals = self
            .outputs
            .iter()
            .map(|output| {
                Ok(named_eval(
                    output.name,
                    output.oracle,
                    self.factor_eval(output.factor, relation)?,
                ))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if !self.deferred_outputs.is_empty() {
            let point = reverse_slice(&self.point);
            let eq = EqPolynomial::<F>::evals(&point, None);
            evals.extend(self.deferred_outputs.iter().map(|output| {
                named_eval(
                    output.name,
                    output.oracle,
                    deferred_output_eval(&output.values, &eq),
                )
            }));
        }
        Ok(evals)
    }
}

#[cfg(feature = "cuda")]
fn cuda_sum_of_products_state<F: Field>(
    kind: SumOfProductsKind,
    factors: &[Vec<F>],
    terms: &[ProductTerm<F>],
    split_point: &[F],
) -> Option<cuda::CudaSumOfProductsState> {
    let fr_factors: Vec<Vec<Fr>> = factors
        .iter()
        .map(|factor| crate::cuda::as_fr_slice(factor).map(<[Fr]>::to_vec))
        .collect::<Option<Vec<_>>>()?;
    let fr_point = crate::cuda::as_fr_slice(split_point)?.to_vec();
    let cuda_kind = match kind {
        SumOfProductsKind::InstructionInput => cuda::CudaGruenKind::InstructionInput {
            gamma: crate::cuda::into_fr(terms[2].coefficient)?,
        },
        SumOfProductsKind::Registers => cuda::CudaGruenKind::Registers {
            gamma: crate::cuda::into_fr(terms[1].coefficient)?,
            gamma2: crate::cuda::into_fr(terms[2].coefficient)?,
        },
    };
    cuda::CudaSumOfProductsState::new(cuda_kind, &fr_factors, &fr_point)
}

#[cfg(feature = "cuda")]
fn cuda_round_poly<F: Field>(
    cuda: &cuda::CudaSumOfProductsState,
    kind: SumOfProductsKind,
    previous_claim: F,
) -> Option<UnivariatePoly<F>> {
    let (q_constant, q_top) = cuda.q_coefficients().ok()?;
    let target: F = crate::cuda::fr_into(cuda.current_target())?;
    let q_constant: F = crate::cuda::fr_into(q_constant)?;
    let poly = match kind {
        SumOfProductsKind::InstructionInput => {
            let q_quadratic: F = crate::cuda::fr_into(q_top)?;
            gruen_cubic_poly(target, q_constant, q_quadratic, previous_claim)
        }
        SumOfProductsKind::Registers => gruen_quadratic_poly(target, q_constant, previous_claim),
    };
    Some(poly)
}

impl<F: Field> SpartanShiftState<F> {
    fn new(
        cycles: &[Stage3Cycle],
        r_outer: &[F],
        r_product: &[F],
        gamma: F,
        gamma2: F,
        gamma3: F,
        gamma4: F,
    ) -> Self {
        Self {
            phase: SpartanShiftPhase::Phase1(SpartanShiftPhase1::new(
                cycles, r_outer, r_product, gamma, gamma2, gamma3, gamma4,
            )),
            r_outer: r_outer.to_vec(),
            r_product: r_product.to_vec(),
            gamma,
            gamma2,
            gamma3,
            gamma4,
            point: Vec::new(),
        }
    }

    fn round_poly(&self, previous_claim: F) -> UnivariatePoly<F> {
        match &self.phase {
            SpartanShiftPhase::Phase1(state) => state.round_poly(),
            SpartanShiftPhase::Phase2(state) => state.round_poly(previous_claim),
        }
    }

    fn bind(&mut self, challenge: F) {
        let transition = match &mut self.phase {
            SpartanShiftPhase::Phase1(state) => {
                if state.should_transition_to_phase2() {
                    true
                } else {
                    state.bind(challenge);
                    false
                }
            }
            SpartanShiftPhase::Phase2(state) => {
                state.bind(challenge);
                false
            }
        };
        self.point.push(challenge);
        if transition {
            let SpartanShiftPhase::Phase1(state) = &self.phase else {
                unreachable!("checked phase before transition");
            };
            let phase2 = SpartanShiftPhase2::new(
                &state.cycles,
                &self.point,
                &self.r_outer,
                &self.r_product,
                self.gamma,
                self.gamma2,
                self.gamma3,
                self.gamma4,
            );
            self.phase = SpartanShiftPhase::Phase2(phase2);
        }
    }

    fn final_evals(
        &self,
        relation: Stage3Relation,
    ) -> Result<Vec<Stage3NamedEval<F>>, Stage3KernelError> {
        let SpartanShiftPhase::Phase2(state) = &self.phase else {
            return Err(Stage3KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "spartan shift did not finish phase 2",
            });
        };
        state.final_evals(relation)
    }
}

impl<F: Field> SpartanShiftPhase1<F> {
    fn new(
        cycles: &[Stage3Cycle],
        r_outer: &[F],
        r_product: &[F],
        gamma: F,
        gamma2: F,
        gamma3: F,
        gamma4: F,
    ) -> Self {
        let outer = EqPlusOnePrefixSuffix::new(r_outer);
        let product = EqPlusOnePrefixSuffix::new(r_product);
        let q_values =
            spartan_shift_phase1_q_values(cycles, &outer, &product, gamma, gamma2, gamma3, gamma4);
        let mut q_outer_0 = Vec::with_capacity(q_values.len());
        let mut q_outer_1 = Vec::with_capacity(q_values.len());
        let mut q_product_0 = Vec::with_capacity(q_values.len());
        let mut q_product_1 = Vec::with_capacity(q_values.len());
        for [outer_0, outer_1, product_0, product_1] in q_values {
            q_outer_0.push(outer_0);
            q_outer_1.push(outer_1);
            q_product_0.push(product_0);
            q_product_1.push(product_1);
        }
        let prefix_suffix_pairs = vec![
            (outer.prefix_0, q_outer_0),
            (outer.prefix_1, q_outer_1),
            (product.prefix_0, q_product_0),
            (product.prefix_1, q_product_1),
        ];
        let scratch = prefix_suffix_pairs
            .iter()
            .map(|_| (Vec::new(), Vec::new()))
            .collect();
        Self {
            prefix_suffix_pairs,
            scratch,
            cycles: cycles.to_vec(),
        }
    }

    fn round_poly(&self) -> UnivariatePoly<F> {
        let half = self.prefix_suffix_pairs[0].0.len() / 2;
        round_poly_from_stage3_coefficients(half, 2, |row, acc| {
            for (prefix, suffix) in &self.prefix_suffix_pairs {
                let (prefix_0, prefix_delta) = linear_pair(prefix, row);
                let (suffix_0, suffix_delta) = linear_pair(suffix, row);
                accumulate_linear_product(
                    acc,
                    F::one(),
                    prefix_0,
                    prefix_delta,
                    suffix_0,
                    suffix_delta,
                );
            }
        })
    }

    fn bind(&mut self, challenge: F) {
        for ((prefix, suffix), (prefix_scratch, suffix_scratch)) in self
            .prefix_suffix_pairs
            .iter_mut()
            .zip(self.scratch.iter_mut())
        {
            bind_dense_evals_reuse_serial(prefix, prefix_scratch, challenge);
            bind_dense_evals_reuse_serial(suffix, suffix_scratch, challenge);
        }
    }

    fn should_transition_to_phase2(&self) -> bool {
        self.prefix_suffix_pairs[0].0.len() == 2
    }
}

impl<F: Field> SpartanShiftPhase2<F> {
    fn new(
        cycles: &[Stage3Cycle],
        low_challenges: &[F],
        r_outer: &[F],
        r_product: &[F],
        gamma: F,
        gamma2: F,
        gamma3: F,
        gamma4: F,
    ) -> Self {
        let low_point = reverse_slice(low_challenges);
        let low_eq = EqPolynomial::<F>::evals(&low_point, None);
        let eq_outer = spartan_shift_phase2_eq_plus_one(r_outer, &low_eq);
        let eq_product = spartan_shift_phase2_eq_plus_one(r_product, &low_eq);
        let (
            unexpanded_pc,
            pc,
            is_virtual,
            is_first_in_sequence,
            is_noop,
            weighted_next_values,
            not_noop,
        ) = spartan_shift_phase2_outputs(cycles, &low_eq, gamma, gamma2, gamma3);
        let not_noop = not_noop
            .iter()
            .map(|&value| gamma4 * value)
            .collect::<Vec<_>>();
        Self {
            eq_outer,
            eq_product,
            weighted_next_values,
            not_noop,
            unexpanded_pc,
            pc,
            is_virtual,
            is_first_in_sequence,
            is_noop,
            scratch: (0..9).map(|_| Vec::new()).collect(),
        }
    }

    fn round_poly(&self, _previous_claim: F) -> UnivariatePoly<F> {
        round_poly_from_stage3_coefficients(self.eq_outer.len() / 2, 2, |row, acc| {
            let (eq_outer_0, eq_outer_delta) = linear_pair(&self.eq_outer, row);
            let (eq_product_0, eq_product_delta) = linear_pair(&self.eq_product, row);
            let (next_values_0, next_values_delta) = linear_pair(&self.weighted_next_values, row);
            let (not_noop_0, not_noop_delta) = linear_pair(&self.not_noop, row);
            accumulate_linear_product(
                acc,
                F::one(),
                eq_outer_0,
                eq_outer_delta,
                next_values_0,
                next_values_delta,
            );
            accumulate_linear_product(
                acc,
                F::one(),
                eq_product_0,
                eq_product_delta,
                not_noop_0,
                not_noop_delta,
            );
        })
    }

    fn bind(&mut self, challenge: F) {
        bind_dense_evals_reuse_serial(&mut self.eq_outer, &mut self.scratch[0], challenge);
        bind_dense_evals_reuse_serial(&mut self.eq_product, &mut self.scratch[1], challenge);
        bind_dense_evals_reuse_serial(
            &mut self.weighted_next_values,
            &mut self.scratch[2],
            challenge,
        );
        bind_dense_evals_reuse_serial(&mut self.not_noop, &mut self.scratch[3], challenge);
        bind_dense_evals_reuse_serial(&mut self.unexpanded_pc, &mut self.scratch[4], challenge);
        bind_dense_evals_reuse_serial(&mut self.pc, &mut self.scratch[5], challenge);
        bind_dense_evals_reuse_serial(&mut self.is_virtual, &mut self.scratch[6], challenge);
        bind_dense_evals_reuse_serial(
            &mut self.is_first_in_sequence,
            &mut self.scratch[7],
            challenge,
        );
        bind_dense_evals_reuse_serial(&mut self.is_noop, &mut self.scratch[8], challenge);
    }

    fn final_evals(
        &self,
        relation: Stage3Relation,
    ) -> Result<Vec<Stage3NamedEval<F>>, Stage3KernelError> {
        let value = |values: &[F]| {
            values
                .first()
                .copied()
                .ok_or(Stage3KernelError::InvalidProof {
                    driver: relation.symbol(),
                    reason: "empty spartan shift output",
                })
        };
        Ok(vec![
            named_eval(
                "stage3.spartan_shift.eval.UnexpandedPC",
                "UnexpandedPC",
                value(&self.unexpanded_pc)?,
            ),
            named_eval("stage3.spartan_shift.eval.PC", "PC", value(&self.pc)?),
            named_eval(
                "stage3.spartan_shift.eval.OpFlagVirtualInstruction",
                "OpFlagVirtualInstruction",
                value(&self.is_virtual)?,
            ),
            named_eval(
                "stage3.spartan_shift.eval.OpFlagIsFirstInSequence",
                "OpFlagIsFirstInSequence",
                value(&self.is_first_in_sequence)?,
            ),
            named_eval(
                "stage3.spartan_shift.eval.InstructionFlagIsNoop",
                "InstructionFlagIsNoop",
                value(&self.is_noop)?,
            ),
        ])
    }
}

fn spartan_shift_state<F: Field>(
    claim: &Stage3SumcheckClaimPlan,
    inputs: &Stage3ProverInputs<'_, F>,
    store: &Stage3ValueStore<F>,
) -> Result<SpartanShiftState<F>, Stage3KernelError> {
    let cycles = stage3_cycles(inputs, claim.num_rounds)?;
    let r_outer = store.point("stage3.input.stage1.NextPC")?;
    let r_product = store.point("stage3.input.stage2.product_virtual.NextIsNoop")?;
    let gamma = store.scalar("stage3.spartan_shift.gamma")?;
    let gamma2 = store.scalar("stage3.spartan_shift.gamma2")?;
    let gamma3 = store.scalar("stage3.spartan_shift.gamma3")?;
    let gamma4 = store.scalar("stage3.spartan_shift.gamma4")?;
    Ok(SpartanShiftState::new(
        cycles, r_outer, r_product, gamma, gamma2, gamma3, gamma4,
    ))
}

fn instruction_input_state<F: Field>(
    claim: &Stage3SumcheckClaimPlan,
    inputs: &Stage3ProverInputs<'_, F>,
    store: &Stage3ValueStore<F>,
    backend: &'static str,
) -> Result<SumOfProductsState<F>, Stage3KernelError> {
    let cycles = stage3_cycles(inputs, claim.num_rounds)?;
    let eq_point = store.point("stage3.input.stage2.product_virtual.LeftInstructionInput")?;
    let gamma = store.scalar("stage3.instruction_input.gamma")?;
    let (
        right_operand_is_rs2,
        rs2_value,
        right_operand_is_imm,
        imm,
        left_operand_is_rs1,
        rs1_value,
        left_operand_is_pc,
        unexpanded_pc,
    ) = instruction_input_factors(cycles);
    let factors = vec![
        right_operand_is_rs2,
        rs2_value,
        right_operand_is_imm,
        imm,
        left_operand_is_rs1,
        rs1_value,
        left_operand_is_pc,
        unexpanded_pc,
    ];
    Ok(SumOfProductsState::new(
        SumOfProductsKind::InstructionInput,
        factors,
        Some(SplitEqState::new_low_to_high(eq_point, None)),
        vec![
            ProductTerm {
                coefficient: F::one(),
            },
            ProductTerm {
                coefficient: F::one(),
            },
            ProductTerm { coefficient: gamma },
            ProductTerm { coefficient: gamma },
        ],
        vec![
            FactorOutput {
                name: "stage3.instruction_input.eval.InstructionFlagLeftOperandIsRs1Value",
                oracle: "InstructionFlagLeftOperandIsRs1Value",
                factor: 4,
            },
            FactorOutput {
                name: "stage3.instruction_input.eval.Rs1Value",
                oracle: "Rs1Value",
                factor: 5,
            },
            FactorOutput {
                name: "stage3.instruction_input.eval.InstructionFlagLeftOperandIsPC",
                oracle: "InstructionFlagLeftOperandIsPC",
                factor: 6,
            },
            FactorOutput {
                name: "stage3.instruction_input.eval.UnexpandedPC",
                oracle: "UnexpandedPC",
                factor: 7,
            },
            FactorOutput {
                name: "stage3.instruction_input.eval.InstructionFlagRightOperandIsRs2Value",
                oracle: "InstructionFlagRightOperandIsRs2Value",
                factor: 0,
            },
            FactorOutput {
                name: "stage3.instruction_input.eval.Rs2Value",
                oracle: "Rs2Value",
                factor: 1,
            },
            FactorOutput {
                name: "stage3.instruction_input.eval.InstructionFlagRightOperandIsImm",
                oracle: "InstructionFlagRightOperandIsImm",
                factor: 2,
            },
            FactorOutput {
                name: "stage3.instruction_input.eval.Imm",
                oracle: "Imm",
                factor: 3,
            },
        ],
        Vec::new(),
        backend,
        eq_point,
    ))
}

fn registers_state<F: Field>(
    claim: &Stage3SumcheckClaimPlan,
    inputs: &Stage3ProverInputs<'_, F>,
    store: &Stage3ValueStore<F>,
    backend: &'static str,
) -> Result<SumOfProductsState<F>, Stage3KernelError> {
    let cycles = stage3_cycles(inputs, claim.num_rounds)?;
    let eq_point = store.point("stage3.input.stage1.RdWriteValue")?;
    let gamma = store.scalar("stage3.registers.gamma")?;
    let gamma2 = store.scalar("stage3.registers.gamma2")?;
    let (rd_write_value, rs1_value, rs2_value) = register_factors(cycles);
    let factors = vec![rd_write_value, rs1_value, rs2_value];
    Ok(SumOfProductsState::new(
        SumOfProductsKind::Registers,
        factors,
        Some(SplitEqState::new_low_to_high(eq_point, None)),
        vec![
            ProductTerm {
                coefficient: F::one(),
            },
            ProductTerm { coefficient: gamma },
            ProductTerm {
                coefficient: gamma2,
            },
        ],
        vec![
            FactorOutput {
                name: "stage3.registers_claim_reduction.eval.RdWriteValue",
                oracle: "RdWriteValue",
                factor: 0,
            },
            FactorOutput {
                name: "stage3.registers_claim_reduction.eval.Rs1Value",
                oracle: "Rs1Value",
                factor: 1,
            },
            FactorOutput {
                name: "stage3.registers_claim_reduction.eval.Rs2Value",
                oracle: "Rs2Value",
                factor: 2,
            },
        ],
        Vec::new(),
        backend,
        eq_point,
    ))
}

fn stage3_cycles<'a, F: Field>(
    inputs: &'a Stage3ProverInputs<'_, F>,
    num_rounds: usize,
) -> Result<&'a [Stage3Cycle], Stage3KernelError> {
    let cycles = inputs.cycles.ok_or(Stage3KernelError::MissingKernelInput {
        kernel: "jolt_stage3_batched",
        input: "cycles",
    })?;
    let expected =
        1usize
            .checked_shl(num_rounds as u32)
            .ok_or(Stage3KernelError::InvalidInputLength {
                input: "stage3.cycles",
                expected: usize::BITS as usize,
                actual: num_rounds,
            })?;
    require_operand_count("stage3.cycles", expected, cycles.len())?;
    Ok(cycles)
}

type InstructionInputFactors<F> = (
    Vec<F>,
    Vec<F>,
    Vec<F>,
    Vec<F>,
    Vec<F>,
    Vec<F>,
    Vec<F>,
    Vec<F>,
);
type RegisterFactors<F> = (Vec<F>, Vec<F>, Vec<F>);

fn spartan_shift_phase1_q_values<F: Field>(
    cycles: &[Stage3Cycle],
    outer: &EqPlusOnePrefixSuffix<F>,
    product: &EqPlusOnePrefixSuffix<F>,
    gamma: F,
    gamma2: F,
    gamma3: F,
    gamma4: F,
) -> Vec<[F; 4]> {
    let prefix_len = outer.prefix_0.len();
    let suffix_len = outer.suffix_0.len();
    debug_assert_eq!(prefix_len * suffix_len, cycles.len());
    (0..prefix_len)
        .into_par_iter()
        .map(|x_lo| {
            let mut acc = [F::Accumulator::default(); 4];
            for x_hi in 0..suffix_len {
                let cycle = cycles[x_lo + x_hi * prefix_len];
                let mut weighted = F::from_u64(cycle.unexpanded_pc) + gamma * F::from_u64(cycle.pc);
                if cycle.is_virtual {
                    weighted += gamma2;
                }
                if cycle.is_first_in_sequence {
                    weighted += gamma3;
                }
                acc[0].fmadd(outer.suffix_0[x_hi], weighted);
                acc[1].fmadd(outer.suffix_1[x_hi], weighted);
                if !cycle.is_noop {
                    acc[2].fmadd(gamma4, product.suffix_0[x_hi]);
                    acc[3].fmadd(gamma4, product.suffix_1[x_hi]);
                }
            }
            [
                acc[0].reduce(),
                acc[1].reduce(),
                acc[2].reduce(),
                acc[3].reduce(),
            ]
        })
        .collect()
}

fn spartan_shift_phase2_eq_plus_one<F: Field>(point: &[F], low_eq: &[F]) -> Vec<F> {
    let split = EqPlusOnePrefixSuffix::new(point);
    let prefix_0_eval = deferred_output_eval(&split.prefix_0, low_eq);
    let prefix_1_eval = deferred_output_eval(&split.prefix_1, low_eq);
    debug_assert_eq!(split.prefix_0.len(), low_eq.len());
    split
        .suffix_0
        .iter()
        .zip(split.suffix_1.iter())
        .map(|(&suffix_0, &suffix_1)| prefix_0_eval * suffix_0 + prefix_1_eval * suffix_1)
        .collect()
}

type SpartanShiftPhase2Outputs<F> = (Vec<F>, Vec<F>, Vec<F>, Vec<F>, Vec<F>, Vec<F>, Vec<F>);

fn spartan_shift_phase2_outputs<F: Field>(
    cycles: &[Stage3Cycle],
    low_eq: &[F],
    gamma: F,
    gamma2: F,
    gamma3: F,
) -> SpartanShiftPhase2Outputs<F> {
    let low_len = low_eq.len();
    let high_len = cycles.len() / low_len;
    let mut unexpanded_pc = vec![F::zero(); high_len];
    let mut pc = vec![F::zero(); high_len];
    let mut is_virtual = vec![F::zero(); high_len];
    let mut is_first_in_sequence = vec![F::zero(); high_len];
    let mut is_noop = vec![F::zero(); high_len];
    let mut weighted_next_values = vec![F::zero(); high_len];
    let mut not_noop = vec![F::zero(); high_len];
    (
        &mut unexpanded_pc,
        &mut pc,
        &mut is_virtual,
        &mut is_first_in_sequence,
        &mut is_noop,
        &mut weighted_next_values,
        &mut not_noop,
        0..high_len,
    )
        .into_par_iter()
        .for_each(
            |(
                unexpanded_pc,
                pc,
                is_virtual,
                is_first_in_sequence,
                is_noop,
                weighted_next_values,
                not_noop,
                x_hi,
            )| {
                let mut unexpanded_acc = F::Accumulator::default();
                let mut pc_acc = F::Accumulator::default();
                let mut virtual_acc = F::Accumulator::default();
                let mut first_acc = F::Accumulator::default();
                let mut noop_acc = F::Accumulator::default();
                let base = x_hi * low_len;
                for (x_lo, &weight) in low_eq.iter().enumerate() {
                    let cycle = cycles[base + x_lo];
                    unexpanded_acc.fmadd_u64(weight, cycle.unexpanded_pc);
                    pc_acc.fmadd_u64(weight, cycle.pc);
                    virtual_acc.fmadd_bool(weight, cycle.is_virtual);
                    first_acc.fmadd_bool(weight, cycle.is_first_in_sequence);
                    noop_acc.fmadd_bool(weight, cycle.is_noop);
                }
                *unexpanded_pc = unexpanded_acc.reduce();
                *pc = pc_acc.reduce();
                *is_virtual = virtual_acc.reduce();
                *is_first_in_sequence = first_acc.reduce();
                *is_noop = noop_acc.reduce();
                *weighted_next_values = *unexpanded_pc
                    + gamma * *pc
                    + gamma2 * *is_virtual
                    + gamma3 * *is_first_in_sequence;
                *not_noop = F::one() - *is_noop;
            },
        );
    (
        unexpanded_pc,
        pc,
        is_virtual,
        is_first_in_sequence,
        is_noop,
        weighted_next_values,
        not_noop,
    )
}

fn instruction_input_factors<F: Field>(cycles: &[Stage3Cycle]) -> InstructionInputFactors<F> {
    let mut right_operand_is_rs2 = vec![F::zero(); cycles.len()];
    let mut rs2_value = vec![F::zero(); cycles.len()];
    let mut right_operand_is_imm = vec![F::zero(); cycles.len()];
    let mut imm = vec![F::zero(); cycles.len()];
    let mut left_operand_is_rs1 = vec![F::zero(); cycles.len()];
    let mut rs1_value = vec![F::zero(); cycles.len()];
    let mut left_operand_is_pc = vec![F::zero(); cycles.len()];
    let mut unexpanded_pc = vec![F::zero(); cycles.len()];
    (
        &mut right_operand_is_rs2,
        &mut rs2_value,
        &mut right_operand_is_imm,
        &mut imm,
        &mut left_operand_is_rs1,
        &mut rs1_value,
        &mut left_operand_is_pc,
        &mut unexpanded_pc,
        cycles,
    )
        .into_par_iter()
        .for_each(
            |(
                right_operand_is_rs2,
                rs2_value,
                right_operand_is_imm,
                imm,
                left_operand_is_rs1,
                rs1_value,
                left_operand_is_pc,
                unexpanded_pc,
                cycle,
            )| {
                *right_operand_is_rs2 = F::from_bool(cycle.right_operand_is_rs2);
                *rs2_value = F::from_u64(cycle.rs2_value);
                *right_operand_is_imm = F::from_bool(cycle.right_operand_is_imm);
                *imm = F::from_i128(cycle.imm);
                *left_operand_is_rs1 = F::from_bool(cycle.left_operand_is_rs1);
                *rs1_value = F::from_u64(cycle.rs1_value);
                *left_operand_is_pc = F::from_bool(cycle.left_operand_is_pc);
                *unexpanded_pc = F::from_u64(cycle.unexpanded_pc);
            },
        );
    (
        right_operand_is_rs2,
        rs2_value,
        right_operand_is_imm,
        imm,
        left_operand_is_rs1,
        rs1_value,
        left_operand_is_pc,
        unexpanded_pc,
    )
}

fn register_factors<F: Field>(cycles: &[Stage3Cycle]) -> RegisterFactors<F> {
    let mut rd_write_value = vec![F::zero(); cycles.len()];
    let mut rs1_value = vec![F::zero(); cycles.len()];
    let mut rs2_value = vec![F::zero(); cycles.len()];
    (&mut rd_write_value, &mut rs1_value, &mut rs2_value, cycles)
        .into_par_iter()
        .for_each(|(rd_write_value, rs1_value, rs2_value, cycle)| {
            *rd_write_value = F::from_u64(cycle.rd_write_value);
            *rs1_value = F::from_u64(cycle.rs1_value);
            *rs2_value = F::from_u64(cycle.rs2_value);
        });
    (rd_write_value, rs1_value, rs2_value)
}

fn expected_batched_output_claim<F: Field>(
    context: Stage3KernelContext<'_>,
    store: &Stage3ValueStore<F>,
    evals: &[Stage3NamedEval<F>],
    point: &[F],
    batching_coeffs: &[F],
) -> Result<F, Stage3KernelError> {
    let mut expected = F::zero();
    for (claim, &coefficient) in context.batch_claims()?.iter().zip(batching_coeffs) {
        let instance = context
            .program
            .instance_results
            .iter()
            .find(|instance| {
                instance.claim == claim.symbol && instance.source == context.driver.symbol
            })
            .ok_or(Stage3KernelError::MissingClaim {
                batch: context.batch.symbol,
                claim: claim.symbol,
            })?;
        let local_point = point
            .get(instance.round_offset..instance.round_offset + instance.num_rounds)
            .ok_or(Stage3KernelError::InvalidInputLength {
                input: instance.symbol,
                expected: instance.round_offset + instance.num_rounds,
                actual: point.len(),
            })?;
        let claim_value = match Stage3Relation::from_symbol(instance.relation).ok_or(
            Stage3KernelError::UnknownRelation {
                relation: instance.relation,
            },
        )? {
            Stage3Relation::SpartanShift => expected_spartan_shift(store, evals, local_point)?,
            Stage3Relation::InstructionInput => {
                expected_instruction_input(store, evals, local_point)?
            }
            Stage3Relation::RegistersClaimReduction => {
                expected_registers(store, evals, local_point)?
            }
            relation @ Stage3Relation::Batched => {
                return Err(Stage3KernelError::KernelNotImplemented {
                    abi: relation.symbol(),
                })
            }
        };
        expected += coefficient * claim_value;
    }
    Ok(expected)
}

fn expected_spartan_shift<F: Field>(
    store: &Stage3ValueStore<F>,
    evals: &[Stage3NamedEval<F>],
    local_point: &[F],
) -> Result<F, Stage3KernelError> {
    let opening_point = reverse_slice(local_point);
    let eq_outer =
        EqPlusOnePolynomial::<F>::new(store.point("stage3.input.stage1.NextPC")?.to_vec())
            .evaluate(&opening_point);
    let eq_product = EqPlusOnePolynomial::<F>::new(
        store
            .point("stage3.input.stage2.product_virtual.NextIsNoop")?
            .to_vec(),
    )
    .evaluate(&opening_point);
    let gamma = store.scalar("stage3.spartan_shift.gamma")?;
    let gamma2 = store.scalar("stage3.spartan_shift.gamma2")?;
    let gamma3 = store.scalar("stage3.spartan_shift.gamma3")?;
    let gamma4 = store.scalar("stage3.spartan_shift.gamma4")?;
    let weighted_outer = eval_by_name(evals, "stage3.spartan_shift.eval.UnexpandedPC")?
        + gamma * eval_by_name(evals, "stage3.spartan_shift.eval.PC")?
        + gamma2 * eval_by_name(evals, "stage3.spartan_shift.eval.OpFlagVirtualInstruction")?
        + gamma3 * eval_by_name(evals, "stage3.spartan_shift.eval.OpFlagIsFirstInSequence")?;
    Ok(eq_outer * weighted_outer
        + gamma4
            * eq_product
            * (F::one() - eval_by_name(evals, "stage3.spartan_shift.eval.InstructionFlagIsNoop")?))
}

fn expected_instruction_input<F: Field>(
    store: &Stage3ValueStore<F>,
    evals: &[Stage3NamedEval<F>],
    local_point: &[F],
) -> Result<F, Stage3KernelError> {
    let opening_point = reverse_slice(local_point);
    let eq_eval = EqPolynomial::<F>::mle(
        &opening_point,
        store.point("stage3.input.stage2.product_virtual.LeftInstructionInput")?,
    );
    let left = eval_by_name(
        evals,
        "stage3.instruction_input.eval.InstructionFlagLeftOperandIsRs1Value",
    )? * eval_by_name(evals, "stage3.instruction_input.eval.Rs1Value")?
        + eval_by_name(
            evals,
            "stage3.instruction_input.eval.InstructionFlagLeftOperandIsPC",
        )? * eval_by_name(evals, "stage3.instruction_input.eval.UnexpandedPC")?;
    let right = eval_by_name(
        evals,
        "stage3.instruction_input.eval.InstructionFlagRightOperandIsRs2Value",
    )? * eval_by_name(evals, "stage3.instruction_input.eval.Rs2Value")?
        + eval_by_name(
            evals,
            "stage3.instruction_input.eval.InstructionFlagRightOperandIsImm",
        )? * eval_by_name(evals, "stage3.instruction_input.eval.Imm")?;
    Ok(eq_eval * (right + store.scalar("stage3.instruction_input.gamma")? * left))
}

fn expected_registers<F: Field>(
    store: &Stage3ValueStore<F>,
    evals: &[Stage3NamedEval<F>],
    local_point: &[F],
) -> Result<F, Stage3KernelError> {
    let opening_point = reverse_slice(local_point);
    let eq_eval = EqPolynomial::<F>::mle(
        &opening_point,
        store.point("stage3.input.stage1.RdWriteValue")?,
    );
    Ok(eq_eval
        * (eval_by_name(evals, "stage3.registers_claim_reduction.eval.RdWriteValue")?
            + store.scalar("stage3.registers.gamma")?
                * eval_by_name(evals, "stage3.registers_claim_reduction.eval.Rs1Value")?
            + store.scalar("stage3.registers.gamma2")?
                * eval_by_name(evals, "stage3.registers_claim_reduction.eval.Rs2Value")?))
}

fn eval_by_name<F: Field>(
    evals: &[Stage3NamedEval<F>],
    name: &'static str,
) -> Result<F, Stage3KernelError> {
    evals
        .iter()
        .find(|eval| eval.name == name)
        .map(|eval| eval.value)
        .ok_or(Stage3KernelError::MissingValue { symbol: name })
}

fn deferred_output_eval<F: Field>(values: &[F], eq: &[F]) -> F {
    debug_assert_eq!(values.len(), eq.len());
    if values.len() >= DENSE_BIND_PAR_THRESHOLD {
        values
            .par_iter()
            .zip(eq.par_iter())
            .map(|(&value, &weight)| value * weight)
            .sum()
    } else {
        values
            .iter()
            .zip(eq)
            .map(|(&value, &weight)| value * weight)
            .sum()
    }
}

fn reverse_slice<F: Field>(values: &[F]) -> Vec<F> {
    values.iter().rev().copied().collect()
}

fn named_eval<F: Field>(name: &'static str, oracle: &'static str, value: F) -> Stage3NamedEval<F> {
    Stage3NamedEval {
        name,
        oracle,
        value,
    }
}

fn claim_relation(
    program: &'static Stage3CpuProgramPlan,
    claim: &Stage3SumcheckClaimPlan,
) -> Result<Stage3Relation, Stage3KernelError> {
    if let Some(relation) = claim.relation {
        return Stage3Relation::from_symbol(relation)
            .ok_or(Stage3KernelError::UnknownRelation { relation });
    }
    let kernel_symbol = claim.kernel.ok_or(Stage3KernelError::MissingKernel {
        driver: claim.symbol,
        kernel: "<missing>",
    })?;
    let kernel = find_kernel(program, kernel_symbol).ok_or(Stage3KernelError::MissingKernel {
        driver: claim.symbol,
        kernel: kernel_symbol,
    })?;
    kernel.relation_kind()
}

fn instance_round_offset(
    program: &'static Stage3CpuProgramPlan,
    driver: &'static str,
    claim: &'static str,
) -> Result<usize, Stage3KernelError> {
    program
        .instance_results
        .iter()
        .find(|instance| instance.source == driver && instance.claim == claim)
        .map(|instance| instance.round_offset)
        .ok_or(Stage3KernelError::MissingClaim {
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

fn round_poly_from_instruction_input<F: Field>(
    factors: &[Vec<F>],
    terms: &[ProductTerm<F>],
    split_eq: &SplitEqState<F>,
    previous_claim: F,
) -> UnivariatePoly<F> {
    debug_assert_eq!(factors.len(), 8);
    debug_assert_eq!(terms.len(), 4);
    let gamma = terms[2].coefficient;
    let (q_constant, q_quadratic) =
        instruction_input_split_round_coefficients(factors, split_eq, gamma);
    gruen_cubic_poly(
        split_eq.current_target(),
        q_constant,
        q_quadratic,
        previous_claim,
    )
}

fn round_poly_from_registers<F: Field>(
    factors: &[Vec<F>],
    terms: &[ProductTerm<F>],
    split_eq: &SplitEqState<F>,
    previous_claim: F,
) -> UnivariatePoly<F> {
    debug_assert_eq!(factors.len(), 3);
    debug_assert_eq!(terms.len(), 3);
    let gamma = terms[1].coefficient;
    let gamma2 = terms[2].coefficient;
    let q_constant = registers_split_round_constant(factors, split_eq, gamma, gamma2);
    gruen_quadratic_poly(split_eq.current_target(), q_constant, previous_claim)
}

pub(crate) fn instruction_input_split_round_coefficients<F: Field>(
    factors: &[Vec<F>],
    split_eq: &SplitEqState<F>,
    gamma: F,
) -> (F, F) {
    let e_in = split_eq.e_in();
    let e_out = split_eq.e_out();
    if e_in.len() > 1 {
        instruction_input_low_round_coefficients(factors, e_in, e_out, gamma)
    } else {
        instruction_input_high_round_coefficients(factors, e_in[0], e_out, gamma)
    }
}

fn instruction_input_low_round_coefficients<F: Field>(
    factors: &[Vec<F>],
    e_in: &[F],
    e_out: &[F],
    gamma: F,
) -> (F, F) {
    let in_len = e_in.len();
    let in_pairs = in_len / 2;
    if factors[0].len() / 2 >= DENSE_BIND_PAR_THRESHOLD {
        let accumulators = (0..e_out.len())
            .into_par_iter()
            .map(|x_out| {
                let mut local = [F::Accumulator::default(); 2];
                let base_pair = x_out * in_pairs;
                let out_weight = e_out[x_out];
                for pair in 0..in_pairs {
                    accumulate_instruction_input_quadratic_pair(
                        &mut local,
                        out_weight * (e_in[2 * pair] + e_in[2 * pair + 1]),
                        factors,
                        base_pair + pair,
                        gamma,
                    );
                }
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
        (accumulators[0].reduce(), accumulators[1].reduce())
    } else {
        let mut total = [F::Accumulator::default(); 2];
        for (x_out, &out_weight) in e_out.iter().enumerate() {
            let base_pair = x_out * in_pairs;
            for pair in 0..in_pairs {
                accumulate_instruction_input_quadratic_pair(
                    &mut total,
                    out_weight * (e_in[2 * pair] + e_in[2 * pair + 1]),
                    factors,
                    base_pair + pair,
                    gamma,
                );
            }
        }
        (total[0].reduce(), total[1].reduce())
    }
}

fn instruction_input_high_round_coefficients<F: Field>(
    factors: &[Vec<F>],
    in_weight: F,
    e_out: &[F],
    gamma: F,
) -> (F, F) {
    let pairs = e_out.len() / 2;
    if pairs >= DENSE_BIND_PAR_THRESHOLD {
        let accumulators = (0..pairs)
            .into_par_iter()
            .map(|pair| {
                let mut local = [F::Accumulator::default(); 2];
                accumulate_instruction_input_quadratic_pair(
                    &mut local,
                    in_weight * (e_out[2 * pair] + e_out[2 * pair + 1]),
                    factors,
                    pair,
                    gamma,
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
        (accumulators[0].reduce(), accumulators[1].reduce())
    } else {
        let mut total = [F::Accumulator::default(); 2];
        for pair in 0..pairs {
            accumulate_instruction_input_quadratic_pair(
                &mut total,
                in_weight * (e_out[2 * pair] + e_out[2 * pair + 1]),
                factors,
                pair,
                gamma,
            );
        }
        (total[0].reduce(), total[1].reduce())
    }
}

fn accumulate_instruction_input_quadratic_pair<F: Field>(
    accumulators: &mut [F::Accumulator; 2],
    weight: F,
    factors: &[Vec<F>],
    row: usize,
    gamma: F,
) {
    accumulate_quadratic_coefficients(
        accumulators,
        weight,
        F::one(),
        &factors[0],
        &factors[1],
        row,
    );
    accumulate_quadratic_coefficients(
        accumulators,
        weight,
        F::one(),
        &factors[2],
        &factors[3],
        row,
    );
    accumulate_quadratic_coefficients(accumulators, weight, gamma, &factors[4], &factors[5], row);
    accumulate_quadratic_coefficients(accumulators, weight, gamma, &factors[6], &factors[7], row);
}

fn accumulate_quadratic_coefficients<F: Field>(
    accumulators: &mut [F::Accumulator; 2],
    weight: F,
    scale: F,
    left: &[F],
    right: &[F],
    row: usize,
) {
    let (left_0, left_delta) = linear_pair(left, row);
    let (right_0, right_delta) = linear_pair(right, row);
    let scaled_weight = weight * scale;
    accumulators[0].fmadd(scaled_weight * left_0, right_0);
    accumulators[1].fmadd(scaled_weight * left_delta, right_delta);
}

pub(crate) fn registers_split_round_constant<F: Field>(
    factors: &[Vec<F>],
    split_eq: &SplitEqState<F>,
    gamma: F,
    gamma2: F,
) -> F {
    let e_in = split_eq.e_in();
    let e_out = split_eq.e_out();
    if e_in.len() > 1 {
        registers_low_round_constant(factors, e_in, e_out, gamma, gamma2)
    } else {
        registers_high_round_constant(factors, e_in[0], e_out, gamma, gamma2)
    }
}

fn registers_low_round_constant<F: Field>(
    factors: &[Vec<F>],
    e_in: &[F],
    e_out: &[F],
    gamma: F,
    gamma2: F,
) -> F {
    let in_len = e_in.len();
    let in_pairs = in_len / 2;
    if factors[0].len() / 2 >= DENSE_BIND_PAR_THRESHOLD {
        (0..e_out.len())
            .into_par_iter()
            .map(|x_out| {
                let mut local = F::Accumulator::default();
                let base_pair = x_out * in_pairs;
                let out_weight = e_out[x_out];
                for pair in 0..in_pairs {
                    accumulate_register_constant(
                        &mut local,
                        out_weight * (e_in[2 * pair] + e_in[2 * pair + 1]),
                        factors,
                        base_pair + pair,
                        gamma,
                        gamma2,
                    );
                }
                local
            })
            .reduce(F::Accumulator::default, |mut left, right| {
                left.merge(right);
                left
            })
            .reduce()
    } else {
        let mut total = F::Accumulator::default();
        for (x_out, &out_weight) in e_out.iter().enumerate() {
            let base_pair = x_out * in_pairs;
            for pair in 0..in_pairs {
                accumulate_register_constant(
                    &mut total,
                    out_weight * (e_in[2 * pair] + e_in[2 * pair + 1]),
                    factors,
                    base_pair + pair,
                    gamma,
                    gamma2,
                );
            }
        }
        total.reduce()
    }
}

fn registers_high_round_constant<F: Field>(
    factors: &[Vec<F>],
    in_weight: F,
    e_out: &[F],
    gamma: F,
    gamma2: F,
) -> F {
    let pairs = e_out.len() / 2;
    if pairs >= DENSE_BIND_PAR_THRESHOLD {
        (0..pairs)
            .into_par_iter()
            .map(|pair| {
                let mut local = F::Accumulator::default();
                accumulate_register_constant(
                    &mut local,
                    in_weight * (e_out[2 * pair] + e_out[2 * pair + 1]),
                    factors,
                    pair,
                    gamma,
                    gamma2,
                );
                local
            })
            .reduce(F::Accumulator::default, |mut left, right| {
                left.merge(right);
                left
            })
            .reduce()
    } else {
        let mut total = F::Accumulator::default();
        for pair in 0..pairs {
            accumulate_register_constant(
                &mut total,
                in_weight * (e_out[2 * pair] + e_out[2 * pair + 1]),
                factors,
                pair,
                gamma,
                gamma2,
            );
        }
        total.reduce()
    }
}

fn accumulate_register_constant<F: Field>(
    accumulator: &mut F::Accumulator,
    weight: F,
    factors: &[Vec<F>],
    row: usize,
    gamma: F,
    gamma2: F,
) {
    accumulator.fmadd(weight, factors[0][2 * row]);
    accumulator.fmadd(weight * gamma, factors[1][2 * row]);
    accumulator.fmadd(weight * gamma2, factors[2][2 * row]);
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

fn gruen_quadratic_poly<F: Field>(
    target: F,
    q_constant: F,
    previous_claim: F,
) -> UnivariatePoly<F> {
    let eq_eval_1 = target;
    let eq_eval_0 = F::one() - target;
    let eq_delta = eq_eval_1 - eq_eval_0;
    let eq_eval_2 = eq_eval_1 + eq_delta;
    let quadratic_eval_0 = eq_eval_0 * q_constant;
    let quadratic_eval_1 = previous_claim - quadratic_eval_0;
    let linear_eval_1 = quadratic_eval_1 / eq_eval_1;
    let linear_eval_2 = linear_eval_1 + linear_eval_1 - q_constant;
    UnivariatePoly::from_evals(&[
        quadratic_eval_0,
        quadratic_eval_1,
        eq_eval_2 * linear_eval_2,
    ])
}

fn round_poly_from_stage3_coefficients<F, C>(
    half: usize,
    degree: usize,
    coefficients: C,
) -> UnivariatePoly<F>
where
    F: Field,
    C: Fn(usize, &mut [F::Accumulator; 4]) + Sync,
{
    let accumulators = if half >= DENSE_BIND_PAR_THRESHOLD {
        (0..half)
            .into_par_iter()
            .map(|row| {
                let mut local = [F::Accumulator::default(); 4];
                coefficients(row, &mut local);
                local
            })
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
            coefficients(row, &mut total);
            total
        })
    };
    UnivariatePoly::new(
        accumulators[..=degree]
            .iter()
            .copied()
            .map(FieldAccumulator::reduce)
            .collect(),
    )
}

#[inline]
fn linear_pair<F: Field>(factor: &[F], row: usize) -> (F, F) {
    let low = factor[2 * row];
    (low, factor[2 * row + 1] - low)
}

#[inline]
fn accumulate_linear_product<F: Field>(
    acc: &mut [F::Accumulator; 4],
    scale: F,
    left_0: F,
    left_delta: F,
    right_0: F,
    right_delta: F,
) {
    acc[0].fmadd(scale * left_0, right_0);
    acc[1].fmadd(scale * left_delta, right_0);
    acc[1].fmadd(scale * left_0, right_delta);
    acc[2].fmadd(scale * left_delta, right_delta);
}

#[inline]
fn bind_dense_evals_reuse_serial<F: Field>(
    values: &mut Vec<F>,
    scratch: &mut Vec<F>,
    challenge: F,
) {
    let half = values.len() / 2;
    scratch.resize(half, F::zero());
    for (index, output) in scratch.iter_mut().enumerate() {
        let low = values[index << 1];
        let high = values[(index << 1) + 1];
        *output = low + challenge * (high - low);
    }
    std::mem::swap(values, scratch);
    scratch.clear();
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
    program: &'static Stage3CpuProgramPlan,
    store: &mut Stage3ValueStore<F>,
    transcript: &mut T,
    evals: &[Stage3NamedEval<F>],
) -> Result<Vec<Stage3OpeningClaimValue<F>>, Stage3KernelError>
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
    let mut seen = seed_stage3_opening_aliases(store, program);
    for batch in program.opening_batches {
        for symbol in batch.claim_operands {
            let claim =
                find_opening_claim(program, symbol).ok_or(Stage3KernelError::MissingClaim {
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
            opening_claims.push(Stage3OpeningClaimValue {
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

fn seed_stage3_opening_aliases<F: Field>(
    store: &Stage3ValueStore<F>,
    program: &'static Stage3CpuProgramPlan,
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
    program: &'a Stage3CpuProgramPlan,
    symbol: &str,
) -> Option<&'a Stage3OpeningClaimPlan> {
    program
        .opening_claims
        .iter()
        .find(|claim| claim.symbol == symbol)
}

pub fn execute_stage3_program<F, E, T>(
    program: &'static Stage3CpuProgramPlan,
    mode: Stage3ExecutionMode,
    executor: &mut E,
    transcript: &mut T,
) -> Result<Stage3ExecutionArtifacts<F>, Stage3KernelError>
where
    F: Field,
    E: Stage3KernelExecutor<F>,
    T: Transcript<Challenge = F>,
{
    verify_static_program_shape(program)?;
    let mut artifacts = Stage3ExecutionArtifacts::default();
    for step in program.steps {
        match step.kind {
            "transcript_squeeze" => {
                let squeeze = find_transcript_squeeze(program, step.symbol).ok_or(
                    Stage3KernelError::MissingValue {
                        symbol: step.symbol,
                    },
                )?;
                execute_stage3_squeeze(squeeze, executor, transcript, &mut artifacts)?;
            }
            "sumcheck_driver" => {
                let driver =
                    find_driver(program, step.symbol).ok_or(Stage3KernelError::MissingKernel {
                        driver: step.symbol,
                        kernel: step.symbol,
                    })?;
                execute_stage3_driver(program, mode, driver, executor, transcript, &mut artifacts)?;
            }
            _ => {
                return Err(Stage3KernelError::InvalidProof {
                    driver: step.symbol,
                    reason: "unsupported stage3 program step",
                });
            }
        }
    }
    artifacts
        .opening_batches
        .extend(program.opening_batches.iter());
    Ok(artifacts)
}

fn execute_stage3_squeeze<F, E, T>(
    squeeze: &'static Stage3TranscriptSqueezePlan,
    executor: &mut E,
    transcript: &mut T,
    artifacts: &mut Stage3ExecutionArtifacts<F>,
) -> Result<(), Stage3KernelError>
where
    F: Field,
    E: Stage3KernelExecutor<F>,
    T: Transcript<Challenge = F>,
{
    let values = transcript.challenge_vector(squeeze.count);
    executor.observe_challenge_vector(squeeze, &values)?;
    artifacts.challenge_vectors.push(Stage3ChallengeVector {
        symbol: squeeze.symbol,
        values,
    });
    Ok(())
}

fn execute_stage3_driver<F, E, T>(
    program: &'static Stage3CpuProgramPlan,
    mode: Stage3ExecutionMode,
    driver: &'static Stage3SumcheckDriverPlan,
    executor: &mut E,
    transcript: &mut T,
    artifacts: &mut Stage3ExecutionArtifacts<F>,
) -> Result<(), Stage3KernelError>
where
    F: Field,
    E: Stage3KernelExecutor<F>,
    T: Transcript<Challenge = F>,
{
    let kernel_symbol = driver.kernel.ok_or(Stage3KernelError::MissingKernel {
        driver: driver.symbol,
        kernel: "<missing>",
    })?;
    let kernel = find_kernel(program, kernel_symbol).ok_or(Stage3KernelError::MissingKernel {
        driver: driver.symbol,
        kernel: kernel_symbol,
    })?;
    let batch = find_batch(program, driver.batch).ok_or(Stage3KernelError::MissingBatch {
        driver: driver.symbol,
        batch: driver.batch,
    })?;
    let context = Stage3KernelContext {
        mode,
        program,
        kernel,
        batch,
        driver,
    };
    let output = match mode {
        Stage3ExecutionMode::Prover => executor.prove_sumcheck(context, transcript)?,
        Stage3ExecutionMode::Verifier => executor.verify_sumcheck(context, transcript)?,
    };
    executor.observe_sumcheck_output(&output)?;
    artifacts
        .opening_claims
        .extend(output.opening_claims.clone());
    artifacts.sumchecks.push(output);
    Ok(())
}

fn verify_static_program_shape(
    program: &'static Stage3CpuProgramPlan,
) -> Result<(), Stage3KernelError> {
    for expr in program.field_exprs {
        verify_count(expr.symbol, expr.operand_names.len(), expr.operands.len())?;
    }
    for batch in program.batches {
        verify_count(batch.symbol, batch.count, batch.ordered_claims.len())?;
        verify_count(batch.symbol, batch.count, batch.claim_operands.len())?;
    }
    for batch in program.opening_batches {
        verify_count(batch.symbol, batch.count, batch.ordered_claims.len())?;
        verify_count(batch.symbol, batch.count, batch.claim_operands.len())?;
    }
    for kernel in program.kernels {
        let relation = kernel.relation_kind()?;
        let abi = kernel.abi_kind()?;
        if relation
            .symbol()
            .replace("jolt.stage3.", "jolt_stage3_")
            .replace('.', "_")
            != abi.name()
        {
            return Err(Stage3KernelError::InvalidProof {
                driver: kernel.symbol,
                reason: "kernel relation and ABI mismatch",
            });
        }
    }
    Ok(())
}

fn verify_count(
    artifact: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), Stage3KernelError> {
    if expected == actual {
        Ok(())
    } else {
        Err(Stage3KernelError::PlanCountMismatch {
            artifact,
            expected,
            actual,
        })
    }
}

fn find_kernel<'a>(
    program: &'a Stage3CpuProgramPlan,
    symbol: &str,
) -> Option<&'a Stage3KernelPlan> {
    program
        .kernels
        .iter()
        .find(|kernel| kernel.symbol == symbol)
}

fn find_batch<'a>(
    program: &'a Stage3CpuProgramPlan,
    symbol: &str,
) -> Option<&'a Stage3SumcheckBatchPlan> {
    program.batches.iter().find(|batch| batch.symbol == symbol)
}

fn find_driver<'a>(
    program: &'a Stage3CpuProgramPlan,
    symbol: &str,
) -> Option<&'a Stage3SumcheckDriverPlan> {
    program
        .drivers
        .iter()
        .find(|driver| driver.symbol == symbol)
}

fn find_transcript_squeeze<'a>(
    program: &'a Stage3CpuProgramPlan,
    symbol: &str,
) -> Option<&'a Stage3TranscriptSqueezePlan> {
    program
        .transcript_squeezes
        .iter()
        .find(|squeeze| squeeze.symbol == symbol)
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "stage3 kernel tests fail fast")]
mod tests {
    use super::*;
    use jolt_field::Fr;
    use jolt_transcript::Blake2bTranscript;

    #[test]
    fn stage3_relation_and_abi_registry_is_complete() {
        let relations = [
            Stage3Relation::SpartanShift,
            Stage3Relation::InstructionInput,
            Stage3Relation::RegistersClaimReduction,
            Stage3Relation::Batched,
        ];
        for relation in relations {
            assert_eq!(
                Stage3Relation::from_symbol(relation.symbol()),
                Some(relation)
            );
        }

        let abis = [
            Stage3KernelAbi::SpartanShift,
            Stage3KernelAbi::InstructionInput,
            Stage3KernelAbi::RegistersClaimReduction,
            Stage3KernelAbi::Batched,
        ];
        for abi in abis {
            assert_eq!(Stage3KernelAbi::from_name(abi.name()), Some(abi));
        }
    }

    #[test]
    fn stage3_batched_kernel_proves_and_verifies_synthetic_trace() {
        let program = synthetic_stage3_program();
        let cycles = synthetic_cycles();
        let opening_inputs = synthetic_opening_inputs(&cycles);
        let prover_inputs = Stage3ProverInputs::new(&opening_inputs).with_cycles(&cycles);
        let mut prover = Stage3ProverKernelExecutor::new(prover_inputs);
        let mut prover_transcript = Blake2bTranscript::<Fr>::new(b"stage3_test");

        let artifacts = execute_stage3_program(
            program,
            Stage3ExecutionMode::Prover,
            &mut prover,
            &mut prover_transcript,
        )
        .expect("stage3 prover succeeds");

        assert_eq!(artifacts.sumchecks.len(), 1);
        assert_eq!(artifacts.sumchecks[0].proof.round_polynomials.len(), 2);
        let proof = Stage3Proof::from(artifacts);
        let mut verifier = Stage3VerifierKernelExecutor::new(&proof, &opening_inputs);
        let mut verifier_transcript = Blake2bTranscript::<Fr>::new(b"stage3_test");
        let verified = execute_stage3_program(
            program,
            Stage3ExecutionMode::Verifier,
            &mut verifier,
            &mut verifier_transcript,
        )
        .expect("stage3 verifier accepts prover proof");

        assert_eq!(verified.sumchecks.len(), 1);
        assert_eq!(prover_transcript.state(), verifier_transcript.state());
    }

    #[test]
    fn stage3_batched_kernel_rejects_tampered_eval() {
        let program = synthetic_stage3_program();
        let cycles = synthetic_cycles();
        let opening_inputs = synthetic_opening_inputs(&cycles);
        let mut prover = Stage3ProverKernelExecutor::new(
            Stage3ProverInputs::new(&opening_inputs).with_cycles(&cycles),
        );
        let mut prover_transcript = Blake2bTranscript::<Fr>::new(b"stage3_test");
        let mut proof = Stage3Proof::from(
            execute_stage3_program(
                program,
                Stage3ExecutionMode::Prover,
                &mut prover,
                &mut prover_transcript,
            )
            .expect("stage3 prover succeeds"),
        );
        proof.sumchecks[0].evals[0].value += Fr::from_u64(1);

        let mut verifier = Stage3VerifierKernelExecutor::new(&proof, &opening_inputs);
        let mut verifier_transcript = Blake2bTranscript::<Fr>::new(b"stage3_test");
        let error = execute_stage3_program(
            program,
            Stage3ExecutionMode::Verifier,
            &mut verifier,
            &mut verifier_transcript,
        )
        .expect_err("tampered proof is rejected");

        assert!(matches!(error, Stage3KernelError::InvalidProof { .. }));
    }

    fn synthetic_stage3_program() -> &'static Stage3CpuProgramPlan {
        let exprs = leak_slice(vec![
            field_expr(
                "stage3.spartan_shift.gamma2",
                "field.pow:2",
                vec!["stage3.spartan_shift.gamma"],
            ),
            field_expr(
                "stage3.spartan_shift.gamma3",
                "field.mul",
                vec!["stage3.spartan_shift.gamma2", "stage3.spartan_shift.gamma"],
            ),
            field_expr(
                "stage3.spartan_shift.gamma4",
                "field.mul",
                vec!["stage3.spartan_shift.gamma2", "stage3.spartan_shift.gamma2"],
            ),
            field_expr(
                "stage3.spartan_shift.term.NextPC",
                "field.mul",
                vec!["stage3.spartan_shift.gamma", "stage3.input.stage1.NextPC"],
            ),
            field_expr(
                "stage3.spartan_shift.term.NextIsVirtual",
                "field.mul",
                vec![
                    "stage3.spartan_shift.gamma2",
                    "stage3.input.stage1.NextIsVirtual",
                ],
            ),
            field_expr(
                "stage3.spartan_shift.term.NextIsFirstInSequence",
                "field.mul",
                vec![
                    "stage3.spartan_shift.gamma3",
                    "stage3.input.stage1.NextIsFirstInSequence",
                ],
            ),
            field_expr(
                "stage3.spartan_shift.one_minus.NextIsNoop",
                "field.sub",
                vec![
                    "stage3.field.one",
                    "stage3.input.stage2.product_virtual.NextIsNoop",
                ],
            ),
            field_expr(
                "stage3.spartan_shift.term.NextIsNoop",
                "field.mul",
                vec![
                    "stage3.spartan_shift.gamma4",
                    "stage3.spartan_shift.one_minus.NextIsNoop",
                ],
            ),
            field_expr(
                "stage3.spartan_shift.partial.NextUnexpandedPCNextPC",
                "field.add",
                vec![
                    "stage3.input.stage1.NextUnexpandedPC",
                    "stage3.spartan_shift.term.NextPC",
                ],
            ),
            field_expr(
                "stage3.spartan_shift.partial.NextIsVirtual",
                "field.add",
                vec![
                    "stage3.spartan_shift.partial.NextUnexpandedPCNextPC",
                    "stage3.spartan_shift.term.NextIsVirtual",
                ],
            ),
            field_expr(
                "stage3.spartan_shift.partial.NextIsFirstInSequence",
                "field.add",
                vec![
                    "stage3.spartan_shift.partial.NextIsVirtual",
                    "stage3.spartan_shift.term.NextIsFirstInSequence",
                ],
            ),
            field_expr(
                "stage3.spartan_shift.claim_expr",
                "field.add",
                vec![
                    "stage3.spartan_shift.partial.NextIsFirstInSequence",
                    "stage3.spartan_shift.term.NextIsNoop",
                ],
            ),
            field_expr(
                "stage3.instruction_input.term.LeftInstructionInput",
                "field.mul",
                vec![
                    "stage3.instruction_input.gamma",
                    "stage3.input.stage2.product_virtual.LeftInstructionInput",
                ],
            ),
            field_expr(
                "stage3.instruction_input.claim_expr",
                "field.add",
                vec![
                    "stage3.input.stage2.product_virtual.RightInstructionInput",
                    "stage3.instruction_input.term.LeftInstructionInput",
                ],
            ),
            field_expr(
                "stage3.registers.gamma2",
                "field.pow:2",
                vec!["stage3.registers.gamma"],
            ),
            field_expr(
                "stage3.registers.term.Rs1Value",
                "field.mul",
                vec!["stage3.registers.gamma", "stage3.input.stage1.Rs1Value"],
            ),
            field_expr(
                "stage3.registers.term.Rs2Value",
                "field.mul",
                vec!["stage3.registers.gamma2", "stage3.input.stage1.Rs2Value"],
            ),
            field_expr(
                "stage3.registers.partial.RdWriteValueRs1Value",
                "field.add",
                vec![
                    "stage3.input.stage1.RdWriteValue",
                    "stage3.registers.term.Rs1Value",
                ],
            ),
            field_expr(
                "stage3.registers.claim_expr",
                "field.add",
                vec![
                    "stage3.registers.partial.RdWriteValueRs1Value",
                    "stage3.registers.term.Rs2Value",
                ],
            ),
        ]);

        Box::leak(Box::new(Stage3CpuProgramPlan {
            params: Stage3Params {
                field: "bn254_fr",
                pcs: "dory",
                transcript: "blake2b_transcript",
            },
            steps: leak_slice(vec![
                Stage3ProgramStepPlan {
                    kind: "transcript_squeeze",
                    symbol: "stage3.spartan_shift.gamma",
                },
                Stage3ProgramStepPlan {
                    kind: "transcript_squeeze",
                    symbol: "stage3.instruction_input.gamma",
                },
                Stage3ProgramStepPlan {
                    kind: "transcript_squeeze",
                    symbol: "stage3.registers.gamma",
                },
                Stage3ProgramStepPlan {
                    kind: "sumcheck_driver",
                    symbol: "stage3.sumcheck",
                },
            ]),
            transcript_squeezes: leak_slice(vec![
                Stage3TranscriptSqueezePlan {
                    symbol: "stage3.spartan_shift.gamma",
                    label: "spartan_shift_gamma",
                    kind: "challenge_scalar",
                    count: 1,
                },
                Stage3TranscriptSqueezePlan {
                    symbol: "stage3.instruction_input.gamma",
                    label: "instruction_input_gamma",
                    kind: "challenge_scalar",
                    count: 1,
                },
                Stage3TranscriptSqueezePlan {
                    symbol: "stage3.registers.gamma",
                    label: "registers_gamma",
                    kind: "challenge_scalar",
                    count: 1,
                },
            ]),
            opening_inputs: leak_slice(stage3_opening_input_plans()),
            field_constants: leak_slice(vec![Stage3FieldConstantPlan {
                symbol: "stage3.field.one",
                field: "bn254_fr",
                value: 1,
            }]),
            field_exprs: exprs,
            kernels: leak_slice(vec![
                kernel(
                    "jolt.cpu.stage3.spartan_shift",
                    "jolt.stage3.spartan_shift",
                    "jolt_stage3_spartan_shift",
                ),
                kernel(
                    "jolt.cpu.stage3.instruction_input",
                    "jolt.stage3.instruction_input",
                    "jolt_stage3_instruction_input",
                ),
                kernel(
                    "jolt.cpu.stage3.registers_claim_reduction",
                    "jolt.stage3.registers_claim_reduction",
                    "jolt_stage3_registers_claim_reduction",
                ),
                kernel(
                    "jolt.cpu.stage3.batched",
                    "jolt.stage3.batched",
                    "jolt_stage3_batched",
                ),
            ]),
            claims: leak_slice(vec![
                claim(
                    "stage3.spartan_shift.input",
                    "jolt.cpu.stage3.spartan_shift",
                    "stage3.spartan_shift.claim_expr",
                    2,
                    vec![
                        "stage3.input.stage1.NextUnexpandedPC",
                        "stage3.input.stage1.NextPC",
                        "stage3.input.stage1.NextIsVirtual",
                        "stage3.input.stage1.NextIsFirstInSequence",
                        "stage3.input.stage2.product_virtual.NextIsNoop",
                    ],
                ),
                claim(
                    "stage3.instruction_input.input",
                    "jolt.cpu.stage3.instruction_input",
                    "stage3.instruction_input.claim_expr",
                    3,
                    vec![
                        "stage3.input.stage2.product_virtual.RightInstructionInput",
                        "stage3.input.stage2.product_virtual.LeftInstructionInput",
                    ],
                ),
                claim(
                    "stage3.registers_claim_reduction.input",
                    "jolt.cpu.stage3.registers_claim_reduction",
                    "stage3.registers.claim_expr",
                    2,
                    vec![
                        "stage3.input.stage1.RdWriteValue",
                        "stage3.input.stage1.Rs1Value",
                        "stage3.input.stage1.Rs2Value",
                    ],
                ),
            ]),
            batches: leak_slice(vec![Stage3SumcheckBatchPlan {
                symbol: "stage3.batch",
                stage: "stage3",
                proof_slot: "stage3.sumcheck",
                policy: "jolt_core_stage3_aligned",
                count: 3,
                ordered_claims: leak_slice(vec![
                    "stage3.spartan_shift.input",
                    "stage3.instruction_input.input",
                    "stage3.registers_claim_reduction.input",
                ]),
                claim_operands: leak_slice(vec![
                    "stage3.spartan_shift.input",
                    "stage3.instruction_input.input",
                    "stage3.registers_claim_reduction.input",
                ]),
                claim_label: "sumcheck_claim",
                round_label: "sumcheck_poly",
                round_schedule: leak_slice(vec![2]),
            }]),
            drivers: leak_slice(vec![Stage3SumcheckDriverPlan {
                symbol: "stage3.sumcheck",
                stage: "stage3",
                proof_slot: "stage3.sumcheck",
                kernel: Some("jolt.cpu.stage3.batched"),
                relation: None,
                batch: "stage3.batch",
                policy: "jolt_core_stage3_aligned",
                round_schedule: leak_slice(vec![2]),
                claim_label: "sumcheck_claim",
                round_label: "sumcheck_poly",
                num_rounds: 2,
                degree: 3,
            }]),
            instance_results: leak_slice(vec![
                instance(
                    "stage3.spartan_shift.instance",
                    "stage3.spartan_shift.input",
                    "jolt.stage3.spartan_shift",
                    0,
                    2,
                ),
                instance(
                    "stage3.instruction_input.instance",
                    "stage3.instruction_input.input",
                    "jolt.stage3.instruction_input",
                    1,
                    3,
                ),
                instance(
                    "stage3.registers_claim_reduction.instance",
                    "stage3.registers_claim_reduction.input",
                    "jolt.stage3.registers_claim_reduction",
                    2,
                    2,
                ),
            ]),
            evals: leak_slice(stage3_eval_plans()),
            point_slices: &[],
            point_concats: &[],
            opening_claims: leak_slice(stage3_opening_claim_plans()),
            opening_equalities: leak_slice(vec![
                Stage3OpeningClaimEqualityPlan {
                    symbol: "stage3.instruction_input.left_claim_consistency",
                    mode: "point_and_eval",
                    lhs: "stage3.input.stage2.product_virtual.LeftInstructionInput",
                    rhs: "stage3.input.stage2.instruction_lookup.LeftInstructionInput",
                },
                Stage3OpeningClaimEqualityPlan {
                    symbol: "stage3.instruction_input.right_claim_consistency",
                    mode: "point_and_eval",
                    lhs: "stage3.input.stage2.product_virtual.RightInstructionInput",
                    rhs: "stage3.input.stage2.instruction_lookup.RightInstructionInput",
                },
            ]),
            opening_batches: leak_slice(vec![Stage3OpeningBatchPlan {
                symbol: "stage3.openings",
                stage: "stage3",
                proof_slot: "stage3.openings",
                policy: "jolt_stage3_output_order",
                count: 16,
                ordered_claims: leak_slice(
                    stage3_opening_claim_plans()
                        .iter()
                        .map(|claim| claim.symbol)
                        .collect(),
                ),
                claim_operands: leak_slice(
                    stage3_opening_claim_plans()
                        .iter()
                        .map(|claim| claim.symbol)
                        .collect(),
                ),
            }]),
        }))
    }

    fn synthetic_cycles() -> [Stage3Cycle; 4] {
        [
            Stage3Cycle {
                unexpanded_pc: 10,
                pc: 0,
                is_virtual: false,
                is_first_in_sequence: false,
                is_noop: false,
                left_operand_is_rs1: true,
                rs1_value: 3,
                left_operand_is_pc: false,
                right_operand_is_rs2: true,
                rs2_value: 5,
                right_operand_is_imm: false,
                imm: 0,
                rd_write_value: 8,
            },
            Stage3Cycle {
                unexpanded_pc: 14,
                pc: 1,
                is_virtual: true,
                is_first_in_sequence: true,
                is_noop: false,
                left_operand_is_rs1: false,
                rs1_value: 0,
                left_operand_is_pc: true,
                right_operand_is_rs2: false,
                rs2_value: 0,
                right_operand_is_imm: true,
                imm: -7,
                rd_write_value: 21,
            },
            Stage3Cycle {
                unexpanded_pc: 18,
                pc: 2,
                is_virtual: false,
                is_first_in_sequence: false,
                is_noop: true,
                left_operand_is_rs1: false,
                rs1_value: 0,
                left_operand_is_pc: false,
                right_operand_is_rs2: false,
                rs2_value: 0,
                right_operand_is_imm: false,
                imm: 0,
                rd_write_value: 0,
            },
            Stage3Cycle {
                unexpanded_pc: 22,
                pc: 3,
                is_virtual: true,
                is_first_in_sequence: false,
                is_noop: false,
                left_operand_is_rs1: true,
                rs1_value: 9,
                left_operand_is_pc: false,
                right_operand_is_rs2: false,
                rs2_value: 0,
                right_operand_is_imm: true,
                imm: 11,
                rd_write_value: 20,
            },
        ]
    }

    fn synthetic_opening_inputs(cycles: &[Stage3Cycle]) -> Vec<Stage3OpeningInputValue<Fr>> {
        let r_outer = vec![Fr::from_u64(2), Fr::from_u64(3)];
        let r_product = vec![Fr::from_u64(5), Fr::from_u64(7)];
        let r_instruction = vec![Fr::from_u64(11), Fr::from_u64(13)];
        let r_registers = vec![Fr::from_u64(17), Fr::from_u64(19)];
        vec![
            opening(
                "stage3.input.stage1.NextUnexpandedPC",
                &r_outer,
                eq_plus_one_eval(cycles, &r_outer, |cycle| Fr::from_u64(cycle.unexpanded_pc)),
            ),
            opening(
                "stage3.input.stage1.NextPC",
                &r_outer,
                eq_plus_one_eval(cycles, &r_outer, |cycle| Fr::from_u64(cycle.pc)),
            ),
            opening(
                "stage3.input.stage1.NextIsVirtual",
                &r_outer,
                eq_plus_one_eval(cycles, &r_outer, |cycle| Fr::from_bool(cycle.is_virtual)),
            ),
            opening(
                "stage3.input.stage1.NextIsFirstInSequence",
                &r_outer,
                eq_plus_one_eval(cycles, &r_outer, |cycle| {
                    Fr::from_bool(cycle.is_first_in_sequence)
                }),
            ),
            opening(
                "stage3.input.stage2.product_virtual.NextIsNoop",
                &r_product,
                eq_plus_one_eval(cycles, &r_product, |cycle| Fr::from_bool(cycle.is_noop))
                    + r_product.iter().copied().product::<Fr>(),
            ),
            opening(
                "stage3.input.stage2.product_virtual.LeftInstructionInput",
                &r_instruction,
                mle_eval(cycles, &r_instruction, left_instruction_input),
            ),
            opening(
                "stage3.input.stage2.product_virtual.RightInstructionInput",
                &r_instruction,
                mle_eval(cycles, &r_instruction, right_instruction_input),
            ),
            opening(
                "stage3.input.stage2.instruction_lookup.LeftInstructionInput",
                &r_instruction,
                mle_eval(cycles, &r_instruction, left_instruction_input),
            ),
            opening(
                "stage3.input.stage2.instruction_lookup.RightInstructionInput",
                &r_instruction,
                mle_eval(cycles, &r_instruction, right_instruction_input),
            ),
            opening(
                "stage3.input.stage1.RdWriteValue",
                &r_registers,
                mle_eval(cycles, &r_registers, |cycle| {
                    Fr::from_u64(cycle.rd_write_value)
                }),
            ),
            opening(
                "stage3.input.stage1.Rs1Value",
                &r_registers,
                mle_eval(cycles, &r_registers, |cycle| Fr::from_u64(cycle.rs1_value)),
            ),
            opening(
                "stage3.input.stage1.Rs2Value",
                &r_registers,
                mle_eval(cycles, &r_registers, |cycle| Fr::from_u64(cycle.rs2_value)),
            ),
        ]
    }

    fn field_expr(
        symbol: &'static str,
        formula: &'static str,
        operands: Vec<&'static str>,
    ) -> Stage3FieldExprPlan {
        let operands = leak_slice(operands);
        Stage3FieldExprPlan {
            symbol,
            kind: "op",
            formula,
            operand_names: operands,
            operands,
        }
    }

    fn kernel(symbol: &'static str, relation: &'static str, abi: &'static str) -> Stage3KernelPlan {
        Stage3KernelPlan {
            symbol,
            relation,
            kind: "sumcheck",
            backend: "cpu",
            abi,
        }
    }

    fn claim(
        symbol: &'static str,
        kernel: &'static str,
        claim_value: &'static str,
        degree: usize,
        input_openings: Vec<&'static str>,
    ) -> Stage3SumcheckClaimPlan {
        Stage3SumcheckClaimPlan {
            symbol,
            stage: "stage3",
            domain: "jolt.trace_domain",
            num_rounds: 2,
            degree,
            claim: symbol,
            kernel: Some(kernel),
            relation: None,
            claim_value,
            input_openings: leak_slice(input_openings),
        }
    }

    fn instance(
        symbol: &'static str,
        claim: &'static str,
        relation: &'static str,
        index: usize,
        degree: usize,
    ) -> Stage3SumcheckInstanceResultPlan {
        Stage3SumcheckInstanceResultPlan {
            symbol,
            source: "stage3.sumcheck",
            claim,
            relation,
            index,
            point_arity: 2,
            num_rounds: 2,
            round_offset: 0,
            point_order: "reverse",
            degree,
        }
    }

    fn opening(symbol: &'static str, point: &[Fr], eval: Fr) -> Stage3OpeningInputValue<Fr> {
        Stage3OpeningInputValue {
            symbol,
            point: point.to_vec(),
            eval,
        }
    }

    fn mle_eval(cycles: &[Stage3Cycle], point: &[Fr], f: impl Fn(&Stage3Cycle) -> Fr) -> Fr {
        EqPolynomial::<Fr>::evals(point, None)
            .iter()
            .zip(cycles)
            .map(|(&weight, cycle)| weight * f(cycle))
            .sum()
    }

    fn eq_plus_one_eval(
        cycles: &[Stage3Cycle],
        point: &[Fr],
        f: impl Fn(&Stage3Cycle) -> Fr,
    ) -> Fr {
        EqPlusOnePolynomial::<Fr>::evals(point, None)
            .1
            .iter()
            .zip(cycles)
            .map(|(&weight, cycle)| weight * f(cycle))
            .sum()
    }

    fn left_instruction_input(cycle: &Stage3Cycle) -> Fr {
        Fr::from_bool(cycle.left_operand_is_rs1) * Fr::from_u64(cycle.rs1_value)
            + Fr::from_bool(cycle.left_operand_is_pc) * Fr::from_u64(cycle.unexpanded_pc)
    }

    fn right_instruction_input(cycle: &Stage3Cycle) -> Fr {
        Fr::from_bool(cycle.right_operand_is_rs2) * Fr::from_u64(cycle.rs2_value)
            + Fr::from_bool(cycle.right_operand_is_imm) * Fr::from_i128(cycle.imm)
    }

    fn stage3_opening_input_plans() -> Vec<Stage3OpeningInputPlan> {
        vec![
            opening_input_plan(
                "stage3.input.stage1.NextUnexpandedPC",
                "stage1",
                "NextUnexpandedPC",
            ),
            opening_input_plan("stage3.input.stage1.NextPC", "stage1", "NextPC"),
            opening_input_plan(
                "stage3.input.stage1.NextIsVirtual",
                "stage1",
                "NextIsVirtual",
            ),
            opening_input_plan(
                "stage3.input.stage1.NextIsFirstInSequence",
                "stage1",
                "NextIsFirstInSequence",
            ),
            opening_input_plan(
                "stage3.input.stage2.product_virtual.NextIsNoop",
                "stage2",
                "NextIsNoop",
            ),
            opening_input_plan(
                "stage3.input.stage2.product_virtual.LeftInstructionInput",
                "stage2",
                "LeftInstructionInput",
            ),
            opening_input_plan(
                "stage3.input.stage2.product_virtual.RightInstructionInput",
                "stage2",
                "RightInstructionInput",
            ),
            opening_input_plan(
                "stage3.input.stage2.instruction_lookup.LeftInstructionInput",
                "stage2",
                "LeftInstructionInput",
            ),
            opening_input_plan(
                "stage3.input.stage2.instruction_lookup.RightInstructionInput",
                "stage2",
                "RightInstructionInput",
            ),
            opening_input_plan("stage3.input.stage1.RdWriteValue", "stage1", "RdWriteValue"),
            opening_input_plan("stage3.input.stage1.Rs1Value", "stage1", "Rs1Value"),
            opening_input_plan("stage3.input.stage1.Rs2Value", "stage1", "Rs2Value"),
        ]
    }

    fn opening_input_plan(
        symbol: &'static str,
        source_stage: &'static str,
        oracle: &'static str,
    ) -> Stage3OpeningInputPlan {
        Stage3OpeningInputPlan {
            symbol,
            source_stage,
            source_claim: symbol,
            oracle,
            domain: "jolt.trace_domain",
            point_arity: 2,
            claim_kind: "virtual",
        }
    }

    fn stage3_eval_plans() -> Vec<Stage3SumcheckEvalPlan> {
        vec![
            eval("stage3.spartan_shift.eval.UnexpandedPC", "UnexpandedPC", 0),
            eval("stage3.spartan_shift.eval.PC", "PC", 1),
            eval(
                "stage3.spartan_shift.eval.OpFlagVirtualInstruction",
                "OpFlagVirtualInstruction",
                2,
            ),
            eval(
                "stage3.spartan_shift.eval.OpFlagIsFirstInSequence",
                "OpFlagIsFirstInSequence",
                3,
            ),
            eval(
                "stage3.spartan_shift.eval.InstructionFlagIsNoop",
                "InstructionFlagIsNoop",
                4,
            ),
            eval(
                "stage3.instruction_input.eval.InstructionFlagLeftOperandIsRs1Value",
                "InstructionFlagLeftOperandIsRs1Value",
                5,
            ),
            eval("stage3.instruction_input.eval.Rs1Value", "Rs1Value", 6),
            eval(
                "stage3.instruction_input.eval.InstructionFlagLeftOperandIsPC",
                "InstructionFlagLeftOperandIsPC",
                7,
            ),
            eval(
                "stage3.instruction_input.eval.UnexpandedPC",
                "UnexpandedPC",
                8,
            ),
            eval(
                "stage3.instruction_input.eval.InstructionFlagRightOperandIsRs2Value",
                "InstructionFlagRightOperandIsRs2Value",
                9,
            ),
            eval("stage3.instruction_input.eval.Rs2Value", "Rs2Value", 10),
            eval(
                "stage3.instruction_input.eval.InstructionFlagRightOperandIsImm",
                "InstructionFlagRightOperandIsImm",
                11,
            ),
            eval("stage3.instruction_input.eval.Imm", "Imm", 12),
            eval(
                "stage3.registers_claim_reduction.eval.RdWriteValue",
                "RdWriteValue",
                13,
            ),
            eval(
                "stage3.registers_claim_reduction.eval.Rs1Value",
                "Rs1Value",
                14,
            ),
            eval(
                "stage3.registers_claim_reduction.eval.Rs2Value",
                "Rs2Value",
                15,
            ),
        ]
    }

    fn eval(symbol: &'static str, oracle: &'static str, index: usize) -> Stage3SumcheckEvalPlan {
        Stage3SumcheckEvalPlan {
            symbol,
            source: "stage3.sumcheck",
            name: symbol,
            index,
            oracle,
        }
    }

    fn stage3_opening_claim_plans() -> Vec<Stage3OpeningClaimPlan> {
        stage3_eval_plans()
            .into_iter()
            .map(|eval| Stage3OpeningClaimPlan {
                symbol: eval.symbol.replace(".eval.", ".opening.").leak(),
                oracle: eval.oracle,
                domain: "jolt.trace_domain",
                point_arity: 2,
                claim_kind: "virtual",
                point_source: match eval.symbol {
                    name if name.starts_with("stage3.spartan_shift.") => {
                        "stage3.spartan_shift.instance"
                    }
                    name if name.starts_with("stage3.instruction_input.") => {
                        "stage3.instruction_input.instance"
                    }
                    _ => "stage3.registers_claim_reduction.instance",
                },
                eval_source: eval.symbol,
            })
            .collect()
    }

    fn leak_slice<T: 'static>(values: Vec<T>) -> &'static [T] {
        Box::leak(values.into_boxed_slice())
    }
}
