//! Stage 3 coarse-kernel ABI used by Bolt-generated Jolt prover code.

use std::error::Error;
use std::fmt::{self, Display, Formatter};

use crate::dense::{bind_dense_evals_reuse, DENSE_BIND_PAR_THRESHOLD};
use jolt_field::{Field, FieldAccumulator};
use jolt_poly::{EqPlusOnePolynomial, EqPolynomial, UnivariatePoly};
use jolt_sumcheck::SumcheckProof;
use jolt_transcript::{Label, LabelWithCount, Transcript};
use rayon::prelude::*;

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
    pub kernel: &'static str,
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
    pub kernel: &'static str,
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
    pub proof: SumcheckProof<F>,
}

#[derive(Clone, Debug)]
pub struct Stage3ChallengeVector<F: Field> {
    pub symbol: &'static str,
    pub values: Vec<F>,
}

#[derive(Clone, Debug)]
pub struct Stage3ExecutionArtifacts<F: Field> {
    pub challenge_vectors: Vec<Stage3ChallengeVector<F>>,
    pub sumchecks: Vec<Stage3SumcheckOutput<F>>,
    pub opening_batches: Vec<&'static Stage3OpeningBatchPlan>,
}

impl<F: Field> Default for Stage3ExecutionArtifacts<F> {
    fn default() -> Self {
        Self {
            challenge_vectors: Vec::new(),
            sumchecks: Vec::new(),
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
        instances.push(Stage3BatchedInstance {
            claim,
            relation: claim_relation(context.program, claim)?,
            offset: instance_round_offset(context.program, context.driver.symbol, claim.symbol)?,
            previous_claim: input_claims[index].mul_pow_2(max_rounds - claim.num_rounds),
            state: Stage3ProverInstanceState::new(context.program, claim, inputs, &store)?,
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
                    .round_poly(round - instance.offset, instance.previous_claim)?
            } else {
                UnivariatePoly::new(vec![instance.previous_claim * two_inv])
            };
            if poly.evaluate(F::zero()) + poly.evaluate(F::one()) != instance.previous_claim {
                return Err(Stage3KernelError::InvalidProof {
                    driver: context.driver.symbol,
                    reason: "batched instance round claim mismatch",
                });
            }
            individual_polys.push(poly);
        }
        let batched_poly = combine_univariate_polys(&individual_polys, &batching_coeffs);
        if batched_poly.evaluate(F::zero()) + batched_poly.evaluate(F::one()) != batched_claim {
            return Err(Stage3KernelError::InvalidProof {
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
        return Err(Stage3KernelError::InvalidProof {
            driver: context.driver.symbol,
            reason: "batched output claim mismatch",
        });
    }
    store.observe_sumcheck_values(context.program, context.driver.symbol, &point, &evals)?;
    append_opening_claims(context.program, &mut store, transcript, &evals)?;
    Ok(Stage3SumcheckOutput {
        driver: context.driver.symbol,
        point,
        evals,
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
    append_opening_claims(context.program, &mut store, transcript, &proof.evals)?;
    Ok(Stage3SumcheckOutput {
        driver: context.driver.symbol,
        point,
        evals: proof.evals.clone(),
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
    SumOfProducts(SumOfProductsState<F>),
}

impl<F: Field> Stage3ProverInstanceState<F> {
    fn new(
        program: &'static Stage3CpuProgramPlan,
        claim: &Stage3SumcheckClaimPlan,
        inputs: &Stage3ProverInputs<'_, F>,
        store: &Stage3ValueStore<F>,
    ) -> Result<Self, Stage3KernelError> {
        match claim_relation(program, claim)? {
            Stage3Relation::SpartanShift => spartan_shift_state(claim, inputs, store),
            Stage3Relation::InstructionInput => instruction_input_state(claim, inputs, store),
            Stage3Relation::RegistersClaimReduction => registers_state(claim, inputs, store),
            relation @ Stage3Relation::Batched => Err(Stage3KernelError::KernelNotImplemented {
                abi: relation.symbol(),
            }),
        }
        .map(Self::SumOfProducts)
    }

    fn round_poly(
        &self,
        _round: usize,
        _previous_claim: F,
    ) -> Result<UnivariatePoly<F>, Stage3KernelError> {
        match self {
            Self::SumOfProducts(state) => Ok(state.round_poly()),
        }
    }

    fn ingest_challenge(&mut self, challenge: F) {
        match self {
            Self::SumOfProducts(state) => state.bind(challenge),
        }
    }

    fn final_evals(
        &self,
        relation: Stage3Relation,
    ) -> Result<Vec<Stage3NamedEval<F>>, Stage3KernelError> {
        match self {
            Self::SumOfProducts(state) => state.final_evals(relation),
        }
    }
}

#[derive(Clone)]
struct SumOfProductsState<F: Field> {
    factors: Vec<Vec<F>>,
    factor_scratch: Vec<Vec<F>>,
    terms: Vec<ProductTerm<F>>,
    outputs: Vec<FactorOutput>,
    degree: usize,
    point: Vec<F>,
}

#[derive(Clone)]
struct ProductTerm<F: Field> {
    coefficient: F,
    factors: Vec<usize>,
}

#[derive(Clone, Copy)]
struct FactorOutput {
    name: &'static str,
    oracle: &'static str,
    factor: usize,
}

impl<F: Field> SumOfProductsState<F> {
    fn new(
        factors: Vec<Vec<F>>,
        terms: Vec<ProductTerm<F>>,
        outputs: Vec<FactorOutput>,
        degree: usize,
    ) -> Self {
        let factor_scratch = (0..factors.len()).map(|_| Vec::new()).collect();
        Self {
            factors,
            factor_scratch,
            terms,
            outputs,
            degree,
            point: Vec::new(),
        }
    }

    fn round_poly(&self) -> UnivariatePoly<F> {
        round_poly_from_sum_of_products(&self.factors, &self.terms, self.degree)
    }

    fn bind(&mut self, challenge: F) {
        for (factor, scratch) in self.factors.iter_mut().zip(&mut self.factor_scratch) {
            bind_dense_evals_reuse(factor, scratch, challenge);
        }
        self.point.push(challenge);
    }

    fn factor_eval(&self, index: usize, relation: Stage3Relation) -> Result<F, Stage3KernelError> {
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

fn spartan_shift_state<F: Field>(
    claim: &Stage3SumcheckClaimPlan,
    inputs: &Stage3ProverInputs<'_, F>,
    store: &Stage3ValueStore<F>,
) -> Result<SumOfProductsState<F>, Stage3KernelError> {
    let cycles = stage3_cycles(inputs, claim.num_rounds)?;
    let (_, eq_outer) =
        EqPlusOnePolynomial::<F>::evals(store.point("stage3.input.stage1.NextPC")?, None);
    let (_, eq_product) = EqPlusOnePolynomial::<F>::evals(
        store.point("stage3.input.stage2.product_virtual.NextIsNoop")?,
        None,
    );
    let one = F::one();
    let gamma = store.scalar("stage3.spartan_shift.gamma")?;
    let gamma2 = store.scalar("stage3.spartan_shift.gamma2")?;
    let gamma3 = store.scalar("stage3.spartan_shift.gamma3")?;
    let gamma4 = store.scalar("stage3.spartan_shift.gamma4")?;
    let factors = vec![
        eq_outer,
        eq_product,
        map_cycles(cycles, |cycle| F::from_u64(cycle.unexpanded_pc)),
        map_cycles(cycles, |cycle| F::from_u64(cycle.pc)),
        map_cycles(cycles, |cycle| F::from_bool(cycle.is_virtual)),
        map_cycles(cycles, |cycle| F::from_bool(cycle.is_first_in_sequence)),
        map_cycles(cycles, |cycle| one - F::from_bool(cycle.is_noop)),
        map_cycles(cycles, |cycle| F::from_bool(cycle.is_noop)),
    ];
    Ok(SumOfProductsState::new(
        factors,
        vec![
            ProductTerm {
                coefficient: F::one(),
                factors: vec![0, 2],
            },
            ProductTerm {
                coefficient: gamma,
                factors: vec![0, 3],
            },
            ProductTerm {
                coefficient: gamma2,
                factors: vec![0, 4],
            },
            ProductTerm {
                coefficient: gamma3,
                factors: vec![0, 5],
            },
            ProductTerm {
                coefficient: gamma4,
                factors: vec![1, 6],
            },
        ],
        vec![
            FactorOutput {
                name: "stage3.spartan_shift.eval.UnexpandedPC",
                oracle: "UnexpandedPC",
                factor: 2,
            },
            FactorOutput {
                name: "stage3.spartan_shift.eval.PC",
                oracle: "PC",
                factor: 3,
            },
            FactorOutput {
                name: "stage3.spartan_shift.eval.OpFlagVirtualInstruction",
                oracle: "OpFlagVirtualInstruction",
                factor: 4,
            },
            FactorOutput {
                name: "stage3.spartan_shift.eval.OpFlagIsFirstInSequence",
                oracle: "OpFlagIsFirstInSequence",
                factor: 5,
            },
            FactorOutput {
                name: "stage3.spartan_shift.eval.InstructionFlagIsNoop",
                oracle: "InstructionFlagIsNoop",
                factor: 7,
            },
        ],
        claim.degree,
    ))
}

fn instruction_input_state<F: Field>(
    claim: &Stage3SumcheckClaimPlan,
    inputs: &Stage3ProverInputs<'_, F>,
    store: &Stage3ValueStore<F>,
) -> Result<SumOfProductsState<F>, Stage3KernelError> {
    let cycles = stage3_cycles(inputs, claim.num_rounds)?;
    let eq = EqPolynomial::<F>::evals(
        store.point("stage3.input.stage2.product_virtual.LeftInstructionInput")?,
        None,
    );
    let gamma = store.scalar("stage3.instruction_input.gamma")?;
    let factors = vec![
        eq,
        map_cycles(cycles, |cycle| F::from_bool(cycle.right_operand_is_rs2)),
        map_cycles(cycles, |cycle| F::from_u64(cycle.rs2_value)),
        map_cycles(cycles, |cycle| F::from_bool(cycle.right_operand_is_imm)),
        map_cycles(cycles, |cycle| F::from_i128(cycle.imm)),
        map_cycles(cycles, |cycle| F::from_bool(cycle.left_operand_is_rs1)),
        map_cycles(cycles, |cycle| F::from_u64(cycle.rs1_value)),
        map_cycles(cycles, |cycle| F::from_bool(cycle.left_operand_is_pc)),
        map_cycles(cycles, |cycle| F::from_u64(cycle.unexpanded_pc)),
    ];
    Ok(SumOfProductsState::new(
        factors,
        vec![
            ProductTerm {
                coefficient: F::one(),
                factors: vec![0, 1, 2],
            },
            ProductTerm {
                coefficient: F::one(),
                factors: vec![0, 3, 4],
            },
            ProductTerm {
                coefficient: gamma,
                factors: vec![0, 5, 6],
            },
            ProductTerm {
                coefficient: gamma,
                factors: vec![0, 7, 8],
            },
        ],
        vec![
            FactorOutput {
                name: "stage3.instruction_input.eval.InstructionFlagLeftOperandIsRs1Value",
                oracle: "InstructionFlagLeftOperandIsRs1Value",
                factor: 5,
            },
            FactorOutput {
                name: "stage3.instruction_input.eval.Rs1Value",
                oracle: "Rs1Value",
                factor: 6,
            },
            FactorOutput {
                name: "stage3.instruction_input.eval.InstructionFlagLeftOperandIsPC",
                oracle: "InstructionFlagLeftOperandIsPC",
                factor: 7,
            },
            FactorOutput {
                name: "stage3.instruction_input.eval.UnexpandedPC",
                oracle: "UnexpandedPC",
                factor: 8,
            },
            FactorOutput {
                name: "stage3.instruction_input.eval.InstructionFlagRightOperandIsRs2Value",
                oracle: "InstructionFlagRightOperandIsRs2Value",
                factor: 1,
            },
            FactorOutput {
                name: "stage3.instruction_input.eval.Rs2Value",
                oracle: "Rs2Value",
                factor: 2,
            },
            FactorOutput {
                name: "stage3.instruction_input.eval.InstructionFlagRightOperandIsImm",
                oracle: "InstructionFlagRightOperandIsImm",
                factor: 3,
            },
            FactorOutput {
                name: "stage3.instruction_input.eval.Imm",
                oracle: "Imm",
                factor: 4,
            },
        ],
        claim.degree,
    ))
}

fn registers_state<F: Field>(
    claim: &Stage3SumcheckClaimPlan,
    inputs: &Stage3ProverInputs<'_, F>,
    store: &Stage3ValueStore<F>,
) -> Result<SumOfProductsState<F>, Stage3KernelError> {
    let cycles = stage3_cycles(inputs, claim.num_rounds)?;
    let eq = EqPolynomial::<F>::evals(store.point("stage3.input.stage1.RdWriteValue")?, None);
    let gamma = store.scalar("stage3.registers.gamma")?;
    let gamma2 = store.scalar("stage3.registers.gamma2")?;
    let factors = vec![
        eq,
        map_cycles(cycles, |cycle| F::from_u64(cycle.rd_write_value)),
        map_cycles(cycles, |cycle| F::from_u64(cycle.rs1_value)),
        map_cycles(cycles, |cycle| F::from_u64(cycle.rs2_value)),
    ];
    Ok(SumOfProductsState::new(
        factors,
        vec![
            ProductTerm {
                coefficient: F::one(),
                factors: vec![0, 1],
            },
            ProductTerm {
                coefficient: gamma,
                factors: vec![0, 2],
            },
            ProductTerm {
                coefficient: gamma2,
                factors: vec![0, 3],
            },
        ],
        vec![
            FactorOutput {
                name: "stage3.registers_claim_reduction.eval.RdWriteValue",
                oracle: "RdWriteValue",
                factor: 1,
            },
            FactorOutput {
                name: "stage3.registers_claim_reduction.eval.Rs1Value",
                oracle: "Rs1Value",
                factor: 2,
            },
            FactorOutput {
                name: "stage3.registers_claim_reduction.eval.Rs2Value",
                oracle: "Rs2Value",
                factor: 3,
            },
        ],
        claim.degree,
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

fn map_cycles<F: Field>(
    cycles: &[Stage3Cycle],
    f: impl Fn(&Stage3Cycle) -> F + Sync + Send,
) -> Vec<F> {
    cycles.par_iter().map(f).collect()
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
    let kernel = find_kernel(program, claim.kernel).ok_or(Stage3KernelError::MissingKernel {
        driver: claim.symbol,
        kernel: claim.kernel,
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

fn round_poly_from_sum_of_products<F: Field>(
    factors: &[Vec<F>],
    terms: &[ProductTerm<F>],
    degree: usize,
) -> UnivariatePoly<F> {
    if factors.is_empty() {
        return UnivariatePoly::zero();
    }
    let half = factors[0].len() / 2;
    let accumulators = if half >= DENSE_BIND_PAR_THRESHOLD {
        (0..half)
            .into_par_iter()
            .map(|row| sum_of_products_coefficients(factors, terms, row))
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
            let row_coeffs = sum_of_products_coefficients(factors, terms, row);
            for index in 0..total.len() {
                total[index].merge(row_coeffs[index]);
            }
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

fn sum_of_products_coefficients<F: Field>(
    factors: &[Vec<F>],
    terms: &[ProductTerm<F>],
    row: usize,
) -> [F::Accumulator; 4] {
    let mut accumulators = [F::Accumulator::default(); 4];
    for term in terms {
        let coefficients = product_term_coefficients(factors, &term.factors, row);
        for (accumulator, coefficient) in accumulators.iter_mut().zip(coefficients) {
            accumulator.fmadd(term.coefficient, coefficient);
        }
    }
    accumulators
}

fn product_term_coefficients<F: Field>(
    factors: &[Vec<F>],
    term_factors: &[usize],
    row: usize,
) -> [F; 4] {
    let mut coefficients = [F::zero(); 4];
    coefficients[0] = F::one();
    for (degree, &factor_index) in term_factors.iter().enumerate() {
        let low = factors[factor_index][2 * row];
        let delta = factors[factor_index][2 * row + 1] - low;
        for index in (0..=degree).rev() {
            coefficients[index + 1] += coefficients[index] * delta;
            coefficients[index] *= low;
        }
    }
    coefficients
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
) -> Result<(), Stage3KernelError>
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
    let mut seen = seed_stage3_opening_aliases(store, program);
    for batch in program.opening_batches {
        for symbol in batch.claim_operands {
            let claim =
                find_opening_claim(program, symbol).ok_or(Stage3KernelError::MissingClaim {
                    batch: batch.symbol,
                    claim: symbol,
                })?;
            let point = store.point(claim.point_source)?.to_vec();
            if has_seen_opening(&seen, claim.claim_kind, claim.oracle, &point) {
                continue;
            }
            let value = store.scalar(claim.eval_source)?;
            append_labeled_scalar(transcript, "opening_claim", &value);
            seen.push((claim.claim_kind, claim.oracle, point));
        }
    }
    Ok(())
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
    let kernel = find_kernel(program, driver.kernel).ok_or(Stage3KernelError::MissingKernel {
        driver: driver.symbol,
        kernel: driver.kernel,
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
                kernel: "jolt.cpu.stage3.batched",
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
            kernel,
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
