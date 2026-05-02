//! Stage 3 coarse-kernel ABI used by Bolt-generated Jolt prover code.

use std::error::Error;
use std::fmt::{self, Display, Formatter};

use crate::dense::{bind_dense_evals_reuse, DENSE_BIND_PAR_THRESHOLD};
use jolt_field::Field;
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
        verify_stage3_kernel(context, self.value_store(context.program)?, proof, transcript)
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
mod tests {
    use super::*;

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
}
