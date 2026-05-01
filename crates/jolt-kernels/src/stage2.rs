use std::cmp::Ordering;
use std::error::Error;
use std::fmt::{self, Display, Formatter};

use jolt_field::signed::{S128, S256};
use jolt_field::{Field, FieldAccumulator, Fr, Limbs};
use jolt_poly::lagrange::{interpolate_to_coeffs, lagrange_evals, lagrange_kernel_eval};
use jolt_poly::EqPolynomial;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::SumcheckProof;
use jolt_transcript::{Label, LabelWithCount, Transcript};
use rayon::prelude::*;

const PRODUCT_VIRTUAL_UNISKIP_DOMAIN_START: i64 = -1;
const PRODUCT_VIRTUAL_UNISKIP_DOMAIN_SIZE: usize = 3;
const PRODUCT_VIRTUAL_UNISKIP_DEGREE: usize = 2;
const PRODUCT_VIRTUAL_UNISKIP_EXTENDED_START: i64 = -(PRODUCT_VIRTUAL_UNISKIP_DEGREE as i64);
const PRODUCT_VIRTUAL_UNISKIP_EXTENDED_SIZE: usize = 2 * PRODUCT_VIRTUAL_UNISKIP_DEGREE + 1;
const PRODUCT_VIRTUAL_UNISKIP_NUM_COEFFS: usize = 3 * PRODUCT_VIRTUAL_UNISKIP_DEGREE + 1;
const PRODUCT_VIRTUAL_UNISKIP_TARGET_COEFFS: [[i32; PRODUCT_VIRTUAL_UNISKIP_DOMAIN_SIZE];
    PRODUCT_VIRTUAL_UNISKIP_DEGREE] = [[3, -3, 1], [1, -3, 3]];
const DENSE_BIND_PAR_THRESHOLD: usize = 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage2ExecutionMode {
    Prover,
    Verifier,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage2Relation {
    ProductVirtualUniskip,
    RamReadWrite,
    ProductVirtualRemainder,
    InstructionLookupClaimReduction,
    RamRafEvaluation,
    RamOutputCheck,
    Batched,
}

impl Stage2Relation {
    pub fn from_symbol(symbol: &str) -> Option<Self> {
        match symbol {
            "jolt.stage2.product_virtual.uniskip" => Some(Self::ProductVirtualUniskip),
            "jolt.stage2.ram.read_write" => Some(Self::RamReadWrite),
            "jolt.stage2.product_virtual.remainder" => Some(Self::ProductVirtualRemainder),
            "jolt.stage2.instruction_lookup.claim_reduction" => {
                Some(Self::InstructionLookupClaimReduction)
            }
            "jolt.stage2.ram.raf_evaluation" => Some(Self::RamRafEvaluation),
            "jolt.stage2.ram.output_check" => Some(Self::RamOutputCheck),
            "jolt.stage2.batched" => Some(Self::Batched),
            _ => None,
        }
    }

    pub fn symbol(self) -> &'static str {
        match self {
            Self::ProductVirtualUniskip => "jolt.stage2.product_virtual.uniskip",
            Self::RamReadWrite => "jolt.stage2.ram.read_write",
            Self::ProductVirtualRemainder => "jolt.stage2.product_virtual.remainder",
            Self::InstructionLookupClaimReduction => {
                "jolt.stage2.instruction_lookup.claim_reduction"
            }
            Self::RamRafEvaluation => "jolt.stage2.ram.raf_evaluation",
            Self::RamOutputCheck => "jolt.stage2.ram.output_check",
            Self::Batched => "jolt.stage2.batched",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage2KernelAbi {
    ProductVirtualUniskip,
    RamReadWrite,
    ProductVirtualRemainder,
    InstructionLookupClaimReduction,
    RamRafEvaluation,
    RamOutputCheck,
    Batched,
}

impl Stage2KernelAbi {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "jolt_stage2_product_virtual_uniskip" => Some(Self::ProductVirtualUniskip),
            "jolt_stage2_ram_read_write" => Some(Self::RamReadWrite),
            "jolt_stage2_product_virtual_remainder" => Some(Self::ProductVirtualRemainder),
            "jolt_stage2_instruction_lookup_claim_reduction" => {
                Some(Self::InstructionLookupClaimReduction)
            }
            "jolt_stage2_ram_raf_evaluation" => Some(Self::RamRafEvaluation),
            "jolt_stage2_ram_output_check" => Some(Self::RamOutputCheck),
            "jolt_stage2_batched" => Some(Self::Batched),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::ProductVirtualUniskip => "jolt_stage2_product_virtual_uniskip",
            Self::RamReadWrite => "jolt_stage2_ram_read_write",
            Self::ProductVirtualRemainder => "jolt_stage2_product_virtual_remainder",
            Self::InstructionLookupClaimReduction => {
                "jolt_stage2_instruction_lookup_claim_reduction"
            }
            Self::RamRafEvaluation => "jolt_stage2_ram_raf_evaluation",
            Self::RamOutputCheck => "jolt_stage2_ram_output_check",
            Self::Batched => "jolt_stage2_batched",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage2Params {
    pub field: &'static str,
    pub pcs: &'static str,
    pub transcript: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage2KernelPlan {
    pub symbol: &'static str,
    pub relation: &'static str,
    pub kind: &'static str,
    pub backend: &'static str,
    pub abi: &'static str,
}

impl Stage2KernelPlan {
    pub fn relation_kind(&self) -> Result<Stage2Relation, Stage2KernelError> {
        Stage2Relation::from_symbol(self.relation).ok_or(Stage2KernelError::UnknownRelation {
            relation: self.relation,
        })
    }

    pub fn abi_kind(&self) -> Result<Stage2KernelAbi, Stage2KernelError> {
        Stage2KernelAbi::from_name(self.abi)
            .ok_or(Stage2KernelError::UnknownKernelAbi { abi: self.abi })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage2TranscriptSqueezePlan {
    pub symbol: &'static str,
    pub label: &'static str,
    pub kind: &'static str,
    pub count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage2OpeningInputPlan {
    pub symbol: &'static str,
    pub source_stage: &'static str,
    pub source_claim: &'static str,
    pub oracle: &'static str,
    pub domain: &'static str,
    pub point_arity: usize,
    pub claim_kind: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage2FieldConstantPlan {
    pub symbol: &'static str,
    pub field: &'static str,
    pub value: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage2ChallengeExtractPlan {
    pub symbol: &'static str,
    pub source: &'static str,
    pub index: usize,
    pub challenge_source: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage2FieldExprPlan {
    pub symbol: &'static str,
    pub kind: &'static str,
    pub formula: &'static str,
    pub operand_names: &'static [&'static str],
    pub operands: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage2SumcheckClaimPlan {
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
pub struct Stage2SumcheckBatchPlan {
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
pub struct Stage2SumcheckDriverPlan {
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
pub struct Stage2SumcheckInstanceResultPlan {
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
pub struct Stage2SumcheckEvalPlan {
    pub symbol: &'static str,
    pub source: &'static str,
    pub name: &'static str,
    pub index: usize,
    pub oracle: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage2PointSlicePlan {
    pub symbol: &'static str,
    pub source: &'static str,
    pub offset: usize,
    pub length: usize,
    pub input: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage2PointConcatPlan {
    pub symbol: &'static str,
    pub layout: &'static str,
    pub arity: usize,
    pub inputs: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage2OpeningClaimPlan {
    pub symbol: &'static str,
    pub oracle: &'static str,
    pub domain: &'static str,
    pub point_arity: usize,
    pub claim_kind: &'static str,
    pub point_source: &'static str,
    pub eval_source: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage2OpeningBatchPlan {
    pub symbol: &'static str,
    pub stage: &'static str,
    pub proof_slot: &'static str,
    pub policy: &'static str,
    pub count: usize,
    pub ordered_claims: &'static [&'static str],
    pub claim_operands: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage2ProgramStepPlan {
    pub kind: &'static str,
    pub symbol: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage2CpuProgramPlan {
    pub params: Stage2Params,
    pub steps: &'static [Stage2ProgramStepPlan],
    pub transcript_squeezes: &'static [Stage2TranscriptSqueezePlan],
    pub opening_inputs: &'static [Stage2OpeningInputPlan],
    pub field_constants: &'static [Stage2FieldConstantPlan],
    pub challenge_extracts: &'static [Stage2ChallengeExtractPlan],
    pub field_exprs: &'static [Stage2FieldExprPlan],
    pub kernels: &'static [Stage2KernelPlan],
    pub claims: &'static [Stage2SumcheckClaimPlan],
    pub batches: &'static [Stage2SumcheckBatchPlan],
    pub drivers: &'static [Stage2SumcheckDriverPlan],
    pub instance_results: &'static [Stage2SumcheckInstanceResultPlan],
    pub evals: &'static [Stage2SumcheckEvalPlan],
    pub point_slices: &'static [Stage2PointSlicePlan],
    pub point_concats: &'static [Stage2PointConcatPlan],
    pub opening_claims: &'static [Stage2OpeningClaimPlan],
    pub opening_batches: &'static [Stage2OpeningBatchPlan],
}

impl Stage2CpuProgramPlan {
    pub fn kernel(&self, symbol: &str) -> Option<&Stage2KernelPlan> {
        find_kernel(self, symbol)
    }

    pub fn batch(&self, symbol: &str) -> Option<&Stage2SumcheckBatchPlan> {
        find_batch(self, symbol)
    }

    pub fn claim(&self, symbol: &str) -> Option<&Stage2SumcheckClaimPlan> {
        self.claims.iter().find(|claim| claim.symbol == symbol)
    }

    pub fn evals_for_driver<'a>(
        &'a self,
        driver: &'a str,
    ) -> impl Iterator<Item = &'a Stage2SumcheckEvalPlan> + 'a {
        self.evals.iter().filter(move |eval| eval.source == driver)
    }

    pub fn instance_results_for_driver<'a>(
        &'a self,
        driver: &'a str,
    ) -> impl Iterator<Item = &'a Stage2SumcheckInstanceResultPlan> + 'a {
        self.instance_results
            .iter()
            .filter(move |instance| instance.source == driver)
    }
}

#[derive(Clone, Debug)]
pub struct Stage2NamedEval<F: Field> {
    pub name: &'static str,
    pub oracle: &'static str,
    pub value: F,
}

#[derive(Clone, Debug)]
pub struct Stage2SumcheckOutput<F: Field> {
    pub driver: &'static str,
    pub point: Vec<F>,
    pub evals: Vec<Stage2NamedEval<F>>,
    pub proof: SumcheckProof<F>,
}

#[derive(Clone, Debug)]
pub struct Stage2ChallengeVector<F: Field> {
    pub symbol: &'static str,
    pub values: Vec<F>,
}

#[derive(Clone, Debug)]
pub struct Stage2ExecutionArtifacts<F: Field> {
    pub challenge_vectors: Vec<Stage2ChallengeVector<F>>,
    pub sumchecks: Vec<Stage2SumcheckOutput<F>>,
    pub opening_batches: Vec<&'static Stage2OpeningBatchPlan>,
}

impl<F: Field> Default for Stage2ExecutionArtifacts<F> {
    fn default() -> Self {
        Self {
            challenge_vectors: Vec::new(),
            sumchecks: Vec::new(),
            opening_batches: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Stage2Proof<F: Field> {
    pub sumchecks: Vec<Stage2SumcheckOutput<F>>,
}

impl<F: Field> From<Stage2ExecutionArtifacts<F>> for Stage2Proof<F> {
    fn from(artifacts: Stage2ExecutionArtifacts<F>) -> Self {
        Self {
            sumchecks: artifacts.sumchecks,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Stage2ScalarValue<F: Field> {
    pub symbol: &'static str,
    pub value: F,
}

#[derive(Clone, Debug)]
pub struct Stage2PointValue<F: Field> {
    pub symbol: &'static str,
    pub point: Vec<F>,
}

#[derive(Clone, Debug)]
pub struct Stage2OpeningInputValue<F: Field> {
    pub symbol: &'static str,
    pub point: Vec<F>,
    pub eval: F,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage2ProductVirtualCycle {
    pub instruction_left_input: u64,
    pub instruction_right_input: i128,
    pub should_branch_lookup_output: u64,
    pub write_lookup_output_to_rd_flag: bool,
    pub jump_flag: bool,
    pub should_branch_flag: bool,
    pub not_next_noop: bool,
    pub virtual_instruction_flag: bool,
}

impl Stage2ProductVirtualCycle {
    pub fn padding() -> Self {
        Self {
            instruction_left_input: 0,
            instruction_right_input: 0,
            should_branch_lookup_output: 0,
            write_lookup_output_to_rd_flag: false,
            jump_flag: false,
            should_branch_flag: false,
            not_next_noop: false,
            virtual_instruction_flag: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage2InstructionLookupCycle {
    pub lookup_output: u64,
    pub left_lookup_operand: u64,
    pub right_lookup_operand: u128,
    pub left_instruction_input: u64,
    pub right_instruction_input: i128,
}

impl Stage2InstructionLookupCycle {
    pub fn padding() -> Self {
        Self {
            lookup_output: 0,
            left_lookup_operand: 0,
            right_lookup_operand: 0,
            left_instruction_input: 0,
            right_instruction_input: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage2RamAccess {
    pub remapped_address: Option<usize>,
    pub read_value: u64,
    pub write_value: u64,
}

impl Stage2RamAccess {
    pub fn noop() -> Self {
        Self {
            remapped_address: None,
            read_value: 0,
            write_value: 0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Stage2RamOutputLayout {
    pub io_start: usize,
    pub io_end: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct Stage2RamData<'a> {
    pub log_k: usize,
    pub start_address: u64,
    pub initial_ram: &'a [u64],
    pub final_ram: &'a [u64],
    pub accesses: &'a [Stage2RamAccess],
    pub output_layout: Option<Stage2RamOutputLayout>,
}

#[derive(Clone, Debug, Default)]
pub struct Stage2ValueStore<F: Field> {
    scalars: Vec<Stage2ScalarValue<F>>,
    points: Vec<Stage2PointValue<F>>,
}

impl<F: Field> Stage2ValueStore<F> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_opening_inputs(inputs: &[Stage2OpeningInputValue<F>]) -> Self {
        let mut store = Self::new();
        store.insert_opening_inputs(inputs);
        store
    }

    pub fn insert_opening_inputs(&mut self, inputs: &[Stage2OpeningInputValue<F>]) {
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
            self.scalars.push(Stage2ScalarValue { symbol, value });
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
            self.points.push(Stage2PointValue { symbol, point });
        }
    }

    pub fn try_scalar(&self, symbol: &str) -> Option<F> {
        self.scalars
            .iter()
            .find(|value| value.symbol == symbol)
            .map(|value| value.value)
    }

    pub fn scalar(&self, symbol: &'static str) -> Result<F, Stage2KernelError> {
        self.try_scalar(symbol)
            .ok_or(Stage2KernelError::MissingValue { symbol })
    }

    pub fn point(&self, symbol: &'static str) -> Result<&[F], Stage2KernelError> {
        self.try_point(symbol)
            .ok_or(Stage2KernelError::MissingValue { symbol })
    }

    pub fn try_point(&self, symbol: &str) -> Option<&[F]> {
        self.points
            .iter()
            .find(|value| value.symbol == symbol)
            .map(|value| value.point.as_slice())
    }

    pub fn seed_constants(
        &mut self,
        program: &'static Stage2CpuProgramPlan,
    ) -> Result<(), Stage2KernelError> {
        for constant in program.field_constants {
            self.insert_scalar(constant.symbol, F::from_u64(constant.value as u64));
        }
        Ok(())
    }

    pub fn observe_challenge_vector(
        &mut self,
        program: &'static Stage2CpuProgramPlan,
        plan: &'static Stage2TranscriptSqueezePlan,
        values: &[F],
    ) -> Result<(), Stage2KernelError> {
        for extract in program
            .challenge_extracts
            .iter()
            .filter(|extract| extract.challenge_source == plan.symbol)
        {
            let value = values.get(extract.index).copied().ok_or(
                Stage2KernelError::InvalidInputLength {
                    input: extract.symbol,
                    expected: extract.index + 1,
                    actual: values.len(),
                },
            )?;
            self.insert_scalar(extract.symbol, value);
        }
        Ok(())
    }

    pub fn observe_sumcheck_output(
        &mut self,
        program: &'static Stage2CpuProgramPlan,
        output: &Stage2SumcheckOutput<F>,
    ) -> Result<(), Stage2KernelError> {
        self.observe_sumcheck_values(program, output.driver, &output.point, &output.evals)
    }

    pub fn observe_sumcheck_values(
        &mut self,
        program: &'static Stage2CpuProgramPlan,
        driver: &'static str,
        point: &[F],
        evals: &[Stage2NamedEval<F>],
    ) -> Result<(), Stage2KernelError> {
        self.insert_point(driver, point.to_vec());
        for instance in program.instance_results_for_driver(driver) {
            let end = instance.round_offset + instance.point_arity;
            let mut point = point
                .get(instance.round_offset..end)
                .ok_or(Stage2KernelError::InvalidInputLength {
                    input: instance.symbol,
                    expected: end,
                    actual: point.len(),
                })?
                .to_vec();
            match instance.point_order {
                "as_is" => {}
                "reverse" => point.reverse(),
                _ => {
                    return Err(Stage2KernelError::InvalidProof {
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
                .ok_or(Stage2KernelError::MissingValue {
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
        program: &'static Stage2CpuProgramPlan,
    ) -> Result<usize, Stage2KernelError> {
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
                    .ok_or(Stage2KernelError::InvalidInputLength {
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
        program: &'static Stage2CpuProgramPlan,
    ) -> Result<usize, Stage2KernelError> {
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
                let value = evaluate_stage2_field_expr(expr, &operands)?;
                self.insert_scalar(expr.symbol, value);
                progress += 1;
            }
            inserted += progress;
            if progress == 0 {
                return Ok(inserted);
            }
        }
    }

    pub fn claim_value(
        &mut self,
        program: &'static Stage2CpuProgramPlan,
        claim: &Stage2SumcheckClaimPlan,
    ) -> Result<F, Stage2KernelError> {
        let _ = self.evaluate_available_field_exprs(program)?;
        self.scalar(claim.claim_value)
    }

    pub fn batch_claim_values(
        &mut self,
        program: &'static Stage2CpuProgramPlan,
        batch: &Stage2SumcheckBatchPlan,
    ) -> Result<Vec<F>, Stage2KernelError> {
        batch
            .claim_operands
            .iter()
            .map(|symbol| {
                let claim = program
                    .claim(symbol)
                    .ok_or(Stage2KernelError::MissingClaim {
                        batch: batch.symbol,
                        claim: symbol,
                    })?;
                self.claim_value(program, claim)
            })
            .collect()
    }

    fn try_expr_operands(&self, expr: &Stage2FieldExprPlan) -> Option<Vec<F>> {
        expr.operands
            .iter()
            .map(|operand| self.try_scalar(operand))
            .collect()
    }

    fn try_concat_point(&self, concat: &Stage2PointConcatPlan) -> Option<Vec<F>> {
        let mut point = Vec::with_capacity(concat.arity);
        for input in concat.inputs {
            point.extend_from_slice(self.try_point(input)?);
        }
        Some(point)
    }
}

pub fn evaluate_stage2_field_expr<F: Field>(
    expr: &Stage2FieldExprPlan,
    operands: &[F],
) -> Result<F, Stage2KernelError> {
    match expr.formula {
        "opening_eval" => single_operand(expr.symbol, operands),
        "jolt_stage2_product_virtual_uniskip_input" => {
            require_operand_count(expr.symbol, 4, operands.len())?;
            let weights = lagrange_evals(
                PRODUCT_VIRTUAL_UNISKIP_DOMAIN_START,
                PRODUCT_VIRTUAL_UNISKIP_DOMAIN_SIZE,
                operands[0],
            );
            Ok(weights[0] * operands[1] + weights[1] * operands[2] + weights[2] * operands[3])
        }
        "jolt_stage2_ram_read_write_input" => {
            require_operand_count(expr.symbol, 3, operands.len())?;
            Ok(operands[1] + operands[0] * operands[2])
        }
        "jolt_stage2_instruction_lookup_input" => {
            require_operand_count(expr.symbol, 6, operands.len())?;
            let gamma = operands[0];
            let gamma_sqr = gamma.square();
            let gamma_cub = gamma_sqr * gamma;
            let gamma_quart = gamma_sqr.square();
            Ok(operands[1]
                + gamma * operands[2]
                + gamma_sqr * operands[3]
                + gamma_cub * operands[4]
                + gamma_quart * operands[5])
        }
        formula => Err(Stage2KernelError::UnsupportedFieldExpr {
            symbol: expr.symbol,
            formula,
        }),
    }
}

fn single_operand<F: Field>(symbol: &'static str, operands: &[F]) -> Result<F, Stage2KernelError> {
    require_operand_count(symbol, 1, operands.len())?;
    Ok(operands[0])
}

fn require_operand_count(
    input: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), Stage2KernelError> {
    if expected == actual {
        Ok(())
    } else {
        Err(Stage2KernelError::InvalidInputLength {
            input,
            expected,
            actual,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Stage2KernelContext<'a> {
    pub mode: Stage2ExecutionMode,
    pub program: &'static Stage2CpuProgramPlan,
    pub kernel: &'a Stage2KernelPlan,
    pub batch: &'a Stage2SumcheckBatchPlan,
    pub driver: &'a Stage2SumcheckDriverPlan,
}

impl Stage2KernelContext<'_> {
    pub fn relation_kind(&self) -> Result<Stage2Relation, Stage2KernelError> {
        self.kernel.relation_kind()
    }

    pub fn abi_kind(&self) -> Result<Stage2KernelAbi, Stage2KernelError> {
        self.kernel.abi_kind()
    }

    pub fn batch_claims(&self) -> Result<Vec<&'static Stage2SumcheckClaimPlan>, Stage2KernelError> {
        self.batch
            .claim_operands
            .iter()
            .map(|symbol| {
                self.program
                    .claim(symbol)
                    .ok_or(Stage2KernelError::MissingClaim {
                        batch: self.batch.symbol,
                        claim: symbol,
                    })
            })
            .collect()
    }
}

pub trait Stage2KernelExecutor<F: Field> {
    fn observe_challenge_vector(
        &mut self,
        _plan: &'static Stage2TranscriptSqueezePlan,
        _values: &[F],
    ) -> Result<(), Stage2KernelError> {
        Ok(())
    }

    fn observe_sumcheck_output(
        &mut self,
        _output: &Stage2SumcheckOutput<F>,
    ) -> Result<(), Stage2KernelError> {
        Ok(())
    }

    fn prove_sumcheck<T>(
        &mut self,
        context: Stage2KernelContext<'_>,
        transcript: &mut T,
    ) -> Result<Stage2SumcheckOutput<F>, Stage2KernelError>
    where
        T: Transcript<Challenge = F>;

    fn verify_sumcheck<T>(
        &mut self,
        context: Stage2KernelContext<'_>,
        transcript: &mut T,
    ) -> Result<Stage2SumcheckOutput<F>, Stage2KernelError>
    where
        T: Transcript<Challenge = F>;
}

#[derive(Clone, Debug, Default)]
pub struct UnsupportedStage2KernelExecutor;

impl<F: Field> Stage2KernelExecutor<F> for UnsupportedStage2KernelExecutor {
    fn prove_sumcheck<T>(
        &mut self,
        context: Stage2KernelContext<'_>,
        _transcript: &mut T,
    ) -> Result<Stage2SumcheckOutput<F>, Stage2KernelError>
    where
        T: Transcript<Challenge = F>,
    {
        Err(Stage2KernelError::KernelNotImplemented {
            abi: context.kernel.abi,
        })
    }

    fn verify_sumcheck<T>(
        &mut self,
        context: Stage2KernelContext<'_>,
        _transcript: &mut T,
    ) -> Result<Stage2SumcheckOutput<F>, Stage2KernelError>
    where
        T: Transcript<Challenge = F>,
    {
        Err(Stage2KernelError::KernelNotImplemented {
            abi: context.kernel.abi,
        })
    }
}

#[derive(Clone, Copy)]
pub struct Stage2ProverInputs<'a, F: Field> {
    pub opening_inputs: &'a [Stage2OpeningInputValue<F>],
    pub product_uniskip_extended_evals: Option<&'a [F]>,
    pub product_virtual_cycles: Option<&'a [Stage2ProductVirtualCycle]>,
    pub instruction_lookup_cycles: Option<&'a [Stage2InstructionLookupCycle]>,
    pub ram: Option<&'a Stage2RamData<'a>>,
}

impl<'a, F: Field> Stage2ProverInputs<'a, F> {
    pub fn new(opening_inputs: &'a [Stage2OpeningInputValue<F>]) -> Self {
        Self {
            opening_inputs,
            product_uniskip_extended_evals: None,
            product_virtual_cycles: None,
            instruction_lookup_cycles: None,
            ram: None,
        }
    }

    pub fn empty() -> Self {
        Self {
            opening_inputs: &[],
            product_uniskip_extended_evals: None,
            product_virtual_cycles: None,
            instruction_lookup_cycles: None,
            ram: None,
        }
    }

    pub fn with_product_uniskip_extended_evals(mut self, evaluations: &'a [F]) -> Self {
        self.product_uniskip_extended_evals = Some(evaluations);
        self
    }

    pub fn with_product_virtual_cycles(mut self, cycles: &'a [Stage2ProductVirtualCycle]) -> Self {
        self.product_virtual_cycles = Some(cycles);
        self
    }

    pub fn with_instruction_lookup_cycles(
        mut self,
        cycles: &'a [Stage2InstructionLookupCycle],
    ) -> Self {
        self.instruction_lookup_cycles = Some(cycles);
        self
    }

    pub fn with_ram_data(mut self, ram: &'a Stage2RamData<'a>) -> Self {
        self.ram = Some(ram);
        self
    }
}

#[derive(Clone)]
pub struct Stage2ProverKernelExecutor<'a, F: Field> {
    pub inputs: Stage2ProverInputs<'a, F>,
    challenge_vectors: Vec<Stage2ChallengeVector<F>>,
    completed_sumchecks: Vec<Stage2SumcheckOutput<F>>,
}

impl<'a, F: Field> Stage2ProverKernelExecutor<'a, F> {
    pub fn new(inputs: Stage2ProverInputs<'a, F>) -> Self {
        Self {
            inputs,
            challenge_vectors: Vec::new(),
            completed_sumchecks: Vec::new(),
        }
    }

    fn value_store(
        &self,
        program: &'static Stage2CpuProgramPlan,
    ) -> Result<Stage2ValueStore<F>, Stage2KernelError> {
        let mut store = Stage2ValueStore::with_opening_inputs(self.inputs.opening_inputs);
        store.seed_constants(program)?;
        for challenge in &self.challenge_vectors {
            store.insert_point(challenge.symbol, challenge.values.clone());
            for extract in program
                .challenge_extracts
                .iter()
                .filter(|extract| extract.challenge_source == challenge.symbol)
            {
                let value = challenge.values.get(extract.index).copied().ok_or(
                    Stage2KernelError::InvalidInputLength {
                        input: extract.symbol,
                        expected: extract.index + 1,
                        actual: challenge.values.len(),
                    },
                )?;
                store.insert_scalar(extract.symbol, value);
            }
        }
        for output in &self.completed_sumchecks {
            store.observe_sumcheck_output(program, output)?;
        }
        let _ = store.evaluate_available_points(program)?;
        let _ = store.evaluate_available_field_exprs(program)?;
        Ok(store)
    }
}

impl<F: Field> Stage2KernelExecutor<F> for Stage2ProverKernelExecutor<'_, F> {
    fn observe_challenge_vector(
        &mut self,
        plan: &'static Stage2TranscriptSqueezePlan,
        values: &[F],
    ) -> Result<(), Stage2KernelError> {
        self.challenge_vectors.push(Stage2ChallengeVector {
            symbol: plan.symbol,
            values: values.to_vec(),
        });
        Ok(())
    }

    fn observe_sumcheck_output(
        &mut self,
        output: &Stage2SumcheckOutput<F>,
    ) -> Result<(), Stage2KernelError> {
        self.completed_sumchecks.push(output.clone());
        Ok(())
    }

    fn prove_sumcheck<T>(
        &mut self,
        context: Stage2KernelContext<'_>,
        transcript: &mut T,
    ) -> Result<Stage2SumcheckOutput<F>, Stage2KernelError>
    where
        T: Transcript<Challenge = F>,
    {
        prove_stage2_kernel(
            context,
            &self.inputs,
            self.value_store(context.program)?,
            transcript,
        )
    }

    fn verify_sumcheck<T>(
        &mut self,
        context: Stage2KernelContext<'_>,
        _transcript: &mut T,
    ) -> Result<Stage2SumcheckOutput<F>, Stage2KernelError>
    where
        T: Transcript<Challenge = F>,
    {
        Err(Stage2KernelError::WrongExecutorMode {
            driver: context.driver.symbol,
            expected: Stage2ExecutionMode::Prover,
            actual: Stage2ExecutionMode::Verifier,
        })
    }
}

#[derive(Clone)]
pub struct Stage2VerifierKernelExecutor<'a, F: Field> {
    pub proof: &'a Stage2Proof<F>,
    pub opening_inputs: &'a [Stage2OpeningInputValue<F>],
    pub ram: Option<&'a Stage2RamData<'a>>,
    pub cursor: usize,
    challenge_vectors: Vec<Stage2ChallengeVector<F>>,
    completed_sumchecks: Vec<Stage2SumcheckOutput<F>>,
}

impl<'a, F: Field> Stage2VerifierKernelExecutor<'a, F> {
    pub fn new(
        proof: &'a Stage2Proof<F>,
        opening_inputs: &'a [Stage2OpeningInputValue<F>],
    ) -> Self {
        Self {
            proof,
            opening_inputs,
            ram: None,
            cursor: 0,
            challenge_vectors: Vec::new(),
            completed_sumchecks: Vec::new(),
        }
    }

    pub fn with_ram_data(mut self, ram: &'a Stage2RamData<'a>) -> Self {
        self.ram = Some(ram);
        self
    }

    fn value_store(
        &self,
        program: &'static Stage2CpuProgramPlan,
    ) -> Result<Stage2ValueStore<F>, Stage2KernelError> {
        let mut store = Stage2ValueStore::with_opening_inputs(self.opening_inputs);
        store.seed_constants(program)?;
        for challenge in &self.challenge_vectors {
            store.insert_point(challenge.symbol, challenge.values.clone());
            for extract in program
                .challenge_extracts
                .iter()
                .filter(|extract| extract.challenge_source == challenge.symbol)
            {
                let value = challenge.values.get(extract.index).copied().ok_or(
                    Stage2KernelError::InvalidInputLength {
                        input: extract.symbol,
                        expected: extract.index + 1,
                        actual: challenge.values.len(),
                    },
                )?;
                store.insert_scalar(extract.symbol, value);
            }
        }
        for output in &self.completed_sumchecks {
            store.observe_sumcheck_output(program, output)?;
        }
        let _ = store.evaluate_available_points(program)?;
        let _ = store.evaluate_available_field_exprs(program)?;
        Ok(store)
    }
}

impl<F: Field> Stage2KernelExecutor<F> for Stage2VerifierKernelExecutor<'_, F> {
    fn observe_challenge_vector(
        &mut self,
        plan: &'static Stage2TranscriptSqueezePlan,
        values: &[F],
    ) -> Result<(), Stage2KernelError> {
        self.challenge_vectors.push(Stage2ChallengeVector {
            symbol: plan.symbol,
            values: values.to_vec(),
        });
        Ok(())
    }

    fn observe_sumcheck_output(
        &mut self,
        output: &Stage2SumcheckOutput<F>,
    ) -> Result<(), Stage2KernelError> {
        self.completed_sumchecks.push(output.clone());
        Ok(())
    }

    fn prove_sumcheck<T>(
        &mut self,
        context: Stage2KernelContext<'_>,
        _transcript: &mut T,
    ) -> Result<Stage2SumcheckOutput<F>, Stage2KernelError>
    where
        T: Transcript<Challenge = F>,
    {
        Err(Stage2KernelError::WrongExecutorMode {
            driver: context.driver.symbol,
            expected: Stage2ExecutionMode::Verifier,
            actual: Stage2ExecutionMode::Prover,
        })
    }

    fn verify_sumcheck<T>(
        &mut self,
        context: Stage2KernelContext<'_>,
        transcript: &mut T,
    ) -> Result<Stage2SumcheckOutput<F>, Stage2KernelError>
    where
        T: Transcript<Challenge = F>,
    {
        let proof =
            self.proof
                .sumchecks
                .get(self.cursor)
                .ok_or(Stage2KernelError::MissingProof {
                    driver: context.driver.symbol,
                })?;
        self.cursor += 1;
        verify_stage2_kernel(
            context,
            self.value_store(context.program)?,
            proof,
            self.ram,
            transcript,
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage2KernelError {
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
        expected: Stage2ExecutionMode,
        actual: Stage2ExecutionMode,
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

impl Display for Stage2KernelError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingKernel { driver, kernel } => {
                write!(
                    formatter,
                    "stage2 driver @{driver} references missing kernel @{kernel}"
                )
            }
            Self::MissingBatch { driver, batch } => {
                write!(
                    formatter,
                    "stage2 driver @{driver} references missing batch @{batch}"
                )
            }
            Self::MissingClaim { batch, claim } => {
                write!(
                    formatter,
                    "stage2 batch @{batch} references missing claim @{claim}"
                )
            }
            Self::MissingValue { symbol } => {
                write!(formatter, "stage2 value @{symbol} is not available")
            }
            Self::PlanCountMismatch {
                artifact,
                expected,
                actual,
            } => write!(
                formatter,
                "stage2 plan @{artifact} count mismatch: expected {expected}, got {actual}"
            ),
            Self::InvalidInputLength {
                input,
                expected,
                actual,
            } => write!(
                formatter,
                "stage2 input `{input}` length mismatch: expected {expected}, got {actual}"
            ),
            Self::UnsupportedFieldExpr { symbol, formula } => write!(
                formatter,
                "stage2 field expr @{symbol} uses unsupported formula `{formula}`"
            ),
            Self::UnknownRelation { relation } => {
                write!(formatter, "stage2 relation @{relation} is not registered")
            }
            Self::UnknownKernelAbi { abi } => {
                write!(formatter, "stage2 kernel ABI `{abi}` is not registered")
            }
            Self::KernelNotImplemented { abi } => {
                write!(formatter, "stage2 kernel ABI `{abi}` is not implemented")
            }
            Self::WrongExecutorMode {
                driver,
                expected,
                actual,
            } => write!(
                formatter,
                "stage2 driver @{driver} ran with {actual:?} executor path, expected {expected:?}"
            ),
            Self::MissingProof { driver } => {
                write!(
                    formatter,
                    "stage2 verifier missing proof for driver @{driver}"
                )
            }
            Self::MissingKernelInput { kernel, input } => {
                write!(
                    formatter,
                    "stage2 kernel `{kernel}` missing input `{input}`"
                )
            }
            Self::InvalidProof { driver, reason } => {
                write!(
                    formatter,
                    "stage2 proof for driver @{driver} is invalid: {reason}"
                )
            }
        }
    }
}

impl Error for Stage2KernelError {}

fn prove_stage2_kernel<F, T>(
    context: Stage2KernelContext<'_>,
    inputs: &Stage2ProverInputs<'_, F>,
    store: Stage2ValueStore<F>,
    transcript: &mut T,
) -> Result<Stage2SumcheckOutput<F>, Stage2KernelError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    match context.abi_kind()? {
        Stage2KernelAbi::ProductVirtualUniskip => {
            prove_product_virtual_uniskip(context, inputs, store, transcript)
        }
        Stage2KernelAbi::Batched => prove_batched_stage2(context, inputs, store, transcript),
        abi => Err(Stage2KernelError::KernelNotImplemented { abi: abi.name() }),
    }
}

fn verify_stage2_kernel<F, T>(
    context: Stage2KernelContext<'_>,
    store: Stage2ValueStore<F>,
    proof: &Stage2SumcheckOutput<F>,
    ram: Option<&Stage2RamData<'_>>,
    transcript: &mut T,
) -> Result<Stage2SumcheckOutput<F>, Stage2KernelError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    match context.abi_kind()? {
        Stage2KernelAbi::ProductVirtualUniskip => {
            verify_product_virtual_uniskip(context, store, proof, transcript)
        }
        Stage2KernelAbi::Batched => verify_batched_stage2(context, store, proof, ram, transcript),
        abi => Err(Stage2KernelError::KernelNotImplemented { abi: abi.name() }),
    }
}

#[tracing::instrument(skip_all, name = "Stage2::prove_product_virtual_uniskip")]
fn prove_product_virtual_uniskip<F, T>(
    context: Stage2KernelContext<'_>,
    inputs: &Stage2ProverInputs<'_, F>,
    mut store: Stage2ValueStore<F>,
    transcript: &mut T,
) -> Result<Stage2SumcheckOutput<F>, Stage2KernelError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    let claim =
        context
            .batch_claims()?
            .into_iter()
            .next()
            .ok_or(Stage2KernelError::MissingClaim {
                batch: context.batch.symbol,
                claim: "stage2.product_virtual.uniskip.input",
            })?;
    let input_claim = store.claim_value(context.program, claim)?;
    let base_evals = product_uniskip_base_evals(&store)?;
    let extended_evals =
        inputs
            .product_uniskip_extended_evals
            .ok_or(Stage2KernelError::MissingKernelInput {
                kernel: context.kernel.abi,
                input: "product_uniskip_extended_evals",
            })?;
    let poly = build_product_uniskip_poly(
        &base_evals,
        extended_evals,
        store.scalar("stage2.product_virtual.tau_high.scalar")?,
    )?;
    if !product_uniskip_sum_matches(&poly, input_claim) {
        return Err(Stage2KernelError::InvalidProof {
            driver: context.driver.symbol,
            reason: "product uniskip input claim mismatch",
        });
    }
    append_univariate_poly(transcript, context.driver.round_label, &poly);
    let r0 = transcript.challenge();
    let eval = poly.evaluate(r0);
    append_labeled_scalar(transcript, "opening_claim", &eval);
    Ok(Stage2SumcheckOutput {
        driver: context.driver.symbol,
        point: vec![r0],
        evals: driver_evals(context, eval),
        proof: SumcheckProof {
            round_polynomials: vec![poly],
        },
    })
}

fn verify_product_virtual_uniskip<F, T>(
    context: Stage2KernelContext<'_>,
    mut store: Stage2ValueStore<F>,
    proof: &Stage2SumcheckOutput<F>,
    transcript: &mut T,
) -> Result<Stage2SumcheckOutput<F>, Stage2KernelError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    if proof.driver != context.driver.symbol {
        return Err(Stage2KernelError::InvalidProof {
            driver: context.driver.symbol,
            reason: "driver symbol mismatch",
        });
    }
    let [poly] = proof.proof.round_polynomials.as_slice() else {
        return Err(Stage2KernelError::InvalidProof {
            driver: context.driver.symbol,
            reason: "unexpected product uniskip round count",
        });
    };
    if polynomial_degree(poly) > context.driver.degree {
        return Err(Stage2KernelError::InvalidProof {
            driver: context.driver.symbol,
            reason: "product uniskip polynomial exceeds degree bound",
        });
    }
    let claim =
        context
            .batch_claims()?
            .into_iter()
            .next()
            .ok_or(Stage2KernelError::MissingClaim {
                batch: context.batch.symbol,
                claim: "stage2.product_virtual.uniskip.input",
            })?;
    let input_claim = store.claim_value(context.program, claim)?;
    if !product_uniskip_sum_matches(poly, input_claim) {
        return Err(Stage2KernelError::InvalidProof {
            driver: context.driver.symbol,
            reason: "product uniskip input claim mismatch",
        });
    }
    append_univariate_poly(transcript, context.driver.round_label, poly);
    let r0 = transcript.challenge();
    if !proof.point.is_empty() && proof.point != [r0] {
        return Err(Stage2KernelError::InvalidProof {
            driver: context.driver.symbol,
            reason: "product uniskip point mismatch",
        });
    }
    let eval = poly.evaluate(r0);
    append_labeled_scalar(transcript, "opening_claim", &eval);
    let evals = driver_evals(context, eval);
    verify_driver_evals(context.driver.symbol, &evals, &proof.evals)?;
    Ok(Stage2SumcheckOutput {
        driver: context.driver.symbol,
        point: vec![r0],
        evals,
        proof: proof.proof.clone(),
    })
}

#[tracing::instrument(skip_all, name = "Stage2::prove_batched")]
fn prove_batched_stage2<F, T>(
    context: Stage2KernelContext<'_>,
    inputs: &Stage2ProverInputs<'_, F>,
    mut store: Stage2ValueStore<F>,
    transcript: &mut T,
) -> Result<Stage2SumcheckOutput<F>, Stage2KernelError>
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
    let mut instances = Vec::with_capacity(claims.len());
    for (index, claim) in claims.iter().enumerate() {
        instances.push(Stage2BatchedInstance {
            claim,
            relation: claim_relation(context.program, claim)?,
            offset: instance_round_offset(context.program, context.driver.symbol, claim.symbol)?,
            previous_claim: input_claims[index].mul_pow_2(max_rounds - claim.num_rounds),
            state: Stage2ProverInstanceState::new(context.program, claim, inputs, &store)?,
        });
    }

    let mut point = Vec::with_capacity(max_rounds);
    let mut round_polynomials = Vec::with_capacity(max_rounds);
    let mut batched_claim = instances
        .iter()
        .zip(&batching_coeffs)
        .map(|(instance, &coefficient)| instance.previous_claim * coefficient)
        .sum::<F>();
    let two_inv = F::from_u64(2).inverse().expect("2 is non-zero");

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
                return Err(Stage2KernelError::InvalidProof {
                    driver: context.driver.symbol,
                    reason: "batched instance round claim mismatch",
                });
            }
            individual_polys.push(poly);
        }
        let batched_poly = combine_univariate_polys(&individual_polys, &batching_coeffs);
        if batched_poly.evaluate(F::zero()) + batched_poly.evaluate(F::one()) != batched_claim {
            return Err(Stage2KernelError::InvalidProof {
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
                instance.state.ingest_challenge(challenge)?;
            }
        }
        round_polynomials.push(batched_poly);
    }

    let mut evals = Vec::new();
    for instance in &instances {
        evals.extend(instance.state.final_evals(instance.relation)?);
    }
    let expected = expected_batched_output_claim(
        context,
        &store,
        &evals,
        &point,
        &batching_coeffs,
        inputs.ram,
    )?;
    if batched_claim != expected {
        return Err(Stage2KernelError::InvalidProof {
            driver: context.driver.symbol,
            reason: "batched output claim mismatch",
        });
    }
    store.observe_sumcheck_values(context.program, context.driver.symbol, &point, &evals)?;
    append_opening_claims(context.program, &mut store, transcript, &evals)?;
    Ok(Stage2SumcheckOutput {
        driver: context.driver.symbol,
        point,
        evals,
        proof: SumcheckProof { round_polynomials },
    })
}

fn verify_batched_stage2<F, T>(
    context: Stage2KernelContext<'_>,
    mut store: Stage2ValueStore<F>,
    proof: &Stage2SumcheckOutput<F>,
    ram: Option<&Stage2RamData<'_>>,
    transcript: &mut T,
) -> Result<Stage2SumcheckOutput<F>, Stage2KernelError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    if proof.driver != context.driver.symbol {
        return Err(Stage2KernelError::InvalidProof {
            driver: context.driver.symbol,
            reason: "driver symbol mismatch",
        });
    }
    if proof.proof.round_polynomials.len() != context.driver.num_rounds {
        return Err(Stage2KernelError::InvalidProof {
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
            return Err(Stage2KernelError::InvalidProof {
                driver: context.driver.symbol,
                reason: "batched polynomial exceeds degree bound",
            });
        }
        if poly.evaluate(F::zero()) + poly.evaluate(F::one()) != running_claim {
            return Err(Stage2KernelError::InvalidProof {
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
        return Err(Stage2KernelError::InvalidProof {
            driver: context.driver.symbol,
            reason: "batched point mismatch",
        });
    }
    let expected = expected_batched_output_claim(
        context,
        &store,
        &proof.evals,
        &point,
        &batching_coeffs,
        ram,
    )?;
    if running_claim != expected {
        return Err(Stage2KernelError::InvalidProof {
            driver: context.driver.symbol,
            reason: "batched output claim mismatch",
        });
    }
    store.observe_sumcheck_values(context.program, context.driver.symbol, &point, &proof.evals)?;
    append_opening_claims(context.program, &mut store, transcript, &proof.evals)?;
    Ok(Stage2SumcheckOutput {
        driver: context.driver.symbol,
        point,
        evals: proof.evals.clone(),
        proof: proof.proof.clone(),
    })
}

struct Stage2BatchedInstance<'a, F: Field> {
    claim: &'a Stage2SumcheckClaimPlan,
    relation: Stage2Relation,
    offset: usize,
    previous_claim: F,
    state: Stage2ProverInstanceState<F>,
}

impl<F: Field> Stage2BatchedInstance<'_, F> {
    fn is_active(&self, round: usize) -> bool {
        round >= self.offset && round < self.offset + self.claim.num_rounds
    }
}

enum Stage2ProverInstanceState<F: Field> {
    RamReadWrite(RamReadWriteState<F>),
    ProductVirtualRemainder(DenseInstanceState<F>),
    InstructionLookupClaimReduction(DenseInstanceState<F>),
    RamRafEvaluation(DenseInstanceState<F>),
    RamOutputCheck(DenseInstanceState<F>),
}

impl<F: Field> Stage2ProverInstanceState<F> {
    fn new(
        program: &'static Stage2CpuProgramPlan,
        claim: &Stage2SumcheckClaimPlan,
        inputs: &Stage2ProverInputs<'_, F>,
        store: &Stage2ValueStore<F>,
    ) -> Result<Self, Stage2KernelError> {
        match claim_relation(program, claim)? {
            Stage2Relation::RamReadWrite => Ok(Self::RamReadWrite(RamReadWriteState::new(
                claim, inputs, store,
            )?)),
            Stage2Relation::ProductVirtualRemainder => Ok(Self::ProductVirtualRemainder(
                product_remainder_state(claim, inputs, store)?,
            )),
            Stage2Relation::InstructionLookupClaimReduction => {
                Ok(Self::InstructionLookupClaimReduction(
                    instruction_lookup_state(claim, inputs, store)?,
                ))
            }
            Stage2Relation::RamRafEvaluation => {
                Ok(Self::RamRafEvaluation(ram_raf_state(claim, inputs, store)?))
            }
            Stage2Relation::RamOutputCheck => Ok(Self::RamOutputCheck(ram_output_state(
                claim, inputs, store,
            )?)),
            relation => Err(Stage2KernelError::KernelNotImplemented {
                abi: relation.symbol(),
            }),
        }
    }

    fn round_poly(
        &mut self,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, Stage2KernelError> {
        match self {
            Self::RamReadWrite(state) => state.round_poly(round, previous_claim),
            Self::ProductVirtualRemainder(state)
            | Self::InstructionLookupClaimReduction(state)
            | Self::RamRafEvaluation(state)
            | Self::RamOutputCheck(state) => Ok(state.round_poly(previous_claim)),
        }
    }

    fn ingest_challenge(&mut self, challenge: F) -> Result<(), Stage2KernelError> {
        match self {
            Self::RamReadWrite(state) => state.ingest_challenge(challenge),
            Self::ProductVirtualRemainder(state)
            | Self::InstructionLookupClaimReduction(state)
            | Self::RamRafEvaluation(state)
            | Self::RamOutputCheck(state) => {
                state.bind(challenge);
                Ok(())
            }
        }
    }

    fn final_evals(
        &self,
        relation: Stage2Relation,
    ) -> Result<Vec<Stage2NamedEval<F>>, Stage2KernelError> {
        match self {
            Self::RamReadWrite(state) => Ok(vec![
                named_eval(
                    "stage2.ram_read_write.eval.RamVal",
                    "RamVal",
                    state.val_eval()?,
                ),
                named_eval(
                    "stage2.ram_read_write.eval.RamRa",
                    "RamRa",
                    state.ra_eval()?,
                ),
                named_eval(
                    "stage2.ram_read_write.eval.RamInc",
                    "RamInc",
                    state.inc_eval()?,
                ),
            ]),
            Self::ProductVirtualRemainder(state)
            | Self::InstructionLookupClaimReduction(state)
            | Self::RamOutputCheck(state) => state.final_evals(relation),
            Self::RamRafEvaluation(state) => Ok(vec![named_eval(
                "stage2.ram_raf.eval.RamRa",
                "RamRa",
                state.factor_eval(0, relation)?,
            )]),
        }
    }
}

#[derive(Clone)]
struct DenseInstanceState<F: Field> {
    factors: Vec<Vec<F>>,
    factor_scratch: Vec<Vec<F>>,
    outputs: Vec<(&'static str, &'static str, Vec<F>)>,
    point: Vec<F>,
}

impl<F: Field> DenseInstanceState<F> {
    fn new(factors: Vec<Vec<F>>, outputs: Vec<(&'static str, &'static str, Vec<F>)>) -> Self {
        let factor_scratch = (0..factors.len()).map(|_| Vec::new()).collect();
        Self {
            factors,
            factor_scratch,
            outputs,
            point: Vec::new(),
        }
    }

    #[tracing::instrument(skip_all, name = "Stage2DenseState::round_poly")]
    fn round_poly(&self, _previous_claim: F) -> UnivariatePoly<F> {
        round_poly_from_factors(&self.factors, self.degree())
    }

    fn degree(&self) -> usize {
        self.factors.len()
    }

    #[tracing::instrument(skip_all, name = "Stage2DenseState::bind")]
    fn bind(&mut self, challenge: F) {
        for (factor, scratch) in self.factors.iter_mut().zip(&mut self.factor_scratch) {
            bind_dense_evals_reuse(factor, scratch, challenge);
        }
        self.point.push(challenge);
    }

    fn final_evals(
        &self,
        relation: Stage2Relation,
    ) -> Result<Vec<Stage2NamedEval<F>>, Stage2KernelError> {
        let values = dense_output_evals(&self.outputs, &self.point, relation)?;
        self.outputs
            .iter()
            .zip(values)
            .map(|((name, oracle, _), value)| Ok(named_eval(name, oracle, value)))
            .collect()
    }

    fn factor_eval(&self, index: usize, relation: Stage2Relation) -> Result<F, Stage2KernelError> {
        self.factors
            .get(index)
            .and_then(|values| values.first())
            .copied()
            .ok_or(Stage2KernelError::InvalidProof {
                driver: relation.symbol(),
                reason: "empty dense factor",
            })
    }
}

#[tracing::instrument(skip_all, name = "Stage2::product_remainder_state")]
fn product_remainder_state<F: Field>(
    _claim: &Stage2SumcheckClaimPlan,
    inputs: &Stage2ProverInputs<'_, F>,
    store: &Stage2ValueStore<F>,
) -> Result<DenseInstanceState<F>, Stage2KernelError> {
    let cycles = inputs
        .product_virtual_cycles
        .ok_or(Stage2KernelError::MissingKernelInput {
            kernel: "jolt_stage2_product_virtual_remainder",
            input: "product_virtual_cycles",
        })?;
    let tau_low = store.point("stage2.input.stage1.Product")?;
    let expected =
        1usize
            .checked_shl(tau_low.len() as u32)
            .ok_or(Stage2KernelError::InvalidInputLength {
                input: "stage2.product_virtual.cycles",
                expected: usize::BITS as usize,
                actual: tau_low.len(),
            })?;
    if cycles.len() != expected {
        return Err(Stage2KernelError::InvalidInputLength {
            input: "stage2.product_virtual.cycles",
            expected,
            actual: cycles.len(),
        });
    }
    let tau_high = store.scalar("stage2.product_virtual.tau_high.scalar")?;
    let r0 = *store
        .point("stage2.product_virtual.uniskip.sumcheck")?
        .first()
        .ok_or(Stage2KernelError::MissingValue {
            symbol: "stage2.product_virtual.uniskip.sumcheck",
        })?;
    let eq = EqPolynomial::new(tau_low.to_vec()).evaluations();
    let lagrange_tau_r0 = lagrange_kernel_eval(
        PRODUCT_VIRTUAL_UNISKIP_DOMAIN_START,
        PRODUCT_VIRTUAL_UNISKIP_DOMAIN_SIZE,
        tau_high,
        r0,
    );
    let weights = lagrange_evals(
        PRODUCT_VIRTUAL_UNISKIP_DOMAIN_START,
        PRODUCT_VIRTUAL_UNISKIP_DOMAIN_SIZE,
        r0,
    );
    let rows = cycles
        .par_iter()
        .zip(eq.par_iter())
        .map(|(cycle, &eq_value)| {
            let left_instruction = F::from_u64(cycle.instruction_left_input);
            let right_instruction = F::from_i128(cycle.instruction_right_input);
            let op_flag_jump = F::from_bool(cycle.jump_flag);
            let op_flag_write_lookup_output_to_rd =
                F::from_bool(cycle.write_lookup_output_to_rd_flag);
            let lookup_output = F::from_u64(cycle.should_branch_lookup_output);
            let instruction_flag_branch = F::from_bool(cycle.should_branch_flag);
            let next_is_noop = F::from_bool(!cycle.not_next_noop);
            let op_flag_virtual_instruction = F::from_bool(cycle.virtual_instruction_flag);
            ProductRemainderRow {
                eq_scaled: eq_value * lagrange_tau_r0,
                left: weights[0] * left_instruction
                    + weights[1] * lookup_output
                    + weights[2] * op_flag_jump,
                right: weights[0] * right_instruction
                    + weights[1] * instruction_flag_branch
                    + weights[2] * F::from_bool(cycle.not_next_noop),
                outputs: [
                    left_instruction,
                    right_instruction,
                    op_flag_jump,
                    op_flag_write_lookup_output_to_rd,
                    lookup_output,
                    instruction_flag_branch,
                    next_is_noop,
                    op_flag_virtual_instruction,
                ],
            }
        })
        .collect::<Vec<_>>();
    let mut eq_scaled = Vec::with_capacity(cycles.len());
    let mut left = Vec::with_capacity(cycles.len());
    let mut right = Vec::with_capacity(cycles.len());
    let mut outputs = product_remainder_output_buffers(cycles.len());
    for row in rows {
        eq_scaled.push(row.eq_scaled);
        left.push(row.left);
        right.push(row.right);
        for (output, value) in outputs.iter_mut().zip(row.outputs) {
            output.2.push(value);
        }
    }
    Ok(DenseInstanceState::new(
        vec![eq_scaled, left, right],
        outputs,
    ))
}

struct ProductRemainderRow<F: Field> {
    eq_scaled: F,
    left: F,
    right: F,
    outputs: [F; 8],
}

#[tracing::instrument(skip_all, name = "Stage2::instruction_lookup_state")]
fn instruction_lookup_state<F: Field>(
    _claim: &Stage2SumcheckClaimPlan,
    inputs: &Stage2ProverInputs<'_, F>,
    store: &Stage2ValueStore<F>,
) -> Result<DenseInstanceState<F>, Stage2KernelError> {
    let cycles = inputs
        .instruction_lookup_cycles
        .ok_or(Stage2KernelError::MissingKernelInput {
            kernel: "jolt_stage2_instruction_lookup_claim_reduction",
            input: "instruction_lookup_cycles",
        })?;
    let r_spartan = store.point("stage2.input.stage1.LookupOutput")?;
    let expected = 1usize.checked_shl(r_spartan.len() as u32).ok_or(
        Stage2KernelError::InvalidInputLength {
            input: "stage2.instruction_lookup.cycles",
            expected: usize::BITS as usize,
            actual: r_spartan.len(),
        },
    )?;
    if cycles.len() != expected {
        return Err(Stage2KernelError::InvalidInputLength {
            input: "stage2.instruction_lookup.cycles",
            expected,
            actual: cycles.len(),
        });
    }
    let gamma = store.scalar("stage2.instruction_lookup.gamma.scalar")?;
    let gamma_sqr = gamma.square();
    let gamma_cub = gamma_sqr * gamma;
    let gamma_quart = gamma_sqr.square();
    let eq = EqPolynomial::new(r_spartan.to_vec()).evaluations();
    let mut combined = Vec::with_capacity(cycles.len());
    let mut outputs = instruction_lookup_output_buffers(cycles.len());
    let rows = cycles
        .par_iter()
        .map(|cycle| {
            let lookup_output = F::from_u64(cycle.lookup_output);
            let left_lookup = F::from_u64(cycle.left_lookup_operand);
            let right_lookup = F::from_u128(cycle.right_lookup_operand);
            let left_instruction = F::from_u64(cycle.left_instruction_input);
            let right_instruction = F::from_i128(cycle.right_instruction_input);
            InstructionLookupRow {
                combined: lookup_output
                    + gamma * left_lookup
                    + gamma_sqr * right_lookup
                    + gamma_cub * left_instruction
                    + gamma_quart * right_instruction,
                outputs: [
                    lookup_output,
                    left_lookup,
                    right_lookup,
                    left_instruction,
                    right_instruction,
                ],
            }
        })
        .collect::<Vec<_>>();
    for row in rows {
        combined.push(row.combined);
        for (output, value) in outputs.iter_mut().zip(row.outputs) {
            output.2.push(value);
        }
    }
    Ok(DenseInstanceState::new(vec![eq, combined], outputs))
}

struct InstructionLookupRow<F: Field> {
    combined: F,
    outputs: [F; 5],
}

#[tracing::instrument(skip_all, name = "Stage2::ram_raf_state")]
fn ram_raf_state<F: Field>(
    claim: &Stage2SumcheckClaimPlan,
    inputs: &Stage2ProverInputs<'_, F>,
    store: &Stage2ValueStore<F>,
) -> Result<DenseInstanceState<F>, Stage2KernelError> {
    let ram = inputs.ram.ok_or(Stage2KernelError::MissingKernelInput {
        kernel: "jolt_stage2_ram_raf_evaluation",
        input: "ram",
    })?;
    require_operand_count("stage2.ram_raf.num_rounds", ram.log_k, claim.num_rounds)?;
    let r_cycle = store.point("stage2.input.stage1.RamAddress")?;
    let eq_cycle = EqPolynomial::new(r_cycle.to_vec()).evaluations();
    if ram.accesses.len() != eq_cycle.len() {
        return Err(Stage2KernelError::InvalidInputLength {
            input: "stage2.ram.accesses",
            expected: eq_cycle.len(),
            actual: ram.accesses.len(),
        });
    }
    let k = 1usize << ram.log_k;
    let mut ra = vec![F::zero(); k];
    for (access, weight) in ram.accesses.iter().zip(eq_cycle) {
        if let Some(address) = access.remapped_address {
            ra[address] += weight;
        }
    }
    let unmap = (0..k)
        .map(|index| F::from_u64(ram.start_address) + F::from_u64((8 * index) as u64))
        .collect::<Vec<_>>();
    Ok(DenseInstanceState::new(
        vec![ra.clone(), unmap],
        vec![("stage2.ram_raf.eval.RamRa", "RamRa", ra)],
    ))
}

#[tracing::instrument(skip_all, name = "Stage2::ram_output_state")]
fn ram_output_state<F: Field>(
    claim: &Stage2SumcheckClaimPlan,
    inputs: &Stage2ProverInputs<'_, F>,
    store: &Stage2ValueStore<F>,
) -> Result<DenseInstanceState<F>, Stage2KernelError> {
    let ram = inputs.ram.ok_or(Stage2KernelError::MissingKernelInput {
        kernel: "jolt_stage2_ram_output_check",
        input: "ram",
    })?;
    require_operand_count("stage2.ram_output.num_rounds", ram.log_k, claim.num_rounds)?;
    let layout = ram
        .output_layout
        .ok_or(Stage2KernelError::MissingKernelInput {
            kernel: "jolt_stage2_ram_output_check",
            input: "ram.output_layout",
        })?;
    let k = 1usize << ram.log_k;
    require_operand_count("stage2.ram.final_ram", k, ram.final_ram.len())?;
    let r_address = store.point("stage2.ram_output.r_address")?;
    let eq = EqPolynomial::new(r_address.to_vec()).evaluations();
    let mut io_mask = vec![F::zero(); k];
    let mut diff = vec![F::zero(); k];
    let mut val_final = Vec::with_capacity(k);
    for index in 0..k {
        let final_value = F::from_u64(ram.final_ram[index]);
        val_final.push(final_value);
        if index >= layout.io_start && index < layout.io_end {
            io_mask[index] = F::one();
            diff[index] = F::zero();
        } else {
            diff[index] = final_value;
        }
    }
    Ok(DenseInstanceState::new(
        vec![eq, io_mask, diff],
        vec![(
            "stage2.ram_output.eval.RamValFinal",
            "RamValFinal",
            val_final,
        )],
    ))
}

#[derive(Clone, Debug)]
struct RamCycleEntry<F: Field> {
    row: usize,
    col: usize,
    prev_val: u64,
    next_val: u64,
    val_coeff: F,
    ra_coeff: F,
}

#[derive(Clone, Debug)]
struct RamAddressEntry<F: Field> {
    row: usize,
    col: usize,
    prev_val: F,
    next_val: F,
    val_coeff: F,
    ra_coeff: F,
}

#[derive(Clone)]
struct RamReadWriteState<F: Field> {
    gamma: F,
    log_t: usize,
    round: usize,
    cycle_entries: Vec<RamCycleEntry<F>>,
    address_entries: Vec<RamAddressEntry<F>>,
    cycle_scratch: Vec<RamCycleEntry<F>>,
    address_scratch: Vec<RamAddressEntry<F>>,
    inc: Vec<F>,
    eq: Vec<F>,
    val_init: Vec<F>,
}

impl<F: Field> RamReadWriteState<F> {
    #[tracing::instrument(skip_all, name = "RamReadWriteState::new")]
    fn new(
        _claim: &Stage2SumcheckClaimPlan,
        inputs: &Stage2ProverInputs<'_, F>,
        store: &Stage2ValueStore<F>,
    ) -> Result<Self, Stage2KernelError> {
        let ram = inputs.ram.ok_or(Stage2KernelError::MissingKernelInput {
            kernel: "jolt_stage2_ram_read_write",
            input: "ram",
        })?;
        let r_cycle = store.point("stage2.input.stage1.RamReadValue")?;
        let log_t = r_cycle.len();
        let t = 1usize << log_t;
        let k = 1usize << ram.log_k;
        require_operand_count("stage2.ram.accesses", t, ram.accesses.len())?;
        require_operand_count("stage2.ram.initial_ram", k, ram.initial_ram.len())?;
        let gamma = store.scalar("stage2.ram_read_write.gamma.scalar")?;
        let mut cycle_entries = Vec::new();
        let mut inc = Vec::with_capacity(t);
        for (row, access) in ram.accesses.iter().enumerate() {
            inc.push(F::from_u64(access.write_value) - F::from_u64(access.read_value));
            if let Some(col) = access.remapped_address {
                cycle_entries.push(RamCycleEntry {
                    row,
                    col,
                    prev_val: access.read_value,
                    next_val: access.write_value,
                    val_coeff: F::from_u64(access.read_value),
                    ra_coeff: F::one(),
                });
            }
        }
        cycle_entries.sort_by_key(|entry| (entry.row, entry.col));
        Ok(Self {
            gamma,
            log_t,
            round: 0,
            cycle_entries,
            address_entries: Vec::new(),
            cycle_scratch: Vec::new(),
            address_scratch: Vec::new(),
            inc,
            eq: EqPolynomial::new(r_cycle.to_vec()).evaluations(),
            val_init: ram
                .initial_ram
                .iter()
                .map(|&value| F::from_u64(value))
                .collect(),
        })
    }

    fn round_poly(
        &mut self,
        _round: usize,
        _previous_claim: F,
    ) -> Result<UnivariatePoly<F>, Stage2KernelError> {
        if self.round < self.log_t {
            Ok(self.cycle_round_poly())
        } else {
            Ok(self.address_round_poly())
        }
    }

    fn ingest_challenge(&mut self, challenge: F) -> Result<(), Stage2KernelError> {
        if self.round < self.log_t {
            self.bind_cycle(challenge);
            if self.round + 1 == self.log_t {
                self.address_entries = self
                    .cycle_entries
                    .drain(..)
                    .map(|entry| RamAddressEntry {
                        row: entry.row,
                        col: entry.col,
                        prev_val: F::from_u64(entry.prev_val),
                        next_val: F::from_u64(entry.next_val),
                        val_coeff: entry.val_coeff,
                        ra_coeff: entry.ra_coeff,
                    })
                    .collect();
                self.address_entries
                    .sort_by_key(|entry| (entry.col, entry.row));
            }
        } else {
            self.bind_address(challenge);
        }
        self.round += 1;
        Ok(())
    }

    #[tracing::instrument(skip_all, name = "RamReadWriteState::cycle_round_poly")]
    fn cycle_round_poly(&self) -> UnivariatePoly<F> {
        let ranges = grouped_cycle_pair_ranges(&self.cycle_entries);
        let xs = [F::zero(), F::one(), F::from_u64(2), F::from_u64(3)];
        let accumulators = if ranges.len() >= DENSE_BIND_PAR_THRESHOLD {
            ranges
                .par_iter()
                .fold(
                    || [F::Accumulator::default(); 4],
                    |mut local, &(start, end)| {
                        accumulate_cycle_pair_range(
                            &mut local,
                            &self.cycle_entries[start..end],
                            &self.inc,
                            &self.eq,
                            self.gamma,
                            xs,
                        );
                        local
                    },
                )
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
            ranges.iter().fold(
                [F::Accumulator::default(); 4],
                |mut local, &(start, end)| {
                    accumulate_cycle_pair_range(
                        &mut local,
                        &self.cycle_entries[start..end],
                        &self.inc,
                        &self.eq,
                        self.gamma,
                        xs,
                    );
                    local
                },
            )
        };
        let evals = accumulators
            .into_iter()
            .map(FieldAccumulator::reduce)
            .collect::<Vec<_>>();
        UnivariatePoly::from_evals(&evals)
    }

    #[tracing::instrument(skip_all, name = "RamReadWriteState::address_round_poly")]
    fn address_round_poly(&self) -> UnivariatePoly<F> {
        let mut evals = vec![F::zero(); 4];
        let mut cursor = 0;
        while cursor < self.address_entries.len() {
            let pair = self.address_entries[cursor].col / 2;
            let start = cursor;
            while cursor < self.address_entries.len()
                && self.address_entries[cursor].col / 2 == pair
            {
                cursor += 1;
            }
            let entries = &self.address_entries[start..cursor];
            let odd_start = entries.partition_point(|entry| entry.col % 2 == 0);
            let even = &entries[..odd_start];
            let odd = &entries[odd_start..];
            let even_checkpoint = self.val_init[2 * pair];
            let odd_checkpoint = self.val_init[2 * pair + 1];
            for (x, eval) in evals.iter_mut().enumerate() {
                *eval += address_pair_eval(
                    even,
                    odd,
                    even_checkpoint,
                    odd_checkpoint,
                    RamEvalWeights {
                        inc: self.inc[0],
                        eq: self.eq[0],
                        gamma: self.gamma,
                        x: F::from_u64(x as u64),
                    },
                );
            }
        }
        UnivariatePoly::from_evals(&evals)
    }

    #[tracing::instrument(skip_all, name = "RamReadWriteState::bind_cycle")]
    fn bind_cycle(&mut self, challenge: F) {
        let mut bound = std::mem::take(&mut self.cycle_scratch);
        bound.clear();
        bound.reserve(self.cycle_entries.len());
        let mut cursor = 0;
        while cursor < self.cycle_entries.len() {
            let pair = self.cycle_entries[cursor].row / 2;
            let start = cursor;
            while cursor < self.cycle_entries.len() && self.cycle_entries[cursor].row / 2 == pair {
                cursor += 1;
            }
            let entries = &self.cycle_entries[start..cursor];
            let odd_start = entries.partition_point(|entry| entry.row % 2 == 0);
            bind_cycle_rows(
                &entries[..odd_start],
                &entries[odd_start..],
                challenge,
                &mut bound,
            );
        }
        std::mem::swap(&mut self.cycle_entries, &mut bound);
        self.cycle_scratch = bound;
        bind_low_to_high(&mut self.inc, challenge);
        bind_low_to_high(&mut self.eq, challenge);
    }

    #[tracing::instrument(skip_all, name = "RamReadWriteState::bind_address")]
    fn bind_address(&mut self, challenge: F) {
        let mut bound = std::mem::take(&mut self.address_scratch);
        bound.clear();
        bound.reserve(self.address_entries.len());
        let mut cursor = 0;
        while cursor < self.address_entries.len() {
            let pair = self.address_entries[cursor].col / 2;
            let start = cursor;
            while cursor < self.address_entries.len()
                && self.address_entries[cursor].col / 2 == pair
            {
                cursor += 1;
            }
            let entries = &self.address_entries[start..cursor];
            let odd_start = entries.partition_point(|entry| entry.col % 2 == 0);
            bind_address_cols(
                &entries[..odd_start],
                &entries[odd_start..],
                self.val_init[2 * pair],
                self.val_init[2 * pair + 1],
                challenge,
                &mut bound,
            );
        }
        std::mem::swap(&mut self.address_entries, &mut bound);
        self.address_scratch = bound;
        bind_low_to_high(&mut self.val_init, challenge);
    }

    fn ra_eval(&self) -> Result<F, Stage2KernelError> {
        Ok(self
            .address_entries
            .first()
            .filter(|entry| entry.col == 0 && entry.row == 0)
            .map_or(F::zero(), |entry| entry.ra_coeff))
    }

    fn val_eval(&self) -> Result<F, Stage2KernelError> {
        Ok(self
            .address_entries
            .first()
            .filter(|entry| entry.col == 0 && entry.row == 0)
            .map_or(self.val_init[0], |entry| entry.val_coeff))
    }

    fn inc_eval(&self) -> Result<F, Stage2KernelError> {
        Ok(self.inc[0])
    }
}

fn grouped_cycle_pair_ranges<F: Field>(entries: &[RamCycleEntry<F>]) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut cursor = 0;
    while cursor < entries.len() {
        let pair = entries[cursor].row / 2;
        let start = cursor;
        while cursor < entries.len() && entries[cursor].row / 2 == pair {
            cursor += 1;
        }
        ranges.push((start, cursor));
    }
    ranges
}

fn accumulate_cycle_pair_range<F: Field>(
    accumulators: &mut [F::Accumulator; 4],
    entries: &[RamCycleEntry<F>],
    inc: &[F],
    eq: &[F],
    gamma: F,
    xs: [F; 4],
) {
    let pair = entries[0].row / 2;
    let odd_start = entries.partition_point(|entry| entry.row % 2 == 0);
    let even = &entries[..odd_start];
    let odd = &entries[odd_start..];
    let row = pair * 2;
    let inc_pair = [inc[row], inc[row + 1]];
    let eq_pair = [eq[row], eq[row + 1]];
    for (x, accumulator) in xs.into_iter().zip(accumulators) {
        accumulator.acc_add(cycle_pair_eval(even, odd, inc_pair, eq_pair, gamma, x));
    }
}

fn cycle_pair_eval<F: Field>(
    even: &[RamCycleEntry<F>],
    odd: &[RamCycleEntry<F>],
    inc_pair: [F; 2],
    eq_pair: [F; 2],
    gamma: F,
    x: F,
) -> F {
    let inc = line(inc_pair[0], inc_pair[1], x);
    let eq = line(eq_pair[0], eq_pair[1], x);
    let mut total = F::zero();
    let mut i = 0;
    let mut j = 0;
    while i < even.len() && j < odd.len() {
        match even[i].col.cmp(&odd[j].col) {
            Ordering::Equal => {
                total += cycle_entry_eval(Some(&even[i]), Some(&odd[j]), inc, eq, gamma, x);
                i += 1;
                j += 1;
            }
            Ordering::Less => {
                total += cycle_entry_eval(Some(&even[i]), None, inc, eq, gamma, x);
                i += 1;
            }
            Ordering::Greater => {
                total += cycle_entry_eval(None, Some(&odd[j]), inc, eq, gamma, x);
                j += 1;
            }
        }
    }
    for entry in &even[i..] {
        total += cycle_entry_eval(Some(entry), None, inc, eq, gamma, x);
    }
    for entry in &odd[j..] {
        total += cycle_entry_eval(None, Some(entry), inc, eq, gamma, x);
    }
    total
}

fn cycle_entry_eval<F: Field>(
    even: Option<&RamCycleEntry<F>>,
    odd: Option<&RamCycleEntry<F>>,
    inc: F,
    eq: F,
    gamma: F,
    x: F,
) -> F {
    let (ra0, ra1, val0, val1) = match (even, odd) {
        (Some(even), Some(odd)) => (even.ra_coeff, odd.ra_coeff, even.val_coeff, odd.val_coeff),
        (Some(even), None) => (
            even.ra_coeff,
            F::zero(),
            even.val_coeff,
            F::from_u64(even.next_val),
        ),
        (None, Some(odd)) => (
            F::zero(),
            odd.ra_coeff,
            F::from_u64(odd.prev_val),
            odd.val_coeff,
        ),
        (None, None) => unreachable!(),
    };
    let ra = line(ra0, ra1, x);
    let val = line(val0, val1, x);
    eq * ra * (val + gamma * (val + inc))
}

fn address_pair_eval<F: Field>(
    even: &[RamAddressEntry<F>],
    odd: &[RamAddressEntry<F>],
    mut even_checkpoint: F,
    mut odd_checkpoint: F,
    weights: RamEvalWeights<F>,
) -> F {
    let mut total = F::zero();
    let mut i = 0;
    let mut j = 0;
    while i < even.len() && j < odd.len() {
        match even[i].row.cmp(&odd[j].row) {
            Ordering::Equal => {
                total += address_entry_eval(
                    Some(&even[i]),
                    Some(&odd[j]),
                    even_checkpoint,
                    odd_checkpoint,
                    weights,
                );
                even_checkpoint = even[i].next_val;
                odd_checkpoint = odd[j].next_val;
                i += 1;
                j += 1;
            }
            Ordering::Less => {
                total += address_entry_eval(
                    Some(&even[i]),
                    None,
                    even_checkpoint,
                    odd_checkpoint,
                    weights,
                );
                even_checkpoint = even[i].next_val;
                i += 1;
            }
            Ordering::Greater => {
                total += address_entry_eval(
                    None,
                    Some(&odd[j]),
                    even_checkpoint,
                    odd_checkpoint,
                    weights,
                );
                odd_checkpoint = odd[j].next_val;
                j += 1;
            }
        }
    }
    for entry in &even[i..] {
        total += address_entry_eval(Some(entry), None, even_checkpoint, odd_checkpoint, weights);
        even_checkpoint = entry.next_val;
    }
    for entry in &odd[j..] {
        total += address_entry_eval(None, Some(entry), even_checkpoint, odd_checkpoint, weights);
        odd_checkpoint = entry.next_val;
    }
    total
}

#[derive(Clone, Copy)]
struct RamEvalWeights<F: Field> {
    inc: F,
    eq: F,
    gamma: F,
    x: F,
}

fn address_entry_eval<F: Field>(
    even: Option<&RamAddressEntry<F>>,
    odd: Option<&RamAddressEntry<F>>,
    even_checkpoint: F,
    odd_checkpoint: F,
    weights: RamEvalWeights<F>,
) -> F {
    let (ra0, ra1, val0, val1) = match (even, odd) {
        (Some(even), Some(odd)) => (even.ra_coeff, odd.ra_coeff, even.val_coeff, odd.val_coeff),
        (Some(even), None) => (even.ra_coeff, F::zero(), even.val_coeff, odd_checkpoint),
        (None, Some(odd)) => (F::zero(), odd.ra_coeff, even_checkpoint, odd.val_coeff),
        (None, None) => unreachable!(),
    };
    let ra = line(ra0, ra1, weights.x);
    let val = line(val0, val1, weights.x);
    weights.eq * ra * (val + weights.gamma * (val + weights.inc))
}

fn bind_cycle_rows<F: Field>(
    even: &[RamCycleEntry<F>],
    odd: &[RamCycleEntry<F>],
    challenge: F,
    out: &mut Vec<RamCycleEntry<F>>,
) {
    let mut i = 0;
    let mut j = 0;
    while i < even.len() && j < odd.len() {
        match even[i].col.cmp(&odd[j].col) {
            Ordering::Equal => {
                out.push(bind_cycle_entry(Some(&even[i]), Some(&odd[j]), challenge));
                i += 1;
                j += 1;
            }
            Ordering::Less => {
                out.push(bind_cycle_entry(Some(&even[i]), None, challenge));
                i += 1;
            }
            Ordering::Greater => {
                out.push(bind_cycle_entry(None, Some(&odd[j]), challenge));
                j += 1;
            }
        }
    }
    for entry in &even[i..] {
        out.push(bind_cycle_entry(Some(entry), None, challenge));
    }
    for entry in &odd[j..] {
        out.push(bind_cycle_entry(None, Some(entry), challenge));
    }
}

fn bind_cycle_entry<F: Field>(
    even: Option<&RamCycleEntry<F>>,
    odd: Option<&RamCycleEntry<F>>,
    r: F,
) -> RamCycleEntry<F> {
    match (even, odd) {
        (Some(even), Some(odd)) => RamCycleEntry {
            row: even.row / 2,
            col: even.col,
            ra_coeff: line(even.ra_coeff, odd.ra_coeff, r),
            val_coeff: line(even.val_coeff, odd.val_coeff, r),
            prev_val: even.prev_val,
            next_val: odd.next_val,
        },
        (Some(even), None) => RamCycleEntry {
            row: even.row / 2,
            col: even.col,
            ra_coeff: line(even.ra_coeff, F::zero(), r),
            val_coeff: line(even.val_coeff, F::from_u64(even.next_val), r),
            prev_val: even.prev_val,
            next_val: even.next_val,
        },
        (None, Some(odd)) => RamCycleEntry {
            row: odd.row / 2,
            col: odd.col,
            ra_coeff: line(F::zero(), odd.ra_coeff, r),
            val_coeff: line(F::from_u64(odd.prev_val), odd.val_coeff, r),
            prev_val: odd.prev_val,
            next_val: odd.next_val,
        },
        (None, None) => unreachable!(),
    }
}

fn bind_address_cols<F: Field>(
    even: &[RamAddressEntry<F>],
    odd: &[RamAddressEntry<F>],
    mut even_checkpoint: F,
    mut odd_checkpoint: F,
    challenge: F,
    out: &mut Vec<RamAddressEntry<F>>,
) {
    let mut i = 0;
    let mut j = 0;
    while i < even.len() && j < odd.len() {
        match even[i].row.cmp(&odd[j].row) {
            Ordering::Equal => {
                out.push(bind_address_entry(
                    Some(&even[i]),
                    Some(&odd[j]),
                    even_checkpoint,
                    odd_checkpoint,
                    challenge,
                ));
                even_checkpoint = even[i].next_val;
                odd_checkpoint = odd[j].next_val;
                i += 1;
                j += 1;
            }
            Ordering::Less => {
                out.push(bind_address_entry(
                    Some(&even[i]),
                    None,
                    even_checkpoint,
                    odd_checkpoint,
                    challenge,
                ));
                even_checkpoint = even[i].next_val;
                i += 1;
            }
            Ordering::Greater => {
                out.push(bind_address_entry(
                    None,
                    Some(&odd[j]),
                    even_checkpoint,
                    odd_checkpoint,
                    challenge,
                ));
                odd_checkpoint = odd[j].next_val;
                j += 1;
            }
        }
    }
    for entry in &even[i..] {
        out.push(bind_address_entry(
            Some(entry),
            None,
            even_checkpoint,
            odd_checkpoint,
            challenge,
        ));
        even_checkpoint = entry.next_val;
    }
    for entry in &odd[j..] {
        out.push(bind_address_entry(
            None,
            Some(entry),
            even_checkpoint,
            odd_checkpoint,
            challenge,
        ));
        odd_checkpoint = entry.next_val;
    }
}

fn bind_address_entry<F: Field>(
    even: Option<&RamAddressEntry<F>>,
    odd: Option<&RamAddressEntry<F>>,
    even_checkpoint: F,
    odd_checkpoint: F,
    r: F,
) -> RamAddressEntry<F> {
    match (even, odd) {
        (Some(even), Some(odd)) => RamAddressEntry {
            row: even.row,
            col: even.col / 2,
            ra_coeff: line(even.ra_coeff, odd.ra_coeff, r),
            val_coeff: line(even.val_coeff, odd.val_coeff, r),
            prev_val: line(even.prev_val, odd.prev_val, r),
            next_val: line(even.next_val, odd.next_val, r),
        },
        (Some(even), None) => RamAddressEntry {
            row: even.row,
            col: even.col / 2,
            ra_coeff: line(even.ra_coeff, F::zero(), r),
            val_coeff: line(even.val_coeff, odd_checkpoint, r),
            prev_val: line(even.prev_val, odd_checkpoint, r),
            next_val: line(even.next_val, odd_checkpoint, r),
        },
        (None, Some(odd)) => RamAddressEntry {
            row: odd.row,
            col: odd.col / 2,
            ra_coeff: line(F::zero(), odd.ra_coeff, r),
            val_coeff: line(even_checkpoint, odd.val_coeff, r),
            prev_val: line(even_checkpoint, odd.prev_val, r),
            next_val: line(even_checkpoint, odd.next_val, r),
        },
        (None, None) => unreachable!(),
    }
}

fn expected_batched_output_claim<F: Field>(
    context: Stage2KernelContext<'_>,
    store: &Stage2ValueStore<F>,
    evals: &[Stage2NamedEval<F>],
    point: &[F],
    batching_coeffs: &[F],
    ram: Option<&Stage2RamData<'_>>,
) -> Result<F, Stage2KernelError> {
    let mut expected = F::zero();
    for (claim, &coefficient) in context.batch_claims()?.iter().zip(batching_coeffs) {
        let instance = context
            .program
            .instance_results
            .iter()
            .find(|instance| {
                instance.claim == claim.symbol && instance.source == context.driver.symbol
            })
            .ok_or(Stage2KernelError::MissingClaim {
                batch: context.batch.symbol,
                claim: claim.symbol,
            })?;
        let local_point = point
            .get(instance.round_offset..instance.round_offset + instance.num_rounds)
            .ok_or(Stage2KernelError::InvalidInputLength {
                input: instance.symbol,
                expected: instance.round_offset + instance.num_rounds,
                actual: point.len(),
            })?;
        let claim_value = match Stage2Relation::from_symbol(instance.relation).ok_or(
            Stage2KernelError::UnknownRelation {
                relation: instance.relation,
            },
        )? {
            Stage2Relation::RamReadWrite => expected_ram_read_write(store, evals, local_point)?,
            Stage2Relation::ProductVirtualRemainder => {
                expected_product_remainder(store, evals, local_point)?
            }
            Stage2Relation::InstructionLookupClaimReduction => {
                expected_instruction_lookup(store, evals, local_point)?
            }
            Stage2Relation::RamRafEvaluation => expected_ram_raf(store, evals, local_point, ram)?,
            Stage2Relation::RamOutputCheck => expected_ram_output(store, evals, local_point, ram)?,
            relation => {
                return Err(Stage2KernelError::KernelNotImplemented {
                    abi: relation.symbol(),
                })
            }
        };
        expected += coefficient * claim_value;
    }
    Ok(expected)
}

fn expected_ram_read_write<F: Field>(
    store: &Stage2ValueStore<F>,
    evals: &[Stage2NamedEval<F>],
    local_point: &[F],
) -> Result<F, Stage2KernelError> {
    let r_cycle_stage1 = store.point("stage2.input.stage1.RamReadValue")?;
    let log_t = r_cycle_stage1.len();
    let r_cycle = reverse_slice(&local_point[..log_t]);
    let eq_eval = EqPolynomial::<F>::mle(r_cycle_stage1, &r_cycle);
    let gamma = store.scalar("stage2.ram_read_write.gamma.scalar")?;
    let val = eval_by_name(evals, "stage2.ram_read_write.eval.RamVal")?;
    let ra = eval_by_name(evals, "stage2.ram_read_write.eval.RamRa")?;
    let inc = eval_by_name(evals, "stage2.ram_read_write.eval.RamInc")?;
    Ok(eq_eval * ra * (val + gamma * (val + inc)))
}

fn expected_product_remainder<F: Field>(
    store: &Stage2ValueStore<F>,
    evals: &[Stage2NamedEval<F>],
    local_point: &[F],
) -> Result<F, Stage2KernelError> {
    let tau_low = store.point("stage2.input.stage1.Product")?;
    let tau_high = store.scalar("stage2.product_virtual.tau_high.scalar")?;
    let r0 = *store
        .point("stage2.product_virtual.uniskip.sumcheck")?
        .first()
        .ok_or(Stage2KernelError::MissingValue {
            symbol: "stage2.product_virtual.uniskip.sumcheck",
        })?;
    let r_tail = reverse_slice(local_point);
    let low = EqPolynomial::<F>::mle(tau_low, &r_tail);
    let high = lagrange_kernel_eval(
        PRODUCT_VIRTUAL_UNISKIP_DOMAIN_START,
        PRODUCT_VIRTUAL_UNISKIP_DOMAIN_SIZE,
        tau_high,
        r0,
    );
    let weights = lagrange_evals(
        PRODUCT_VIRTUAL_UNISKIP_DOMAIN_START,
        PRODUCT_VIRTUAL_UNISKIP_DOMAIN_SIZE,
        r0,
    );
    let left = weights[0]
        * eval_by_name(
            evals,
            "stage2.product_virtual.remainder.eval.LeftInstructionInput",
        )?
        + weights[1] * eval_by_name(evals, "stage2.product_virtual.remainder.eval.LookupOutput")?
        + weights[2] * eval_by_name(evals, "stage2.product_virtual.remainder.eval.OpFlagJump")?;
    let right = weights[0]
        * eval_by_name(
            evals,
            "stage2.product_virtual.remainder.eval.RightInstructionInput",
        )?
        + weights[1]
            * eval_by_name(
                evals,
                "stage2.product_virtual.remainder.eval.InstructionFlagBranch",
            )?
        + weights[2]
            * (F::one() - eval_by_name(evals, "stage2.product_virtual.remainder.eval.NextIsNoop")?);
    Ok(high * low * left * right)
}

fn expected_instruction_lookup<F: Field>(
    store: &Stage2ValueStore<F>,
    evals: &[Stage2NamedEval<F>],
    local_point: &[F],
) -> Result<F, Stage2KernelError> {
    let opening_point = reverse_slice(local_point);
    let r_spartan = store.point("stage2.input.stage1.LookupOutput")?;
    let eq_eval = EqPolynomial::<F>::mle(&opening_point, r_spartan);
    let gamma = store.scalar("stage2.instruction_lookup.gamma.scalar")?;
    let gamma_sqr = gamma.square();
    let gamma_cub = gamma_sqr * gamma;
    let gamma_quart = gamma_sqr.square();
    let weighted = eval_by_name(
        evals,
        "stage2.instruction_lookup.claim_reduction.eval.LookupOutput",
    )? + gamma
        * eval_by_name(
            evals,
            "stage2.instruction_lookup.claim_reduction.eval.LeftLookupOperand",
        )?
        + gamma_sqr
            * eval_by_name(
                evals,
                "stage2.instruction_lookup.claim_reduction.eval.RightLookupOperand",
            )?
        + gamma_cub
            * eval_by_name(
                evals,
                "stage2.instruction_lookup.claim_reduction.eval.LeftInstructionInput",
            )?
        + gamma_quart
            * eval_by_name(
                evals,
                "stage2.instruction_lookup.claim_reduction.eval.RightInstructionInput",
            )?;
    Ok(eq_eval * weighted)
}

fn expected_ram_raf<F: Field>(
    _store: &Stage2ValueStore<F>,
    evals: &[Stage2NamedEval<F>],
    local_point: &[F],
    ram: Option<&Stage2RamData<'_>>,
) -> Result<F, Stage2KernelError> {
    let ram = ram.ok_or(Stage2KernelError::MissingKernelInput {
        kernel: "jolt_stage2_ram_raf_evaluation",
        input: "ram",
    })?;
    let address = reverse_slice(local_point);
    let unmap = unmap_eval(ram.log_k, ram.start_address, &address);
    Ok(unmap * eval_by_name(evals, "stage2.ram_raf.eval.RamRa")?)
}

fn expected_ram_output<F: Field>(
    store: &Stage2ValueStore<F>,
    evals: &[Stage2NamedEval<F>],
    local_point: &[F],
    ram: Option<&Stage2RamData<'_>>,
) -> Result<F, Stage2KernelError> {
    let ram = ram.ok_or(Stage2KernelError::MissingKernelInput {
        kernel: "jolt_stage2_ram_output_check",
        input: "ram",
    })?;
    let layout = ram
        .output_layout
        .ok_or(Stage2KernelError::MissingKernelInput {
            kernel: "jolt_stage2_ram_output_check",
            input: "ram.output_layout",
        })?;
    let r_address = store.point("stage2.ram_output.r_address")?;
    let opening_point = reverse_slice(local_point);
    let eq_eval = EqPolynomial::<F>::mle(r_address, &opening_point);
    let io_mask = range_mask_eval(layout.io_start, layout.io_end, &opening_point);
    let val_io = sparse_final_ram_eval(
        ram.final_ram,
        layout.io_start,
        layout.io_end,
        &opening_point,
    );
    let val_final = eval_by_name(evals, "stage2.ram_output.eval.RamValFinal")?;
    Ok(eq_eval * io_mask * (val_final - val_io))
}

fn eval_by_name<F: Field>(
    evals: &[Stage2NamedEval<F>],
    name: &'static str,
) -> Result<F, Stage2KernelError> {
    evals
        .iter()
        .find(|eval| eval.name == name)
        .map(|eval| eval.value)
        .ok_or(Stage2KernelError::MissingValue { symbol: name })
}

fn reverse_slice<F: Field>(values: &[F]) -> Vec<F> {
    values.iter().rev().copied().collect()
}

fn unmap_eval<F: Field>(log_k: usize, start_address: u64, point: &[F]) -> F {
    point
        .iter()
        .enumerate()
        .fold(F::from_u64(start_address), |acc, (index, &value)| {
            acc + value.mul_pow_2(log_k - 1 - index).mul_u64(8)
        })
}

fn range_mask_eval<F: Field>(start: usize, end: usize, point: &[F]) -> F {
    (start..end)
        .map(|index| eq_eval_at_index(index, point))
        .sum()
}

fn sparse_final_ram_eval<F: Field>(values: &[u64], start: usize, end: usize, point: &[F]) -> F {
    (start..end)
        .map(|index| F::from_u64(values[index]) * eq_eval_at_index(index, point))
        .sum()
}

fn eq_eval_at_index<F: Field>(index: usize, point: &[F]) -> F {
    point.iter().enumerate().fold(F::one(), |acc, (bit, &r)| {
        let mask = 1usize << (point.len() - 1 - bit);
        if index & mask == 0 {
            acc * (F::one() - r)
        } else {
            acc * r
        }
    })
}

fn product_remainder_output_buffers<F: Field>(
    capacity: usize,
) -> Vec<(&'static str, &'static str, Vec<F>)> {
    [
        (
            "stage2.product_virtual.remainder.eval.LeftInstructionInput",
            "LeftInstructionInput",
        ),
        (
            "stage2.product_virtual.remainder.eval.RightInstructionInput",
            "RightInstructionInput",
        ),
        (
            "stage2.product_virtual.remainder.eval.OpFlagJump",
            "OpFlagJump",
        ),
        (
            "stage2.product_virtual.remainder.eval.OpFlagWriteLookupOutputToRD",
            "OpFlagWriteLookupOutputToRD",
        ),
        (
            "stage2.product_virtual.remainder.eval.LookupOutput",
            "LookupOutput",
        ),
        (
            "stage2.product_virtual.remainder.eval.InstructionFlagBranch",
            "InstructionFlagBranch",
        ),
        (
            "stage2.product_virtual.remainder.eval.NextIsNoop",
            "NextIsNoop",
        ),
        (
            "stage2.product_virtual.remainder.eval.OpFlagVirtualInstruction",
            "OpFlagVirtualInstruction",
        ),
    ]
    .into_iter()
    .map(|(name, oracle)| (name, oracle, Vec::with_capacity(capacity)))
    .collect()
}

fn instruction_lookup_output_buffers<F: Field>(
    capacity: usize,
) -> Vec<(&'static str, &'static str, Vec<F>)> {
    [
        (
            "stage2.instruction_lookup.claim_reduction.eval.LookupOutput",
            "LookupOutput",
        ),
        (
            "stage2.instruction_lookup.claim_reduction.eval.LeftLookupOperand",
            "LeftLookupOperand",
        ),
        (
            "stage2.instruction_lookup.claim_reduction.eval.RightLookupOperand",
            "RightLookupOperand",
        ),
        (
            "stage2.instruction_lookup.claim_reduction.eval.LeftInstructionInput",
            "LeftInstructionInput",
        ),
        (
            "stage2.instruction_lookup.claim_reduction.eval.RightInstructionInput",
            "RightInstructionInput",
        ),
    ]
    .into_iter()
    .map(|(name, oracle)| (name, oracle, Vec::with_capacity(capacity)))
    .collect()
}

fn named_eval<F: Field>(name: &'static str, oracle: &'static str, value: F) -> Stage2NamedEval<F> {
    Stage2NamedEval {
        name,
        oracle,
        value,
    }
}

fn claim_relation(
    program: &'static Stage2CpuProgramPlan,
    claim: &Stage2SumcheckClaimPlan,
) -> Result<Stage2Relation, Stage2KernelError> {
    let kernel = find_kernel(program, claim.kernel).ok_or(Stage2KernelError::MissingKernel {
        driver: claim.symbol,
        kernel: claim.kernel,
    })?;
    kernel.relation_kind()
}

fn instance_round_offset(
    program: &'static Stage2CpuProgramPlan,
    driver: &'static str,
    claim: &'static str,
) -> Result<usize, Stage2KernelError> {
    program
        .instance_results
        .iter()
        .find(|instance| instance.source == driver && instance.claim == claim)
        .map(|instance| instance.round_offset)
        .ok_or(Stage2KernelError::MissingClaim {
            batch: driver,
            claim,
        })
}

fn combine_univariate_polys<F: Field>(
    polynomials: &[UnivariatePoly<F>],
    coefficients: &[F],
) -> UnivariatePoly<F> {
    polynomials.iter().zip(coefficients).fold(
        UnivariatePoly::zero(),
        |mut combined, (poly, &coefficient)| {
            combined += &(poly * coefficient);
            combined
        },
    )
}

fn round_poly_from_factors<F: Field>(factors: &[Vec<F>], degree: usize) -> UnivariatePoly<F> {
    if factors.is_empty() {
        return UnivariatePoly::zero();
    }
    let half = factors[0].len() / 2;
    let accumulators = if half >= DENSE_BIND_PAR_THRESHOLD {
        (0..half)
            .into_par_iter()
            .map(|row| dense_product_coefficients(factors, row))
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
            let row_coeffs = dense_product_coefficients(factors, row);
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

fn dense_product_coefficients<F: Field>(factors: &[Vec<F>], row: usize) -> [F::Accumulator; 4] {
    if factors.len() == 2 {
        let left0 = factors[0][2 * row];
        let left_delta = factors[0][2 * row + 1] - left0;
        let right0 = factors[1][2 * row];
        let right_delta = factors[1][2 * row + 1] - right0;
        let mut accumulators = [F::Accumulator::default(); 4];
        accumulators[0].fmadd(left0, right0);
        accumulators[1].fmadd(left_delta, right0);
        accumulators[1].fmadd(left0, right_delta);
        accumulators[2].fmadd(left_delta, right_delta);
        return accumulators;
    }
    if factors.len() == 3 {
        let first0 = factors[0][2 * row];
        let first_delta = factors[0][2 * row + 1] - first0;
        let second0 = factors[1][2 * row];
        let second_delta = factors[1][2 * row + 1] - second0;
        let third0 = factors[2][2 * row];
        let third_delta = factors[2][2 * row + 1] - third0;
        let mut accumulators = [F::Accumulator::default(); 4];
        accumulate_cubic_product_coefficients(
            &mut accumulators,
            first0,
            first_delta,
            second0,
            second_delta,
            third0,
            third_delta,
        );
        return accumulators;
    }

    let mut coefficients = [F::zero(); 4];
    coefficients[0] = F::one();
    let mut degree = 0;
    for factor in factors {
        let low = factor[2 * row];
        let delta = factor[2 * row + 1] - low;
        for index in (0..=degree).rev() {
            coefficients[index + 1] += coefficients[index] * delta;
            coefficients[index] *= low;
        }
        degree += 1;
    }
    let mut accumulators = [F::Accumulator::default(); 4];
    for (accumulator, coefficient) in accumulators.iter_mut().zip(coefficients) {
        accumulator.acc_add(coefficient);
    }
    accumulators
}

fn accumulate_cubic_product_coefficients<F: Field>(
    coefficients: &mut [F::Accumulator; 4],
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

    coefficients[0].fmadd(first0, second0_third0);
    coefficients[1].fmadd(first_delta, second0_third0);
    coefficients[1].fmadd(first0, second_delta_third0);
    coefficients[1].fmadd(first0, second0_third_delta);
    coefficients[2].fmadd(first_delta, second_delta_third0);
    coefficients[2].fmadd(first_delta, second0_third_delta);
    coefficients[2].fmadd(first0, second_delta_third_delta);
    coefficients[3].fmadd(first_delta, second_delta_third_delta);
}

#[tracing::instrument(skip_all, name = "Stage2DenseState::output_evals")]
fn dense_output_evals<F: Field>(
    outputs: &[(&'static str, &'static str, Vec<F>)],
    point: &[F],
    relation: Stage2Relation,
) -> Result<Vec<F>, Stage2KernelError> {
    let expected =
        1usize
            .checked_shl(point.len() as u32)
            .ok_or(Stage2KernelError::InvalidInputLength {
                input: "stage2.dense_output.point",
                expected: usize::BITS as usize,
                actual: point.len(),
            })?;
    for (_, _, values) in outputs {
        if values.len() != expected {
            return Err(Stage2KernelError::InvalidInputLength {
                input: relation.symbol(),
                expected,
                actual: values.len(),
            });
        }
    }
    let eq = EqPolynomial::new(point.iter().rev().copied().collect()).evaluations();
    let accumulators = if expected >= DENSE_BIND_PAR_THRESHOLD {
        (0..expected)
            .into_par_iter()
            .fold(
                || vec![F::Accumulator::default(); outputs.len()],
                |mut local, index| {
                    let weight = eq[index];
                    for (accumulator, (_, _, values)) in local.iter_mut().zip(outputs) {
                        accumulator.fmadd(values[index], weight);
                    }
                    local
                },
            )
            .reduce(
                || vec![F::Accumulator::default(); outputs.len()],
                |mut left, right| {
                    for (left, right) in left.iter_mut().zip(right) {
                        left.merge(right);
                    }
                    left
                },
            )
    } else {
        (0..expected).fold(
            vec![F::Accumulator::default(); outputs.len()],
            |mut local, index| {
                let weight = eq[index];
                for (accumulator, (_, _, values)) in local.iter_mut().zip(outputs) {
                    accumulator.fmadd(values[index], weight);
                }
                local
            },
        )
    };
    Ok(accumulators
        .into_iter()
        .map(FieldAccumulator::reduce)
        .collect())
}

fn bind_low_to_high<F: Field>(values: &mut Vec<F>, challenge: F) {
    let half = values.len() / 2;
    for index in 0..half {
        values[index] = line(values[2 * index], values[2 * index + 1], challenge);
    }
    values.truncate(half);
}

fn bind_dense_evals_reuse<F: Field>(values: &mut Vec<F>, scratch: &mut Vec<F>, challenge: F) {
    let half = values.len() / 2;
    scratch.resize(half, F::zero());
    if half >= DENSE_BIND_PAR_THRESHOLD {
        scratch
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, output)| {
                let low = values[index << 1];
                let high = values[(index << 1) + 1];
                *output = line(low, high, challenge);
            });
    } else {
        for (index, output) in scratch.iter_mut().enumerate() {
            let low = values[index << 1];
            let high = values[(index << 1) + 1];
            *output = line(low, high, challenge);
        }
    }
    std::mem::swap(values, scratch);
    scratch.clear();
}

fn line<F: Field>(lo: F, hi: F, x: F) -> F {
    lo + x * (hi - lo)
}

fn product_uniskip_base_evals<F: Field>(
    store: &Stage2ValueStore<F>,
) -> Result<[F; PRODUCT_VIRTUAL_UNISKIP_DOMAIN_SIZE], Stage2KernelError> {
    Ok([
        store.scalar("stage2.input.stage1.Product")?,
        store.scalar("stage2.input.stage1.ShouldBranch")?,
        store.scalar("stage2.input.stage1.ShouldJump")?,
    ])
}

#[tracing::instrument(skip_all, name = "stage2_product_virtual_uniskip_extended_evals")]
pub fn product_virtual_uniskip_extended_evals(
    cycles: &[Stage2ProductVirtualCycle],
    tau_low: &[Fr],
) -> Result<[Fr; PRODUCT_VIRTUAL_UNISKIP_DEGREE], Stage2KernelError> {
    let expected =
        1usize
            .checked_shl(tau_low.len() as u32)
            .ok_or(Stage2KernelError::InvalidInputLength {
                input: "stage2.product_virtual.tau_low",
                expected: usize::BITS as usize,
                actual: tau_low.len(),
            })?;
    if cycles.len() != expected {
        return Err(Stage2KernelError::InvalidInputLength {
            input: "stage2.product_virtual.cycles",
            expected,
            actual: cycles.len(),
        });
    }

    let eq_evals = EqPolynomial::new(tau_low.to_vec()).evaluations();
    let accumulators = eq_evals
        .par_iter()
        .zip(cycles.par_iter())
        .fold(
            || [FrSignedScalarAccumulator::zero(); PRODUCT_VIRTUAL_UNISKIP_DEGREE],
            |mut local, (&weight, cycle)| {
                for (target, accumulator) in local.iter_mut().enumerate() {
                    accumulator.fmadd_s256(
                        weight,
                        product_virtual_extended_fused_product(cycle, target),
                    );
                }
                local
            },
        )
        .reduce(
            || [FrSignedScalarAccumulator::zero(); PRODUCT_VIRTUAL_UNISKIP_DEGREE],
            |mut left, right| {
                for (left, right) in left.iter_mut().zip(right) {
                    left.merge(right);
                }
                left
            },
        );

    Ok(accumulators.map(FrSignedScalarAccumulator::reduce))
}

fn product_virtual_extended_fused_product(
    cycle: &Stage2ProductVirtualCycle,
    target: usize,
) -> S256 {
    let coefficients = PRODUCT_VIRTUAL_UNISKIP_TARGET_COEFFS[target];
    let left = coefficients[0] as i128 * cycle.instruction_left_input as i128
        + coefficients[1] as i128 * cycle.should_branch_lookup_output as i128
        + coefficients[2] as i128 * i128::from(cycle.jump_flag);
    let right = coefficients[0] as i128 * cycle.instruction_right_input
        + coefficients[1] as i128 * i128::from(cycle.should_branch_flag)
        + coefficients[2] as i128 * i128::from(cycle.not_next_noop);
    S128::from_i128(left).mul_trunc::<2, 4>(&S128::from_i128(right))
}

#[derive(Clone, Copy)]
struct FrSignedScalarAccumulator {
    positive: Limbs<9>,
    negative: Limbs<9>,
}

impl FrSignedScalarAccumulator {
    fn zero() -> Self {
        Self {
            positive: Limbs::zero(),
            negative: Limbs::zero(),
        }
    }

    fn fmadd_s256(&mut self, field: Fr, scalar: S256) {
        if scalar.magnitude_limbs() == [0u64; 4] {
            return;
        }
        self.fmadd_limbs(field, scalar.as_magnitude(), scalar.is_positive);
    }

    fn fmadd_limbs<const L: usize>(&mut self, field: Fr, scalar: &Limbs<L>, is_positive: bool) {
        let mut product = Limbs::<9>::zero();
        product.fmadd::<4, L>(&field.inner_limbs(), scalar);
        if is_positive {
            self.positive.add_assign_trunc::<9>(&product);
        } else {
            self.negative.add_assign_trunc::<9>(&product);
        }
    }

    fn merge(&mut self, other: Self) {
        self.positive.add_assign_trunc::<9>(&other.positive);
        self.negative.add_assign_trunc::<9>(&other.negative);
    }

    fn reduce(self) -> Fr {
        match self.positive.cmp(&self.negative) {
            Ordering::Greater | Ordering::Equal => {
                Fr::from_barrett_reduced_limbs(self.positive.sub_trunc::<9, 9>(&self.negative))
            }
            Ordering::Less => {
                -Fr::from_barrett_reduced_limbs(self.negative.sub_trunc::<9, 9>(&self.positive))
            }
        }
    }
}

fn build_product_uniskip_poly<F: Field>(
    base_evals: &[F; PRODUCT_VIRTUAL_UNISKIP_DOMAIN_SIZE],
    extended_evals: &[F],
    tau_high: F,
) -> Result<UnivariatePoly<F>, Stage2KernelError> {
    if extended_evals.len() != PRODUCT_VIRTUAL_UNISKIP_DEGREE {
        return Err(Stage2KernelError::InvalidInputLength {
            input: "product_uniskip_extended_evals",
            expected: PRODUCT_VIRTUAL_UNISKIP_DEGREE,
            actual: extended_evals.len(),
        });
    }

    let mut t1_values = vec![F::zero(); PRODUCT_VIRTUAL_UNISKIP_EXTENDED_SIZE];
    for (index, value) in base_evals.iter().enumerate() {
        let target = PRODUCT_VIRTUAL_UNISKIP_DOMAIN_START + index as i64;
        t1_values[(target - PRODUCT_VIRTUAL_UNISKIP_EXTENDED_START) as usize] = *value;
    }
    for (value, target) in extended_evals.iter().zip(product_uniskip_targets()) {
        t1_values[(target - PRODUCT_VIRTUAL_UNISKIP_EXTENDED_START) as usize] = *value;
    }

    let t1_coeffs = interpolate_to_coeffs(PRODUCT_VIRTUAL_UNISKIP_EXTENDED_START, &t1_values);
    let lagrange_values = lagrange_evals(
        PRODUCT_VIRTUAL_UNISKIP_DOMAIN_START,
        PRODUCT_VIRTUAL_UNISKIP_DOMAIN_SIZE,
        tau_high,
    );
    let lagrange_coeffs =
        interpolate_to_coeffs(PRODUCT_VIRTUAL_UNISKIP_DOMAIN_START, &lagrange_values);

    let mut coefficients = vec![F::zero(); PRODUCT_VIRTUAL_UNISKIP_NUM_COEFFS];
    for (i, &lagrange_coeff) in lagrange_coeffs.iter().enumerate() {
        for (j, &t1_coeff) in t1_coeffs.iter().enumerate() {
            coefficients[i + j] += lagrange_coeff * t1_coeff;
        }
    }
    Ok(UnivariatePoly::new(coefficients))
}

fn product_uniskip_targets() -> [i64; PRODUCT_VIRTUAL_UNISKIP_DEGREE] {
    [-2, 2]
}

fn product_uniskip_sum_matches<F: Field>(poly: &UnivariatePoly<F>, claim: F) -> bool {
    (0..PRODUCT_VIRTUAL_UNISKIP_DOMAIN_SIZE)
        .map(|index| {
            poly.evaluate(F::from_i64(
                PRODUCT_VIRTUAL_UNISKIP_DOMAIN_START + index as i64,
            ))
        })
        .sum::<F>()
        == claim
}

fn polynomial_degree<F: Field>(poly: &UnivariatePoly<F>) -> usize {
    poly.coefficients()
        .iter()
        .rposition(|coefficient| *coefficient != F::zero())
        .unwrap_or(0)
}

fn append_univariate_poly<F, T>(transcript: &mut T, label: &'static str, poly: &UnivariatePoly<F>)
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    transcript.append(&LabelWithCount(
        label.as_bytes(),
        poly.coefficients().len() as u64,
    ));
    for coefficient in poly.coefficients() {
        transcript.append(coefficient);
    }
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
    program: &'static Stage2CpuProgramPlan,
    store: &mut Stage2ValueStore<F>,
    transcript: &mut T,
    evals: &[Stage2NamedEval<F>],
) -> Result<(), Stage2KernelError>
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
    let mut seen = seed_stage2_opening_aliases(store, program);
    for batch in program.opening_batches {
        for symbol in batch.claim_operands {
            let claim =
                find_opening_claim(program, symbol).ok_or(Stage2KernelError::MissingClaim {
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

fn seed_stage2_opening_aliases<F: Field>(
    store: &Stage2ValueStore<F>,
    program: &'static Stage2CpuProgramPlan,
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

fn driver_evals<F: Field>(context: Stage2KernelContext<'_>, value: F) -> Vec<Stage2NamedEval<F>> {
    context
        .program
        .evals_for_driver(context.driver.symbol)
        .map(|eval| Stage2NamedEval {
            name: eval.name,
            oracle: eval.oracle,
            value,
        })
        .collect()
}

fn verify_driver_evals<F: Field>(
    driver: &'static str,
    expected: &[Stage2NamedEval<F>],
    actual: &[Stage2NamedEval<F>],
) -> Result<(), Stage2KernelError> {
    if expected.len() != actual.len() {
        return Err(Stage2KernelError::InvalidProof {
            driver,
            reason: "product uniskip eval count mismatch",
        });
    }
    for (expected, actual) in expected.iter().zip(actual) {
        if expected.name != actual.name
            || expected.oracle != actual.oracle
            || expected.value != actual.value
        {
            return Err(Stage2KernelError::InvalidProof {
                driver,
                reason: "product uniskip eval mismatch",
            });
        }
    }
    Ok(())
}

pub fn execute_stage2_program<F, E, T>(
    program: &'static Stage2CpuProgramPlan,
    mode: Stage2ExecutionMode,
    executor: &mut E,
    transcript: &mut T,
) -> Result<Stage2ExecutionArtifacts<F>, Stage2KernelError>
where
    F: Field,
    E: Stage2KernelExecutor<F>,
    T: Transcript<Challenge = F>,
{
    verify_static_program_shape(program)?;
    let mut artifacts = Stage2ExecutionArtifacts::default();
    if program.steps.is_empty() {
        for squeeze in program.transcript_squeezes {
            execute_stage2_squeeze(squeeze, executor, transcript, &mut artifacts)?;
        }
        for driver in program.drivers {
            execute_stage2_driver(program, mode, driver, executor, transcript, &mut artifacts)?;
        }
    } else {
        for step in program.steps {
            match step.kind {
                "transcript_squeeze" => {
                    let squeeze = find_transcript_squeeze(program, step.symbol).ok_or(
                        Stage2KernelError::MissingValue {
                            symbol: step.symbol,
                        },
                    )?;
                    execute_stage2_squeeze(squeeze, executor, transcript, &mut artifacts)?;
                }
                "sumcheck_driver" => {
                    let driver = find_driver(program, step.symbol).ok_or(
                        Stage2KernelError::MissingKernel {
                            driver: step.symbol,
                            kernel: step.symbol,
                        },
                    )?;
                    execute_stage2_driver(
                        program,
                        mode,
                        driver,
                        executor,
                        transcript,
                        &mut artifacts,
                    )?;
                }
                _ => {
                    return Err(Stage2KernelError::InvalidProof {
                        driver: step.symbol,
                        reason: "unsupported stage2 program step",
                    })
                }
            }
        }
    }
    artifacts
        .opening_batches
        .extend(program.opening_batches.iter());
    Ok(artifacts)
}

fn execute_stage2_squeeze<F, E, T>(
    squeeze: &'static Stage2TranscriptSqueezePlan,
    executor: &mut E,
    transcript: &mut T,
    artifacts: &mut Stage2ExecutionArtifacts<F>,
) -> Result<(), Stage2KernelError>
where
    F: Field,
    E: Stage2KernelExecutor<F>,
    T: Transcript<Challenge = F>,
{
    let values = transcript.challenge_vector(squeeze.count);
    executor.observe_challenge_vector(squeeze, &values)?;
    artifacts.challenge_vectors.push(Stage2ChallengeVector {
        symbol: squeeze.symbol,
        values,
    });
    Ok(())
}

fn execute_stage2_driver<F, E, T>(
    program: &'static Stage2CpuProgramPlan,
    mode: Stage2ExecutionMode,
    driver: &'static Stage2SumcheckDriverPlan,
    executor: &mut E,
    transcript: &mut T,
    artifacts: &mut Stage2ExecutionArtifacts<F>,
) -> Result<(), Stage2KernelError>
where
    F: Field,
    E: Stage2KernelExecutor<F>,
    T: Transcript<Challenge = F>,
{
    let kernel = find_kernel(program, driver.kernel).ok_or(Stage2KernelError::MissingKernel {
        driver: driver.symbol,
        kernel: driver.kernel,
    })?;
    let batch = find_batch(program, driver.batch).ok_or(Stage2KernelError::MissingBatch {
        driver: driver.symbol,
        batch: driver.batch,
    })?;
    let context = Stage2KernelContext {
        mode,
        program,
        kernel,
        batch,
        driver,
    };
    let output = match mode {
        Stage2ExecutionMode::Prover => executor.prove_sumcheck(context, transcript)?,
        Stage2ExecutionMode::Verifier => executor.verify_sumcheck(context, transcript)?,
    };
    executor.observe_sumcheck_output(&output)?;
    artifacts.sumchecks.push(output);
    Ok(())
}

fn verify_static_program_shape(
    program: &'static Stage2CpuProgramPlan,
) -> Result<(), Stage2KernelError> {
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
    for step in program.steps {
        match step.kind {
            "transcript_squeeze" => {
                let _ = find_transcript_squeeze(program, step.symbol).ok_or(
                    Stage2KernelError::MissingValue {
                        symbol: step.symbol,
                    },
                )?;
            }
            "sumcheck_driver" => {
                let _ =
                    find_driver(program, step.symbol).ok_or(Stage2KernelError::MissingKernel {
                        driver: step.symbol,
                        kernel: step.symbol,
                    })?;
            }
            _ => {
                return Err(Stage2KernelError::InvalidProof {
                    driver: step.symbol,
                    reason: "unsupported stage2 program step",
                })
            }
        }
    }
    Ok(())
}

fn verify_count(
    artifact: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), Stage2KernelError> {
    if expected == actual {
        Ok(())
    } else {
        Err(Stage2KernelError::PlanCountMismatch {
            artifact,
            expected,
            actual,
        })
    }
}

fn find_kernel<'a>(
    program: &'a Stage2CpuProgramPlan,
    symbol: &str,
) -> Option<&'a Stage2KernelPlan> {
    program
        .kernels
        .iter()
        .find(|kernel| kernel.symbol == symbol)
}

fn find_batch<'a>(
    program: &'a Stage2CpuProgramPlan,
    symbol: &str,
) -> Option<&'a Stage2SumcheckBatchPlan> {
    program.batches.iter().find(|batch| batch.symbol == symbol)
}

fn find_driver<'a>(
    program: &'a Stage2CpuProgramPlan,
    symbol: &str,
) -> Option<&'a Stage2SumcheckDriverPlan> {
    program
        .drivers
        .iter()
        .find(|driver| driver.symbol == symbol)
}

fn find_transcript_squeeze<'a>(
    program: &'a Stage2CpuProgramPlan,
    symbol: &str,
) -> Option<&'a Stage2TranscriptSqueezePlan> {
    program
        .transcript_squeezes
        .iter()
        .find(|squeeze| squeeze.symbol == symbol)
}

fn find_opening_claim<'a>(
    program: &'a Stage2CpuProgramPlan,
    symbol: &str,
) -> Option<&'a Stage2OpeningClaimPlan> {
    program
        .opening_claims
        .iter()
        .find(|claim| claim.symbol == symbol)
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Fr;
    use jolt_transcript::MockTranscript;

    #[test]
    fn stage2_relation_and_abi_registry_is_complete() {
        let relations = [
            Stage2Relation::ProductVirtualUniskip,
            Stage2Relation::RamReadWrite,
            Stage2Relation::ProductVirtualRemainder,
            Stage2Relation::InstructionLookupClaimReduction,
            Stage2Relation::RamRafEvaluation,
            Stage2Relation::RamOutputCheck,
            Stage2Relation::Batched,
        ];
        for relation in relations {
            assert_eq!(
                Stage2Relation::from_symbol(relation.symbol()),
                Some(relation)
            );
        }

        let abis = [
            Stage2KernelAbi::ProductVirtualUniskip,
            Stage2KernelAbi::RamReadWrite,
            Stage2KernelAbi::ProductVirtualRemainder,
            Stage2KernelAbi::InstructionLookupClaimReduction,
            Stage2KernelAbi::RamRafEvaluation,
            Stage2KernelAbi::RamOutputCheck,
            Stage2KernelAbi::Batched,
        ];
        for abi in abis {
            assert_eq!(Stage2KernelAbi::from_name(abi.name()), Some(abi));
        }
    }

    #[test]
    fn stage2_field_exprs_match_jolt_input_claim_formulas() {
        let product_expr = Stage2FieldExprPlan {
            symbol: "product_expr",
            kind: "weighted_sum",
            formula: "jolt_stage2_product_virtual_uniskip_input",
            operand_names: &[],
            operands: &[],
        };
        let product = evaluate_stage2_field_expr(
            &product_expr,
            &[
                Fr::from_u64(0),
                Fr::from_u64(11),
                Fr::from_u64(13),
                Fr::from_u64(17),
            ],
        )
        .expect("product expression evaluates");
        assert_eq!(product, Fr::from_u64(13));

        let ram_expr = Stage2FieldExprPlan {
            symbol: "ram_expr",
            kind: "weighted_sum",
            formula: "jolt_stage2_ram_read_write_input",
            operand_names: &[],
            operands: &[],
        };
        let ram = evaluate_stage2_field_expr(
            &ram_expr,
            &[Fr::from_u64(7), Fr::from_u64(11), Fr::from_u64(13)],
        )
        .expect("ram expression evaluates");
        assert_eq!(ram, Fr::from_u64(102));

        let instruction_expr = Stage2FieldExprPlan {
            symbol: "instruction_expr",
            kind: "weighted_sum",
            formula: "jolt_stage2_instruction_lookup_input",
            operand_names: &[],
            operands: &[],
        };
        let instruction = evaluate_stage2_field_expr(
            &instruction_expr,
            &[
                Fr::from_u64(2),
                Fr::from_u64(1),
                Fr::from_u64(3),
                Fr::from_u64(5),
                Fr::from_u64(7),
                Fr::from_u64(11),
            ],
        )
        .expect("instruction expression evaluates");
        assert_eq!(instruction, Fr::from_u64(259));
    }

    #[test]
    fn value_store_resolves_challenges_openings_and_claims() {
        let program = minimal_program(
            vec![Stage2TranscriptSqueezePlan {
                symbol: "gamma_vec",
                label: "gamma",
                kind: "challenge_scalar",
                count: 1,
            }],
            Vec::new(),
            vec![Stage2ChallengeExtractPlan {
                symbol: "gamma",
                source: "gamma_vec",
                index: 0,
                challenge_source: "gamma_vec",
            }],
            vec![Stage2FieldExprPlan {
                symbol: "ram_expr",
                kind: "weighted_sum",
                formula: "jolt_stage2_ram_read_write_input",
                operand_names: &["gamma", "read", "write"],
                operands: &["gamma", "read", "write"],
            }],
            vec![Stage2SumcheckClaimPlan {
                symbol: "claim",
                stage: "stage2",
                domain: "domain",
                num_rounds: 1,
                degree: 3,
                claim: "claim",
                kernel: "kernel",
                claim_value: "ram_expr",
                input_openings: &["read", "write"],
            }],
            Vec::new(),
            Vec::new(),
        );

        let mut store = Stage2ValueStore::with_opening_inputs(&[
            Stage2OpeningInputValue {
                symbol: "read",
                point: vec![Fr::from_u64(0)],
                eval: Fr::from_u64(11),
            },
            Stage2OpeningInputValue {
                symbol: "write",
                point: vec![Fr::from_u64(1)],
                eval: Fr::from_u64(13),
            },
        ]);
        store
            .observe_challenge_vector(program, &program.transcript_squeezes[0], &[Fr::from_u64(7)])
            .expect("challenge vector observed");
        assert_eq!(
            store
                .claim_value(program, &program.claims[0])
                .expect("claim value resolves"),
            Fr::from_u64(102)
        );
    }

    #[test]
    fn value_store_records_sumcheck_instance_points_and_evals() {
        let program = minimal_program(
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            vec![Stage2SumcheckInstanceResultPlan {
                symbol: "instance",
                source: "driver",
                claim: "claim",
                relation: "relation",
                index: 0,
                point_arity: 2,
                num_rounds: 2,
                round_offset: 1,
                point_order: "as_is",
                degree: 3,
            }],
            vec![Stage2SumcheckEvalPlan {
                symbol: "eval.symbol",
                source: "driver",
                name: "eval.name",
                index: 0,
                oracle: "Oracle",
            }],
        );
        let output = Stage2SumcheckOutput {
            driver: "driver",
            point: vec![Fr::from_u64(3), Fr::from_u64(5), Fr::from_u64(7)],
            evals: vec![Stage2NamedEval {
                name: "eval.name",
                oracle: "Oracle",
                value: Fr::from_u64(11),
            }],
            proof: SumcheckProof::default(),
        };
        let mut store = Stage2ValueStore::new();
        store
            .observe_sumcheck_output(program, &output)
            .expect("sumcheck output observed");

        assert_eq!(
            store.point("instance").expect("instance point"),
            &[Fr::from_u64(5), Fr::from_u64(7)]
        );
        assert_eq!(
            store.scalar("eval.symbol").expect("eval symbol"),
            Fr::from_u64(11)
        );
        assert_eq!(
            store.scalar("eval.name").expect("eval name"),
            Fr::from_u64(11)
        );
    }

    #[test]
    fn product_virtual_uniskip_kernel_proves_first_round_arithmetic() {
        let program = product_uniskip_program();
        let opening_inputs = [
            Stage2OpeningInputValue {
                symbol: "stage2.input.stage1.Product",
                point: vec![Fr::from_u64(0)],
                eval: Fr::from_u64(11),
            },
            Stage2OpeningInputValue {
                symbol: "stage2.input.stage1.ShouldBranch",
                point: vec![Fr::from_u64(0)],
                eval: Fr::from_u64(13),
            },
            Stage2OpeningInputValue {
                symbol: "stage2.input.stage1.ShouldJump",
                point: vec![Fr::from_u64(0)],
                eval: Fr::from_u64(17),
            },
        ];
        let extended_evals = [Fr::from_u64(23), Fr::from_u64(29)];
        let inputs = Stage2ProverInputs::new(&opening_inputs)
            .with_product_uniskip_extended_evals(&extended_evals);
        let mut executor = Stage2ProverKernelExecutor::new(inputs);
        let mut transcript = MockTranscript::<Fr>::new(b"stage2");

        let artifacts = execute_stage2_program(
            program,
            Stage2ExecutionMode::Prover,
            &mut executor,
            &mut transcript,
        )
        .expect("product uniskip proves");

        assert_eq!(artifacts.challenge_vectors.len(), 1);
        assert_eq!(artifacts.sumchecks.len(), 1);
        let output = &artifacts.sumchecks[0];
        let poly = &output.proof.round_polynomials[0];
        let tau = artifacts.challenge_vectors[0].values[0];
        let input_claim = evaluate_stage2_field_expr(
            &program.field_exprs[0],
            &[tau, Fr::from_u64(11), Fr::from_u64(13), Fr::from_u64(17)],
        )
        .expect("input claim evaluates");

        assert_eq!(
            poly.coefficients().len(),
            PRODUCT_VIRTUAL_UNISKIP_NUM_COEFFS
        );
        assert!(product_uniskip_sum_matches(poly, input_claim));
        assert_eq!(output.evals.len(), 1);
        assert_eq!(
            output.evals[0].name,
            "stage2.product_virtual.uniskip.eval.UnivariateSkip"
        );
        assert_eq!(output.evals[0].value, poly.evaluate(output.point[0]));
    }

    #[test]
    fn product_virtual_uniskip_kernel_requires_extended_evals() {
        let program = product_uniskip_program();
        let opening_inputs = [
            Stage2OpeningInputValue {
                symbol: "stage2.input.stage1.Product",
                point: Vec::new(),
                eval: Fr::from_u64(11),
            },
            Stage2OpeningInputValue {
                symbol: "stage2.input.stage1.ShouldBranch",
                point: Vec::new(),
                eval: Fr::from_u64(13),
            },
            Stage2OpeningInputValue {
                symbol: "stage2.input.stage1.ShouldJump",
                point: Vec::new(),
                eval: Fr::from_u64(17),
            },
        ];
        let mut executor =
            Stage2ProverKernelExecutor::new(Stage2ProverInputs::new(&opening_inputs));
        let mut transcript = MockTranscript::<Fr>::new(b"stage2");

        let error = execute_stage2_program(
            program,
            Stage2ExecutionMode::Prover,
            &mut executor,
            &mut transcript,
        )
        .expect_err("missing extended evals are rejected");

        assert_eq!(
            error,
            Stage2KernelError::MissingKernelInput {
                kernel: "jolt_stage2_product_virtual_uniskip",
                input: "product_uniskip_extended_evals",
            }
        );
    }

    #[test]
    fn product_virtual_uniskip_extended_evals_use_trace_cycle_data() {
        let cycles = [
            Stage2ProductVirtualCycle {
                instruction_left_input: 7,
                instruction_right_input: -3,
                should_branch_lookup_output: 11,
                write_lookup_output_to_rd_flag: true,
                jump_flag: false,
                should_branch_flag: true,
                not_next_noop: true,
                virtual_instruction_flag: false,
            },
            Stage2ProductVirtualCycle {
                instruction_left_input: 13,
                instruction_right_input: 5,
                should_branch_lookup_output: 17,
                write_lookup_output_to_rd_flag: false,
                jump_flag: true,
                should_branch_flag: false,
                not_next_noop: false,
                virtual_instruction_flag: true,
            },
        ];
        let tau_low = [Fr::from_u64(19)];
        let evals = product_virtual_uniskip_extended_evals(&cycles, &tau_low)
            .expect("extended evals compute");

        let eq = EqPolynomial::new(tau_low.to_vec()).evaluations();
        let expected = core::array::from_fn(|target| {
            eq.iter()
                .zip(&cycles)
                .map(|(&weight, cycle)| {
                    let product = product_virtual_extended_fused_product(cycle, target);
                    let mut accumulator = FrSignedScalarAccumulator::zero();
                    accumulator.fmadd_s256(weight, product);
                    accumulator.reduce()
                })
                .sum::<Fr>()
        });
        assert_eq!(evals, expected);
    }

    fn minimal_program(
        transcript_squeezes: Vec<Stage2TranscriptSqueezePlan>,
        field_constants: Vec<Stage2FieldConstantPlan>,
        challenge_extracts: Vec<Stage2ChallengeExtractPlan>,
        field_exprs: Vec<Stage2FieldExprPlan>,
        claims: Vec<Stage2SumcheckClaimPlan>,
        instance_results: Vec<Stage2SumcheckInstanceResultPlan>,
        evals: Vec<Stage2SumcheckEvalPlan>,
    ) -> &'static Stage2CpuProgramPlan {
        Box::leak(Box::new(Stage2CpuProgramPlan {
            params: Stage2Params {
                field: "bn254_fr",
                pcs: "dory",
                transcript: "blake2b_transcript",
            },
            steps: &[],
            transcript_squeezes: leak_slice(transcript_squeezes),
            opening_inputs: &[],
            field_constants: leak_slice(field_constants),
            challenge_extracts: leak_slice(challenge_extracts),
            field_exprs: leak_slice(field_exprs),
            kernels: &[],
            claims: leak_slice(claims),
            batches: &[],
            drivers: &[],
            instance_results: leak_slice(instance_results),
            evals: leak_slice(evals),
            point_slices: &[],
            point_concats: &[],
            opening_claims: &[],
            opening_batches: &[],
        }))
    }

    fn leak_slice<T>(values: Vec<T>) -> &'static [T] {
        Box::leak(values.into_boxed_slice())
    }

    fn product_uniskip_program() -> &'static Stage2CpuProgramPlan {
        Box::leak(Box::new(Stage2CpuProgramPlan {
            params: Stage2Params {
                field: "bn254_fr",
                pcs: "dory",
                transcript: "blake2b_transcript",
            },
            steps: &[],
            transcript_squeezes: leak_slice(vec![Stage2TranscriptSqueezePlan {
                symbol: "stage2.product_virtual.tau_high",
                label: "product_virtual_tau_high",
                kind: "challenge_scalar",
                count: 1,
            }]),
            opening_inputs: &[],
            field_constants: &[],
            challenge_extracts: leak_slice(vec![Stage2ChallengeExtractPlan {
                symbol: "stage2.product_virtual.tau_high.scalar",
                source: "stage2.product_virtual.tau_high",
                index: 0,
                challenge_source: "stage2.product_virtual.tau_high",
            }]),
            field_exprs: leak_slice(vec![Stage2FieldExprPlan {
                symbol: "stage2.product_virtual.uniskip.claim_expr",
                kind: "weighted_sum",
                formula: "jolt_stage2_product_virtual_uniskip_input",
                operand_names: &[
                    "stage2.product_virtual.tau_high.scalar",
                    "stage2.input.stage1.Product",
                    "stage2.input.stage1.ShouldBranch",
                    "stage2.input.stage1.ShouldJump",
                ],
                operands: &[
                    "stage2.product_virtual.tau_high.scalar",
                    "stage2.input.stage1.Product",
                    "stage2.input.stage1.ShouldBranch",
                    "stage2.input.stage1.ShouldJump",
                ],
            }]),
            kernels: leak_slice(vec![Stage2KernelPlan {
                symbol: "jolt.cpu.stage2.product_virtual.uniskip",
                relation: "jolt.stage2.product_virtual.uniskip",
                kind: "sumcheck",
                backend: "cpu",
                abi: "jolt_stage2_product_virtual_uniskip",
            }]),
            claims: leak_slice(vec![Stage2SumcheckClaimPlan {
                symbol: "stage2.product_virtual.uniskip.input",
                stage: "stage2",
                domain: "jolt.stage2_uniskip_domain",
                num_rounds: 1,
                degree: 6,
                claim: "stage2.product_virtual.weighted_stage1_outputs",
                kernel: "jolt.cpu.stage2.product_virtual.uniskip",
                claim_value: "stage2.product_virtual.uniskip.claim_expr",
                input_openings: &[
                    "stage2.input.stage1.Product",
                    "stage2.input.stage1.ShouldBranch",
                    "stage2.input.stage1.ShouldJump",
                ],
            }]),
            batches: leak_slice(vec![Stage2SumcheckBatchPlan {
                symbol: "stage2.product_virtual.uniskip.batch",
                stage: "stage2",
                proof_slot: "stage2.product_virtual.uni_skip_first_round",
                policy: "single_instance",
                count: 1,
                ordered_claims: &["stage2.product_virtual.uniskip.input"],
                claim_operands: &["stage2.product_virtual.uniskip.input"],
                claim_label: "uniskip_claim",
                round_label: "uniskip_poly",
                round_schedule: &[1],
            }]),
            drivers: leak_slice(vec![Stage2SumcheckDriverPlan {
                symbol: "stage2.product_virtual.uniskip.sumcheck",
                stage: "stage2",
                proof_slot: "stage2.product_virtual.uni_skip_first_round",
                kernel: "jolt.cpu.stage2.product_virtual.uniskip",
                batch: "stage2.product_virtual.uniskip.batch",
                policy: "univariate_skip",
                round_schedule: &[1],
                claim_label: "uniskip_claim",
                round_label: "uniskip_poly",
                num_rounds: 1,
                degree: 6,
            }]),
            instance_results: &[],
            evals: leak_slice(vec![Stage2SumcheckEvalPlan {
                symbol: "stage2.product_virtual.uniskip.eval.UnivariateSkip",
                source: "stage2.product_virtual.uniskip.sumcheck",
                name: "stage2.product_virtual.uniskip.eval.UnivariateSkip",
                index: 0,
                oracle: "UnivariateSkip",
            }]),
            point_slices: &[],
            point_concats: &[],
            opening_claims: &[],
            opening_batches: &[],
        }))
    }
}
