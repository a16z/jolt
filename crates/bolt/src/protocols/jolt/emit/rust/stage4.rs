#![expect(
    clippy::needless_raw_string_hashes,
    reason = "generated Rust templates are kept as raw string blocks for copyable output"
)]

use std::collections::{BTreeMap, BTreeSet};

use melior::ir::block::BlockLike;
use melior::ir::operation::OperationLike;

use crate::emit::rust::{push_format, EmitError, RustSourceFile};
use crate::ir::{BoltModule, Cpu, Role};
use crate::protocols::jolt::cpu_attrs::{
    operand_symbol, operand_symbols, operation_name, string_attr, symbol_array_attr, symbol_attr,
};
use crate::protocols::jolt::verifier_opening_rows;
use crate::protocols::jolt::verifier_plan::{self, VerifierStagePlan};
use crate::protocols::jolt::verifier_point_rows;
use crate::protocols::jolt::verifier_program_rows;
use crate::protocols::jolt::verifier_relation_outputs::{
    self, RelationOutputAst as Stage4RelationOutputAst,
    RelationOutputFieldExprPlan as Stage4RelationOutputFieldExprPlan,
    RelationOutputPlan as Stage4RelationOutputPlan,
    StructuredPolynomialEvalPlan as Stage4StructuredPolynomialEvalPlan,
};
use crate::protocols::jolt::verifier_sumcheck_rows;
use crate::protocols::jolt::verifier_value_rows;
use crate::protocols::jolt::verifier_values;
use crate::schema::verify_cpu_schema;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4CpuProgram {
    pub role: Role,
    pub(crate) verifier_plan: Option<VerifierStagePlan>,
    pub params: Stage4Params,
    pub steps: Vec<verifier_program_rows::CpuProgramStepPlan>,
    pub transcript_squeezes: Vec<verifier_program_rows::CpuTranscriptSqueezePlan>,
    pub transcript_absorb_bytes: Vec<verifier_program_rows::CpuTranscriptAbsorbBytesPlan>,
    pub opening_inputs: Vec<verifier_program_rows::CpuOpeningInputPlan>,
    pub field_constants: Vec<verifier_value_rows::CpuFieldConstantPlan>,
    pub field_exprs: Vec<verifier_value_rows::CpuFieldExprPlan>,
    pub scalar_exprs: Vec<verifier_value_rows::CpuScalarExprPlan>,
    pub kernels: Vec<Stage4KernelPlan>,
    pub claims: Vec<verifier_sumcheck_rows::CpuSumcheckClaimPlan>,
    pub batches: Vec<verifier_sumcheck_rows::CpuSumcheckBatchPlan>,
    pub drivers: Vec<verifier_sumcheck_rows::CpuSumcheckDriverPlan>,
    pub instance_results: Vec<verifier_sumcheck_rows::CpuSumcheckInstanceResultPlan>,
    pub evals: Vec<verifier_sumcheck_rows::CpuSumcheckEvalPlan>,
    pub relation_output_values: Vec<Stage4StructuredPolynomialEvalPlan>,
    pub relation_outputs: Vec<Stage4RelationOutputPlan>,
    pub point_slices: Vec<verifier_point_rows::CpuPointSlicePlan>,
    pub point_concats: Vec<verifier_point_rows::CpuPointConcatPlan>,
    pub opening_claims: Vec<verifier_opening_rows::CpuOpeningClaimPlan>,
    pub opening_equalities: Vec<verifier_opening_rows::CpuOpeningClaimEqualityPlan>,
    pub opening_batches: Vec<verifier_opening_rows::CpuOpeningBatchPlan>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4Params {
    pub field: String,
    pub pcs: String,
    pub transcript: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4KernelPlan {
    pub symbol: String,
    pub relation: String,
    pub kind: String,
    pub backend: String,
    pub abi: String,
}

fn stage4_scalar_expr(
    expr: Stage4RelationOutputFieldExprPlan,
) -> verifier_value_rows::CpuScalarExprPlan {
    verifier_value_rows::CpuScalarExprPlan::op(expr.symbol, expr.formula, expr.operands)
}

verifier_plan::impl_verifier_plan_source_traits!(
    program = Stage4CpuProgram,
    absorb = transcript_absorb_bytes,
);

pub fn stage4_cpu_program(module: &BoltModule<'_, Cpu>) -> Result<Stage4CpuProgram, EmitError> {
    verify_cpu_schema(module)?;
    let program = Stage4CpuProgram::from_module(module)?;
    program.verify_supported_target()?;
    Ok(program)
}

pub fn emit_stage4_rust(module: &BoltModule<'_, Cpu>) -> Result<RustSourceFile, EmitError> {
    let program = stage4_cpu_program(module)?;

    Ok(RustSourceFile {
        filename: program.filename().to_owned(),
        source: program.emit_source()?,
    })
}

impl Stage4CpuProgram {
    fn from_module(module: &BoltModule<'_, Cpu>) -> Result<Self, EmitError> {
        let mut params = None;
        let mut steps = Vec::new();
        let mut transcript_squeezes = Vec::new();
        let mut transcript_absorb_bytes = Vec::new();
        let mut opening_inputs = Vec::new();
        let mut field_constants = Vec::new();
        let mut field_exprs = Vec::new();
        let mut scalar_exprs = Vec::new();
        let mut kernels = Vec::new();
        let mut claims = Vec::new();
        let mut batches = Vec::new();
        let mut drivers = Vec::new();
        let mut instance_results = Vec::new();
        let mut evals = Vec::new();
        let mut relation_output_values = Vec::new();
        let mut relation_output_asts = Vec::new();
        let mut point_slices = Vec::new();
        let mut point_concats = Vec::new();
        let mut opening_claims = Vec::new();
        let mut opening_equalities = Vec::new();
        let mut opening_batches = Vec::new();

        let mut operation = module.as_mlir_module().body().first_operation();
        while let Some(op) = operation {
            operation = op.next_in_block();
            match operation_name(op).as_str() {
                "cpu.params" => {
                    params = Some(Stage4Params {
                        field: symbol_attr(op, "field")?,
                        pcs: symbol_attr(op, "pcs")?,
                        transcript: symbol_attr(op, "transcript")?,
                    });
                }
                "cpu.kernel" => {
                    kernels.push(Stage4KernelPlan {
                        symbol: string_attr(op, "sym_name")?,
                        relation: symbol_attr(op, "relation")?,
                        kind: string_attr(op, "kind")?,
                        backend: string_attr(op, "backend")?,
                        abi: string_attr(op, "abi")?,
                    });
                }
                "cpu.transcript_squeeze" => {
                    let squeeze = verifier_program_rows::CpuTranscriptSqueezePlan::from_cpu(op)?;
                    steps.push(verifier_program_rows::CpuProgramStepPlan::new(
                        "transcript_squeeze",
                        squeeze.symbol.clone(),
                    ));
                    transcript_squeezes.push(squeeze);
                }
                "cpu.transcript_absorb_bytes" => {
                    let absorb = verifier_program_rows::CpuTranscriptAbsorbBytesPlan::from_cpu(op)?;
                    steps.push(verifier_program_rows::CpuProgramStepPlan::new(
                        "transcript_absorb_bytes",
                        absorb.symbol.clone(),
                    ));
                    transcript_absorb_bytes.push(absorb);
                }
                "cpu.opening_input" => {
                    opening_inputs.push(verifier_program_rows::CpuOpeningInputPlan::from_cpu(op)?);
                }
                "cpu.field_const" => {
                    field_constants
                        .push(verifier_value_rows::CpuFieldConstantPlan::from_const(op)?);
                }
                "cpu.field_zero" => {
                    field_constants.push(verifier_value_rows::CpuFieldConstantPlan::from_zero(op)?);
                }
                "cpu.field_one" => {
                    field_constants.push(verifier_value_rows::CpuFieldConstantPlan::from_one(op)?);
                }
                "cpu.field_add" | "cpu.field_sub" | "cpu.field_mul" | "cpu.field_neg" => {
                    field_exprs.push(verifier_value_rows::CpuFieldExprPlan::from_field_op(op)?);
                }
                "cpu.field_pow" => {
                    field_exprs.push(verifier_value_rows::CpuFieldExprPlan::from_field_pow(op)?);
                }
                "cpu.sumcheck_claim" => {
                    claims.push(verifier_sumcheck_rows::CpuSumcheckClaimPlan::from_claim(
                        op,
                    )?);
                }
                "cpu.sumcheck_verify_claim" => {
                    claims
                        .push(verifier_sumcheck_rows::CpuSumcheckClaimPlan::from_verify_claim(op)?);
                }
                "cpu.sumcheck_batch" => {
                    batches.push(verifier_sumcheck_rows::CpuSumcheckBatchPlan::from_cpu(op)?);
                }
                "cpu.sumcheck_driver" => {
                    let driver = verifier_sumcheck_rows::CpuSumcheckDriverPlan::from_driver(op)?;
                    steps.push(verifier_program_rows::CpuProgramStepPlan::new(
                        "sumcheck_driver",
                        driver.symbol.clone(),
                    ));
                    drivers.push(driver);
                }
                "cpu.sumcheck_verify" => {
                    let driver = verifier_sumcheck_rows::CpuSumcheckDriverPlan::from_verify(op)?;
                    steps.push(verifier_program_rows::CpuProgramStepPlan::new(
                        "sumcheck_driver",
                        driver.symbol.clone(),
                    ));
                    drivers.push(driver);
                }
                "cpu.sumcheck_instance_result" => {
                    instance_results
                        .push(verifier_sumcheck_rows::CpuSumcheckInstanceResultPlan::from_cpu(op)?);
                }
                "cpu.sumcheck_eval" => {
                    evals.push(verifier_sumcheck_rows::CpuSumcheckEvalPlan::from_cpu(op)?);
                }
                "cpu.structured_polynomial_eval" => {
                    relation_output_values.push(
                        verifier_relation_outputs::parse_structured_polynomial_eval_plan(op)?,
                    );
                }
                "cpu.sumcheck_output_claim" => {
                    relation_output_asts.push(Stage4RelationOutputAst {
                        relation: symbol_attr(op, "relation")?,
                        expected_output: operand_symbol(op, 0)?,
                        polynomial_evals: symbol_array_attr(op, "polynomial_evals")?,
                        polynomial_eval_operands: operand_symbols(op, 1)?,
                    });
                }
                "cpu.point_slice" => {
                    point_slices.push(verifier_point_rows::CpuPointSlicePlan::from_cpu(op)?);
                }
                "cpu.point_concat" => {
                    point_concats.push(verifier_point_rows::CpuPointConcatPlan::from_cpu(op)?);
                }
                "cpu.opening_claim" => {
                    opening_claims.push(verifier_opening_rows::CpuOpeningClaimPlan::from_cpu(op)?);
                }
                "cpu.opening_claim_equal" => {
                    opening_equalities
                        .push(verifier_opening_rows::CpuOpeningClaimEqualityPlan::from_cpu(op)?);
                }
                "cpu.opening_batch" => {
                    opening_batches.push(verifier_opening_rows::CpuOpeningBatchPlan::from_cpu(op)?);
                }
                _ => {}
            }
        }

        let role = module
            .role()
            .ok_or_else(|| EmitError::new("missing cpu party role"))?;
        let is_verifier = role == Role::Verifier;
        if is_verifier {
            scalar_exprs.extend(
                relation_output_values
                    .iter()
                    .map(verifier_relation_outputs::structured_polynomial_scalar_expr_plan)
                    .map(stage4_scalar_expr),
            );
        }
        if role == Role::Prover {
            verifier_relation_outputs::prune_output_only_field_exprs(
                &mut field_exprs,
                claims.iter().map(|claim| claim.claim_value.as_str()),
                relation_output_asts
                    .iter()
                    .map(|claim| claim.expected_output.as_str()),
            );
        }
        let relation_outputs = if role == Role::Verifier {
            verifier_relation_outputs::resolve_relation_outputs(
                "stage4",
                &relation_output_values,
                &[],
                &[],
                &[],
                &field_exprs,
                relation_output_asts,
            )?
        } else {
            Vec::new()
        };

        let mut program = Self {
            params: params.ok_or_else(|| EmitError::new("missing cpu.params"))?,
            role,
            verifier_plan: None,
            steps,
            transcript_squeezes,
            transcript_absorb_bytes,
            opening_inputs,
            field_constants,
            field_exprs,
            scalar_exprs,
            kernels,
            claims,
            batches,
            drivers,
            instance_results,
            evals,
            relation_output_values,
            relation_outputs,
            point_slices,
            point_concats,
            opening_claims,
            opening_equalities,
            opening_batches,
        };
        if is_verifier {
            program.verifier_plan = Some(program.plan_verifier()?);
        }
        Ok(program)
    }

    fn plan_verifier(&self) -> Result<VerifierStagePlan, EmitError> {
        verifier_plan::stage_plan_from_cpu_sources(self)
    }

    fn verifier_plan(&self) -> Result<&VerifierStagePlan, EmitError> {
        self.verifier_plan
            .as_ref()
            .ok_or_else(|| EmitError::new("missing stage4 verifier plan"))
    }

    fn verify_supported_target(&self) -> Result<(), EmitError> {
        require_supported_symbol("field", &self.params.field, "bn254_fr")?;
        require_supported_symbol("pcs", &self.params.pcs, "dory")?;
        require_supported_symbol("transcript", &self.params.transcript, "blake2b_transcript")?;
        self.verify_transcript_steps()?;
        self.verify_field_flow()?;
        self.verify_claim_batches()?;
        match self.role {
            Role::Prover => {
                self.verify_kernel_definitions()?;
                self.verify_prover_driver_bindings()?;
            }
            Role::Verifier => self.verify_verifier_driver_bindings()?,
        }
        if self.role == Role::Verifier {
            self.verify_relation_outputs()?;
        }
        self.verify_opening_flow()
    }

    fn verify_transcript_steps(&self) -> Result<(), EmitError> {
        for squeeze in &self.transcript_squeezes {
            if !matches!(
                squeeze.kind.as_str(),
                "challenge_scalar" | "challenge_vector"
            ) {
                return Err(EmitError::new(format!(
                    "stage4 transcript squeeze @{} has unsupported kind `{}`",
                    squeeze.symbol, squeeze.kind
                )));
            }
            if squeeze.count == 0 {
                return Err(EmitError::new(format!(
                    "stage4 transcript squeeze @{} has zero count",
                    squeeze.symbol
                )));
            }
        }
        for absorb in &self.transcript_absorb_bytes {
            if absorb.label.is_empty() {
                return Err(EmitError::new(format!(
                    "stage4 transcript byte absorb @{} has empty label",
                    absorb.symbol
                )));
            }
        }
        Ok(())
    }

    fn verify_field_flow(&self) -> Result<(), EmitError> {
        for constant in &self.field_constants {
            require_supported_symbol("field constant field", &constant.field, "bn254_fr")?;
        }
        let field_values = if self.role == Role::Verifier {
            self.verifier_plan()?.scalar_value_sources()
        } else {
            self.cpu_field_value_sources()
        };
        let point_values = if self.role == Role::Verifier {
            Some(self.verifier_plan()?.point_value_sources())
        } else {
            None
        };
        for expr in &self.field_exprs {
            verify_count(
                "field expr operands",
                &expr.symbol,
                expr.operand_names.len(),
                expr.operands.len(),
            )?;
            for operand in &expr.operands {
                if !field_values.contains(operand) {
                    return Err(EmitError::new(format!(
                        "field expr @{} references missing field value @{operand}",
                        expr.symbol
                    )));
                }
            }
        }
        for expr in &self.scalar_exprs {
            super::plan_tokens::verify_scalar_expr_operands(
                super::plan_tokens::ScalarExprVerification {
                    stage: "stage4",
                    symbol: &expr.symbol,
                    formula: &expr.formula,
                    operand_names: &expr.operand_names,
                    operands: &expr.operands,
                    field_values: &field_values,
                    field_vector_values: None,
                    point_values: point_values.as_ref(),
                },
            )?;
        }
        for claim in &self.claims {
            if !field_values.contains(&claim.claim_value) {
                return Err(EmitError::new(format!(
                    "sumcheck claim @{} uses missing claim value @{}",
                    claim.symbol, claim.claim_value
                )));
            }
        }
        Ok(())
    }

    fn cpu_field_value_sources(&self) -> verifier_values::VerifierScalarSourceSet {
        let mut values = verifier_values::VerifierScalarSourceSet::default();
        values.extend(
            self.opening_inputs.iter().map(|input| &input.symbol),
            verifier_values::VerifierScalarSourceKind::OpeningInput,
        );
        values.extend(
            self.field_constants.iter().map(|constant| &constant.symbol),
            verifier_values::VerifierScalarSourceKind::FieldConstant,
        );
        values.extend(
            self.transcript_squeezes
                .iter()
                .filter(|squeeze| matches!(squeeze.kind.as_str(), "challenge_scalar" | "scalar"))
                .map(|squeeze| &squeeze.symbol),
            verifier_values::VerifierScalarSourceKind::TranscriptScalar,
        );
        values.extend(
            self.field_exprs.iter().map(|expr| &expr.symbol),
            verifier_values::VerifierScalarSourceKind::FieldExpr,
        );
        values.extend(
            self.scalar_exprs.iter().map(|expr| &expr.symbol),
            verifier_values::VerifierScalarSourceKind::ScalarExpr,
        );
        values.extend(
            self.evals.iter().map(|eval| &eval.symbol),
            verifier_values::VerifierScalarSourceKind::SumcheckEval,
        );
        values
    }

    fn verify_kernel_definitions(&self) -> Result<(), EmitError> {
        for kernel in &self.kernels {
            if kernel.backend != "cpu" {
                return Err(EmitError::new(format!(
                    "stage4 kernel @{} targets unsupported backend `{}`",
                    kernel.symbol, kernel.backend
                )));
            }
            if kernel.kind != "sumcheck" {
                return Err(EmitError::new(format!(
                    "stage4 kernel @{} has unsupported kind `{}`",
                    kernel.symbol, kernel.kind
                )));
            }
            let expected_abi = match kernel.relation.as_str() {
                "jolt.stage4.registers_read_write" => "jolt_stage4_registers_read_write",
                "jolt.stage4.ram_val_check" => "jolt_stage4_ram_val_check",
                "jolt.stage4.batched" => "jolt_stage4_batched",
                _ => {
                    return Err(EmitError::new(format!(
                        "unsupported stage4 kernel relation @{}",
                        kernel.relation
                    )));
                }
            };
            if kernel.abi != expected_abi {
                return Err(EmitError::new(format!(
                    "stage4 kernel @{} ABI `{}` does not match relation @{}",
                    kernel.symbol, kernel.abi, kernel.relation
                )));
            }
        }
        Ok(())
    }

    fn verify_claim_batches(&self) -> Result<(), EmitError> {
        let claims = symbols(self.claims.iter().map(|claim| &claim.symbol));
        for batch in &self.batches {
            verify_count(
                "sumcheck batch",
                &batch.symbol,
                batch.count,
                batch.ordered_claims.len(),
            )?;
            verify_count(
                "sumcheck batch operands",
                &batch.symbol,
                batch.count,
                batch.claim_operands.len(),
            )?;
            if batch.ordered_claims != batch.claim_operands {
                return Err(EmitError::new(format!(
                    "sumcheck batch @{} operand order does not match ordered_claims",
                    batch.symbol
                )));
            }
            for claim in &batch.ordered_claims {
                if !claims.contains(claim) {
                    return Err(EmitError::new(format!(
                        "sumcheck batch @{} references missing claim @{claim}",
                        batch.symbol
                    )));
                }
            }
        }
        Ok(())
    }

    fn verify_prover_driver_bindings(&self) -> Result<(), EmitError> {
        let kernels = symbols(self.kernels.iter().map(|kernel| &kernel.symbol));
        let batches: BTreeMap<_, _> = self
            .batches
            .iter()
            .map(|batch| (batch.symbol.as_str(), batch))
            .collect();
        for claim in &self.claims {
            let Some(kernel) = claim.kernel.as_deref() else {
                return Err(EmitError::new(format!(
                    "prover sumcheck claim @{} is missing kernel",
                    claim.symbol
                )));
            };
            if !kernels.contains(kernel) {
                return Err(EmitError::new(format!(
                    "sumcheck claim @{} references missing kernel @{kernel}",
                    claim.symbol
                )));
            }
        }
        for driver in &self.drivers {
            let Some(kernel) = driver.kernel.as_deref() else {
                return Err(EmitError::new(format!(
                    "prover sumcheck driver @{} is missing kernel",
                    driver.symbol
                )));
            };
            if !kernels.contains(kernel) {
                return Err(EmitError::new(format!(
                    "sumcheck driver @{} references missing kernel @{kernel}",
                    driver.symbol
                )));
            }
            let batch = batches.get(driver.batch.as_str()).ok_or_else(|| {
                EmitError::new(format!(
                    "sumcheck driver @{} references missing batch @{}",
                    driver.symbol, driver.batch
                ))
            })?;
            verify_count(
                "sumcheck driver round_schedule",
                &driver.symbol,
                driver.num_rounds,
                driver.round_schedule.iter().sum(),
            )?;
            if driver.round_schedule != batch.round_schedule {
                return Err(EmitError::new(format!(
                    "sumcheck driver @{} round_schedule differs from batch @{}",
                    driver.symbol, batch.symbol
                )));
            }
        }
        Ok(())
    }

    fn verify_verifier_driver_bindings(&self) -> Result<(), EmitError> {
        if !self.kernels.is_empty() {
            return Err(EmitError::new(
                "verifier stage4 program must not contain kernels",
            ));
        }
        let batches: BTreeMap<_, _> = self
            .batches
            .iter()
            .map(|batch| (batch.symbol.as_str(), batch))
            .collect();
        for claim in &self.claims {
            if claim.kernel.is_some() || claim.relation.is_none() {
                return Err(EmitError::new(format!(
                    "verifier sumcheck claim @{} must carry relation and no kernel",
                    claim.symbol
                )));
            }
        }
        for driver in &self.drivers {
            if driver.kernel.is_some() || driver.relation.is_none() {
                return Err(EmitError::new(format!(
                    "verifier sumcheck driver @{} must carry relation and no kernel",
                    driver.symbol
                )));
            }
            let batch = batches.get(driver.batch.as_str()).ok_or_else(|| {
                EmitError::new(format!(
                    "sumcheck driver @{} references missing batch @{}",
                    driver.symbol, driver.batch
                ))
            })?;
            verify_count(
                "sumcheck driver round_schedule",
                &driver.symbol,
                driver.num_rounds,
                driver.round_schedule.iter().sum(),
            )?;
            if driver.round_schedule != batch.round_schedule {
                return Err(EmitError::new(format!(
                    "sumcheck driver @{} round_schedule differs from batch @{}",
                    driver.symbol, batch.symbol
                )));
            }
        }
        Ok(())
    }

    fn verify_relation_outputs(&self) -> Result<(), EmitError> {
        let relations = symbols(
            self.instance_results
                .iter()
                .map(|instance| &instance.relation),
        );
        let plan = self.verifier_plan()?;
        let field_values = plan.scalar_value_sources();
        let point_values = plan.point_value_sources();
        verifier_relation_outputs::verify_relation_outputs(
            "stage4",
            verifier_relation_outputs::RelationOutputVerification {
                relation_output_values: &self.relation_output_values,
                relation_output_eval_families: &[],
                relation_output_product_families: &[],
                relation_output_function_families: &[],
                relation_outputs: &self.relation_outputs,
                relations: &relations,
                field_values: &field_values,
                point_values: &point_values,
            },
        )
    }

    fn verify_opening_flow(&self) -> Result<(), EmitError> {
        let point_sources = if self.role == Role::Verifier {
            self.verifier_plan()?.opening_point_sources()
        } else {
            let mut point_sources = symbols(self.drivers.iter().map(|driver| &driver.symbol));
            point_sources.extend(symbols(
                self.instance_results
                    .iter()
                    .map(|instance| &instance.symbol),
            ));
            point_sources.extend(symbols(
                self.opening_inputs.iter().map(|input| &input.symbol),
            ));
            point_sources.extend(symbols(self.point_slices.iter().map(|slice| &slice.symbol)));
            point_sources.extend(symbols(
                self.point_concats.iter().map(|concat| &concat.symbol),
            ));
            point_sources
        };
        for slice in &self.point_slices {
            if !point_sources.contains(&slice.input) {
                return Err(EmitError::new(format!(
                    "point slice @{} uses missing point source @{}",
                    slice.symbol, slice.input
                )));
            }
        }
        for concat in &self.point_concats {
            for input in &concat.inputs {
                if !point_sources.contains(input) {
                    return Err(EmitError::new(format!(
                        "point concat @{} uses missing point source @{input}",
                        concat.symbol
                    )));
                }
            }
        }
        let eval_sources = if self.role == Role::Verifier {
            self.verifier_plan()?.scalar_value_sources()
        } else {
            self.cpu_field_value_sources()
        };
        let mut opening_sources = symbols(self.opening_inputs.iter().map(|input| &input.symbol));
        opening_sources.extend(symbols(
            self.opening_claims.iter().map(|claim| &claim.symbol),
        ));
        for equality in &self.opening_equalities {
            if !opening_sources.contains(&equality.lhs) {
                return Err(EmitError::new(format!(
                    "opening equality @{} uses missing lhs opening @{}",
                    equality.symbol, equality.lhs
                )));
            }
            if !opening_sources.contains(&equality.rhs) {
                return Err(EmitError::new(format!(
                    "opening equality @{} uses missing rhs opening @{}",
                    equality.symbol, equality.rhs
                )));
            }
        }
        for claim in &self.claims {
            for input in &claim.input_openings {
                if !opening_sources.contains(input) {
                    return Err(EmitError::new(format!(
                        "sumcheck claim @{} uses missing opening @{input}",
                        claim.symbol
                    )));
                }
            }
        }
        let drivers = symbols(self.drivers.iter().map(|driver| &driver.symbol));
        for instance in &self.instance_results {
            if !drivers.contains(&instance.source) {
                return Err(EmitError::new(format!(
                    "sumcheck instance result @{} references missing driver @{}",
                    instance.symbol, instance.source
                )));
            }
        }
        for eval in &self.evals {
            if !drivers.contains(&eval.source) {
                return Err(EmitError::new(format!(
                    "sumcheck eval @{} references missing driver @{}",
                    eval.symbol, eval.source
                )));
            }
        }
        for claim in &self.opening_claims {
            if !point_sources.contains(&claim.point_source) {
                return Err(EmitError::new(format!(
                    "opening claim @{} uses missing point source @{}",
                    claim.symbol, claim.point_source
                )));
            }
            if !eval_sources.contains(&claim.eval_source) {
                return Err(EmitError::new(format!(
                    "opening claim @{} uses missing eval source @{}",
                    claim.symbol, claim.eval_source
                )));
            }
        }
        let openings = symbols(self.opening_claims.iter().map(|claim| &claim.symbol));
        for batch in &self.opening_batches {
            verify_count(
                "opening batch",
                &batch.symbol,
                batch.count,
                batch.ordered_claims.len(),
            )?;
            verify_count(
                "opening batch operands",
                &batch.symbol,
                batch.count,
                batch.claim_operands.len(),
            )?;
            if batch.ordered_claims != batch.claim_operands {
                return Err(EmitError::new(format!(
                    "opening batch @{} operand order does not match ordered_claims",
                    batch.symbol
                )));
            }
            for claim in &batch.ordered_claims {
                if !openings.contains(claim) {
                    return Err(EmitError::new(format!(
                        "opening batch @{} references missing opening @{claim}",
                        batch.symbol
                    )));
                }
            }
        }
        Ok(())
    }

    fn filename(&self) -> &'static str {
        match self.role {
            Role::Prover => "prove_stage4.rs",
            Role::Verifier => "verify_stage4.rs",
        }
    }

    fn emit_source(&self) -> Result<String, EmitError> {
        let mut source = String::new();
        source.push_str("#![allow(dead_code)]\n\n");
        match self.role {
            Role::Prover => {
                source.push_str(Self::emit_prover_imports());
                source.push_str("\n\n");
                source.push_str(Self::emit_prover_types());
            }
            Role::Verifier => {
                source.push_str(Self::emit_verifier_imports());
                source.push_str("\n\n");
                source.push_str(&Self::emit_verifier_types());
            }
        }
        source.push('\n');
        source.push_str(&self.emit_constants()?);
        source.push('\n');
        source.push_str(self.emit_entrypoint());
        Ok(source)
    }

    fn emit_prover_imports() -> &'static str {
        "use jolt_field::Fr;\n\
         use jolt_kernels::stage4::{execute_stage4_program, Stage4CpuProgramPlan, Stage4ExecutionArtifacts, Stage4ExecutionMode, Stage4FieldConstantPlan, Stage4FieldExprPlan, Stage4KernelError, Stage4KernelExecutor, Stage4KernelPlan, Stage4OpeningBatchPlan, Stage4OpeningClaimEqualityPlan, Stage4OpeningClaimPlan, Stage4OpeningInputPlan, Stage4Params, Stage4PointConcatPlan, Stage4PointSlicePlan, Stage4ProgramStepPlan, Stage4SumcheckBatchPlan, Stage4SumcheckClaimPlan, Stage4SumcheckDriverPlan, Stage4SumcheckEvalPlan, Stage4SumcheckInstanceResultPlan, Stage4TranscriptAbsorbBytesPlan, Stage4TranscriptSqueezePlan};\n\
         use jolt_transcript::{Blake2bTranscript, Transcript};"
    }

    fn emit_prover_types() -> &'static str {
        "pub type DefaultStage4Transcript = Blake2bTranscript<Fr>;\n"
    }

    fn emit_verifier_imports() -> &'static str {
        "use bolt_verifier_runtime::{batch_claims, find_batch, find_plan};\n\
         use jolt_field::{Field, Fr};\n\
         use jolt_sumcheck::SumcheckError;\n\
         use jolt_transcript::{Blake2bTranscript, LabelWithCount, Transcript};"
    }

    #[expect(dead_code)]
    fn emit_types() -> &'static str {
        r#"#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
"#
    }

    fn emit_verifier_type_aliases() -> &'static str {
        r#"pub type Stage4NamedEval<F> = bolt_verifier_runtime::StageNamedEval<F>;
pub type Stage4SumcheckOutput<F> = bolt_verifier_runtime::StageSumcheckOutput<F>;
pub type Stage4ChallengeVector<F> = bolt_verifier_runtime::StageChallengeVector<F>;
pub type Stage4ExecutionArtifacts<F> = bolt_verifier_runtime::StageExecutionArtifacts<F>;
pub type Stage4Proof<F> = bolt_verifier_runtime::StageProof<F>;
pub type Stage4OpeningInputValue<F> = bolt_verifier_runtime::StageOpeningInputValue<F>;
pub type Stage4CpuProgramPlan = bolt_verifier_runtime::StageProgramPlan<Stage4RelationKind>;
pub type Stage4SumcheckClaimPlan = bolt_verifier_runtime::SumcheckClaimPlan<Stage4RelationKind>;
pub type Stage4SumcheckDriverPlan = bolt_verifier_runtime::SumcheckDriverPlan<Stage4RelationKind>;
pub type Stage4SumcheckInstanceResultPlan = bolt_verifier_runtime::SumcheckInstanceResultPlan<Stage4RelationKind>;
pub type Stage4RelationOutputPlan = bolt_verifier_runtime::RelationOutputPlan<Stage4RelationKind>;

pub use super::jolt_relations::JoltRelationKind as Stage4RelationKind;
pub use bolt_verifier_runtime::{
    ClaimKind as Stage4ClaimKind, FieldConstantPlan as Stage4FieldConstantPlan,
    FieldExprKind as Stage4FieldExprKind,
    FieldExprPlan as Stage4FieldExprPlan,
    ScalarExprKind as Stage4ScalarExprKind,
    ScalarExprPlan as Stage4ScalarExprPlan,
    KernelPlan as Stage4KernelPlan, OpeningBatchPlan as Stage4OpeningBatchPlan,
    OpeningClaimEqualityPlan as Stage4OpeningClaimEqualityPlan,
    OpeningClaimPlan as Stage4OpeningClaimPlan, OpeningInputPlan as Stage4OpeningInputPlan,
    OpeningEqualityMode as Stage4OpeningEqualityMode, PointExprKind as Stage4PointExprKind,
    PointExprPlan as Stage4PointExprPlan,
    ProgramStepKind as Stage4ProgramStepKind,
    ProgramStepPlan as Stage4ProgramStepPlan, StageParams as Stage4Params,
    SumcheckBatchPlan as Stage4SumcheckBatchPlan,
    SumcheckEvalPlan as Stage4SumcheckEvalPlan,
    TranscriptAbsorbBytesPlan as Stage4TranscriptAbsorbBytesPlan,
    TranscriptSqueezeKind as Stage4TranscriptSqueezeKind,
    TranscriptSqueezePlan as Stage4TranscriptSqueezePlan,
};
"#
    }

    fn emit_verifier_types() -> String {
        let mut source = Self::emit_verifier_type_aliases().to_owned();
        source.push_str(
            r#"
pub type DefaultStage4Transcript = Blake2bTranscript<Fr>;
pub type Stage4VerifierProgramPlan = Stage4CpuProgramPlan;

#[derive(Debug)]
pub enum VerifyStage4Error {
    UnexpectedProofCount { expected: usize, got: usize },
    MissingProof { driver: &'static str },
    MissingBatch { driver: &'static str, batch: &'static str },
    MissingClaim { batch: &'static str, claim: &'static str },
    MissingValue { symbol: &'static str },
    InvalidInputLength { input: &'static str, expected: usize, actual: usize },
    InvalidProof { driver: &'static str, reason: &'static str },
    UnsupportedRelation { relation: Stage4RelationKind },
    Sumcheck { driver: &'static str, error: SumcheckError<Fr> },
}

bolt_verifier_runtime::impl_runtime_plan_error_conversion!(VerifyStage4Error);
"#,
        );
        source
    }

    fn emit_constants(&self) -> Result<String, EmitError> {
        let mut source = self.emit_shared_constants()?;
        source.push_str(&self.emit_kernel_constants());
        source.push_str(&self.emit_sumcheck_claim_constants()?);
        source.push_str(&self.emit_sumcheck_batch_constants());
        source.push_str(&self.emit_sumcheck_driver_constants()?);
        source.push_str(&self.emit_tail_constants()?);
        if self.role == Role::Verifier {
            source.push_str(&self.emit_verifier_relation_output_constants()?);
        }
        let relation_outputs_field = if self.role == Role::Verifier {
            "    relation_outputs: STAGE4_RELATION_OUTPUTS,\n"
        } else {
            ""
        };
        let scalar_exprs_field = if self.role == Role::Verifier {
            "    scalar_exprs: STAGE4_SCALAR_EXPRS,\n"
        } else {
            ""
        };
        let point_exprs_field = if self.role == Role::Verifier {
            "    point_exprs: STAGE4_POINT_EXPRS,\n"
        } else {
            "    point_slices: STAGE4_POINT_SLICES,\n    point_concats: STAGE4_POINT_CONCATS,\n"
        };
        push_format(
            &mut source,
            format_args!(
                "pub const STAGE4_PROGRAM: {} = Stage4CpuProgramPlan {{\n\
                 \x20   role: {},\n\
                 \x20   params: STAGE4_PARAMS,\n\
                 \x20   steps: STAGE4_PROGRAM_STEPS,\n\
                 \x20   transcript_squeezes: STAGE4_TRANSCRIPT_SQUEEZES,\n\
                 \x20   transcript_absorb_bytes: STAGE4_TRANSCRIPT_ABSORB_BYTES,\n\
                 \x20   opening_inputs: STAGE4_OPENING_INPUTS,\n\
                 \x20   field_constants: STAGE4_FIELD_CONSTANTS,\n\
                 \x20   field_exprs: STAGE4_FIELD_EXPRS,\n\
                 {scalar_exprs_field}\
                 \x20   kernels: STAGE4_KERNELS,\n\
                 \x20   claims: STAGE4_SUMCHECK_CLAIMS,\n\
                 \x20   batches: STAGE4_SUMCHECK_BATCHES,\n\
                 \x20   drivers: STAGE4_SUMCHECK_DRIVERS,\n\
                 \x20   instance_results: STAGE4_SUMCHECK_INSTANCE_RESULTS,\n\
                 \x20   evals: STAGE4_SUMCHECK_EVALS,\n\
                 {relation_outputs_field}\
                 {point_exprs_field}\
                 \x20   opening_claims: STAGE4_OPENING_CLAIMS,\n\
                 \x20   opening_equalities: STAGE4_OPENING_EQUALITIES,\n\
                 \x20   opening_batches: STAGE4_OPENING_BATCHES,\n\
                 }};\n",
                self.program_plan_type(),
                rust_str(self.role_label())
            ),
        );
        Ok(source)
    }

    fn emit_shared_constants(&self) -> Result<String, EmitError> {
        let mut source = String::new();
        push_format(
            &mut source,
            format_args!(
                "pub const STAGE4_PARAMS: Stage4Params = Stage4Params {{ field: {}, pcs: {}, transcript: {} }};\n",
                rust_str(&self.params.field),
                rust_str(&self.params.pcs),
                rust_str(&self.params.transcript)
            ),
        );
        source.push_str(&self.emit_program_step_constants()?);
        source.push_str(&self.emit_transcript_squeeze_constants()?);
        source.push_str(&self.emit_transcript_absorb_bytes_constants()?);
        source.push_str(&self.emit_opening_input_constants()?);
        source.push_str(&self.emit_field_constant_constants());
        source.push_str(&self.emit_field_expr_constants()?);
        if self.role == Role::Verifier {
            source.push_str(&self.emit_scalar_expr_constants()?);
        }
        Ok(source)
    }

    fn emit_program_step_constants(&self) -> Result<String, EmitError> {
        if self.role == Role::Verifier {
            let plan = self.verifier_plan()?;
            return Ok(verifier_plan::emit_program_step_constants(
                "Stage4",
                "STAGE4",
                &plan.steps,
            ));
        }
        let steps = self
            .steps
            .iter()
            .map(|step| {
                Ok(format!(
                    "    Stage4ProgramStepPlan {{ kind: {}, symbol: {} }},",
                    super::plan_tokens::role_program_step_kind_expr(
                        "Stage4", &self.role, &step.kind
                    )?,
                    rust_str(&step.symbol),
                ))
            })
            .collect::<Result<Vec<_>, EmitError>>()?
            .join("\n");
        Ok(format!(
            "pub const STAGE4_PROGRAM_STEPS: &[Stage4ProgramStepPlan] = &[\n{steps}\n];\n\n"
        ))
    }

    fn emit_transcript_squeeze_constants(&self) -> Result<String, EmitError> {
        if self.role == Role::Verifier {
            let plan = self.verifier_plan()?;
            return Ok(verifier_plan::emit_transcript_squeeze_constants(
                "Stage4",
                "STAGE4",
                &plan.transcript_squeezes,
            ));
        }
        let squeezes = self
            .transcript_squeezes
            .iter()
            .map(|squeeze| {
                Ok(format!(
                    "    Stage4TranscriptSqueezePlan {{ symbol: {}, label: {}, kind: {}, count: {} }},",
                    rust_str(&squeeze.symbol),
                    rust_str(&squeeze.label),
                    super::plan_tokens::role_transcript_squeeze_kind_expr("Stage4", &self.role, &squeeze.kind)?,
                    squeeze.count,
                ))
            })
            .collect::<Result<Vec<_>, EmitError>>()?
            .join("\n");
        Ok(format!(
            "pub const STAGE4_TRANSCRIPT_SQUEEZES: &[Stage4TranscriptSqueezePlan] = &[\n{squeezes}\n];\n\n"
        ))
    }

    fn emit_transcript_absorb_bytes_constants(&self) -> Result<String, EmitError> {
        if self.role == Role::Verifier {
            let plan = self.verifier_plan()?;
            return Ok(verifier_plan::emit_transcript_absorb_bytes_constants(
                "Stage4",
                "STAGE4",
                &plan.transcript_absorb_bytes,
            ));
        }
        let absorbs = self
            .transcript_absorb_bytes
            .iter()
            .map(|absorb| {
                format!(
                    "    Stage4TranscriptAbsorbBytesPlan {{ symbol: {}, label: {}, payload: {} }},",
                    rust_str(&absorb.symbol),
                    rust_str(&absorb.label),
                    rust_str(&absorb.payload),
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        Ok(format!(
            "pub const STAGE4_TRANSCRIPT_ABSORB_BYTES: &[Stage4TranscriptAbsorbBytesPlan] = &[\n{absorbs}\n];\n\n"
        ))
    }

    fn emit_opening_input_constants(&self) -> Result<String, EmitError> {
        if self.role == Role::Verifier {
            let plan = self.verifier_plan()?;
            return Ok(verifier_plan::emit_opening_input_constants(
                "Stage4",
                "STAGE4",
                &plan.opening_inputs,
            ));
        }
        let inputs = self
            .opening_inputs
            .iter()
            .map(|input| {
                Ok(format!(
                    "    Stage4OpeningInputPlan {{ symbol: {}, source_stage: {}, source_claim: {}, oracle: {}, domain: {}, point_arity: {}, claim_kind: {} }},",
                    rust_str(&input.symbol),
                    rust_str(&input.source_stage),
                    rust_str(&input.source_claim),
                    rust_str(&input.oracle),
                    rust_str(&input.domain),
                    input.point_arity,
                    super::plan_tokens::role_claim_kind_expr("Stage4", &self.role, &input.claim_kind)?
                ))
            })
            .collect::<Result<Vec<_>, EmitError>>()?
            .join("\n");
        Ok(format!(
            "pub const STAGE4_OPENING_INPUTS: &[Stage4OpeningInputPlan] = &[\n{inputs}\n];\n\n"
        ))
    }

    fn emit_field_constant_constants(&self) -> String {
        let constants = self
            .field_constants
            .iter()
            .map(|constant| {
                format!(
                    "    Stage4FieldConstantPlan {{ symbol: {}, field: {}, value: {} }},",
                    rust_str(&constant.symbol),
                    rust_str(&constant.field),
                    constant.value
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "pub const STAGE4_FIELD_CONSTANTS: &[Stage4FieldConstantPlan] = &[\n{constants}\n];\n\n"
        )
    }

    fn emit_field_expr_constants(&self) -> Result<String, EmitError> {
        if self.role == Role::Verifier {
            let plan = self.verifier_plan()?;
            return Ok(verifier_plan::emit_field_expr_constants(
                "Stage4",
                "STAGE4",
                &plan.field_exprs,
            ));
        }

        let mut source = String::new();
        let mut arrays = Vec::new();
        let mut array_refs = Vec::new();
        for (index, expr) in self.field_exprs.iter().enumerate() {
            let operands = intern_str_array(
                &mut source,
                &mut arrays,
                "STAGE4_FIELD_EXPR_OPERANDS",
                &expr.operands,
            );
            let operand_names = intern_str_array(
                &mut source,
                &mut arrays,
                "STAGE4_FIELD_EXPR_OPERANDS",
                &expr.operand_names,
            );
            array_refs.push((index, operand_names, operands));
        }
        let exprs = self
            .field_exprs
            .iter()
            .enumerate()
            .map(|(index, expr)| {
                let (_, operand_names, operands) = &array_refs[index];
                format!(
                    "    Stage4FieldExprPlan {{ symbol: {}, kind: {}, formula: {}, operand_names: {operand_names}, operands: {operands} }},",
                    rust_str(&expr.symbol),
                    rust_str(&expr.kind),
                    rust_str(&expr.formula)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        push_format(
            &mut source,
            format_args!(
                "pub const STAGE4_FIELD_EXPRS: &[Stage4FieldExprPlan] = &[\n{exprs}\n];\n"
            ),
        );
        Ok(source)
    }

    fn emit_scalar_expr_constants(&self) -> Result<String, EmitError> {
        let plan = self.verifier_plan()?;
        Ok(verifier_plan::emit_scalar_expr_constants(
            "Stage4",
            "STAGE4",
            &plan.scalar_exprs,
        ))
    }

    fn emit_kernel_constants(&self) -> String {
        let kernels = self
            .kernels
            .iter()
            .map(|kernel| {
                format!(
                    "    Stage4KernelPlan {{ symbol: {}, relation: {}, kind: {}, backend: {}, abi: {} }},",
                    rust_str(&kernel.symbol),
                    rust_str(&kernel.relation),
                    rust_str(&kernel.kind),
                    rust_str(&kernel.backend),
                    rust_str(&kernel.abi)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!("pub const STAGE4_KERNELS: &[Stage4KernelPlan] = &[\n{kernels}\n];\n\n")
    }

    fn emit_sumcheck_claim_constants(&self) -> Result<String, EmitError> {
        if self.role == Role::Verifier {
            let plan = self.verifier_plan()?;
            return Ok(verifier_plan::emit_sumcheck_claim_constants(
                "Stage4",
                "STAGE4",
                &plan.claims,
            ));
        }

        let mut source = String::new();
        for (index, claim) in self.claims.iter().enumerate() {
            source.push_str(&emit_str_array(
                &format!("STAGE4_SUMCHECK_CLAIM_{index}_INPUT_OPENINGS"),
                &claim.input_openings,
            ));
        }
        let claims = self
            .claims
            .iter()
            .enumerate()
            .map(|(index, claim)| {
                Ok(format!(
                    "    Stage4SumcheckClaimPlan {{ symbol: {}, stage: {}, domain: {}, num_rounds: {}, degree: {}, claim: {}, kernel: {}, relation: {}, claim_value: {}, input_openings: STAGE4_SUMCHECK_CLAIM_{index}_INPUT_OPENINGS }},",
                    rust_str(&claim.symbol),
                    rust_str(&claim.stage),
                    rust_str(&claim.domain),
                    claim.num_rounds,
                    claim.degree,
                    rust_str(&claim.claim),
                    rust_option_str(claim.kernel.as_deref()),
                    super::plan_tokens::role_optional_relation_kind_expr(
                        "Stage4",
                        &self.role,
                        claim.relation.as_deref()
                    )?,
                    rust_str(&claim.claim_value)
                ))
            })
            .collect::<Result<Vec<_>, EmitError>>()?
            .join("\n");
        push_format(
            &mut source,
            format_args!(
                "pub const STAGE4_SUMCHECK_CLAIMS: &[Stage4SumcheckClaimPlan] = &[\n{claims}\n];\n"
            ),
        );
        Ok(source)
    }

    fn emit_sumcheck_batch_constants(&self) -> String {
        if self.role == Role::Verifier {
            let mut source = String::new();
            for (index, batch) in self.batches.iter().enumerate() {
                source.push_str(&emit_usize_array(
                    &format!("STAGE4_SUMCHECK_BATCH_{index}_ROUND_SCHEDULE"),
                    &batch.round_schedule,
                ));
            }
            let batches = self
                .batches
                .iter()
                .enumerate()
                .map(|(index, batch)| {
                    format!(
                        "    Stage4SumcheckBatchPlan {{ symbol: {}, stage: {}, proof_slot: {}, policy: {}, count: {}, claim_operands: {}, claim_label: {}, round_label: {}, round_schedule: STAGE4_SUMCHECK_BATCH_{index}_ROUND_SCHEDULE }},",
                        rust_str(&batch.symbol),
                        rust_str(&batch.stage),
                        rust_str(&batch.proof_slot),
                        rust_str(&batch.policy),
                        batch.count,
                        super::plan_tokens::rust_str_slice_expr(&batch.claim_operands),
                        rust_str(&batch.claim_label),
                        rust_str(&batch.round_label)
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");
            push_format(
                &mut source,
                format_args!(
                    "pub const STAGE4_SUMCHECK_BATCHES: &[Stage4SumcheckBatchPlan] = &[\n{batches}\n];\n"
                ),
            );
            return source;
        }

        let mut source = String::new();
        for (index, batch) in self.batches.iter().enumerate() {
            source.push_str(&emit_str_array(
                &format!("STAGE4_SUMCHECK_BATCH_{index}_ORDERED_CLAIMS"),
                &batch.ordered_claims,
            ));
            source.push_str(&emit_str_array(
                &format!("STAGE4_SUMCHECK_BATCH_{index}_CLAIM_OPERANDS"),
                &batch.claim_operands,
            ));
            source.push_str(&emit_usize_array(
                &format!("STAGE4_SUMCHECK_BATCH_{index}_ROUND_SCHEDULE"),
                &batch.round_schedule,
            ));
        }
        let batches = self
            .batches
            .iter()
            .enumerate()
            .map(|(index, batch)| {
                format!(
                    "    Stage4SumcheckBatchPlan {{ symbol: {}, stage: {}, proof_slot: {}, policy: {}, count: {}, ordered_claims: STAGE4_SUMCHECK_BATCH_{index}_ORDERED_CLAIMS, claim_operands: STAGE4_SUMCHECK_BATCH_{index}_CLAIM_OPERANDS, claim_label: {}, round_label: {}, round_schedule: STAGE4_SUMCHECK_BATCH_{index}_ROUND_SCHEDULE }},",
                    rust_str(&batch.symbol),
                    rust_str(&batch.stage),
                    rust_str(&batch.proof_slot),
                    rust_str(&batch.policy),
                    batch.count,
                    rust_str(&batch.claim_label),
                    rust_str(&batch.round_label)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        push_format(
            &mut source,
            format_args!(
                "pub const STAGE4_SUMCHECK_BATCHES: &[Stage4SumcheckBatchPlan] = &[\n{batches}\n];\n"
            ),
        );
        source
    }

    fn emit_sumcheck_driver_constants(&self) -> Result<String, EmitError> {
        if self.role == Role::Verifier {
            let plan = self.verifier_plan()?;
            return Ok(verifier_plan::emit_sumcheck_driver_constants(
                "Stage4",
                "STAGE4",
                &plan.drivers,
            ));
        }
        let mut source = String::new();
        for (index, driver) in self.drivers.iter().enumerate() {
            source.push_str(&emit_usize_array(
                &format!("STAGE4_SUMCHECK_DRIVER_{index}_ROUND_SCHEDULE"),
                &driver.round_schedule,
            ));
        }
        let drivers = self
            .drivers
            .iter()
            .enumerate()
            .map(|(index, driver)| {
                Ok(format!(
                    "    Stage4SumcheckDriverPlan {{ symbol: {}, stage: {}, proof_slot: {}, kernel: {}, relation: {}, batch: {}, policy: {}, round_schedule: STAGE4_SUMCHECK_DRIVER_{index}_ROUND_SCHEDULE, claim_label: {}, round_label: {}, num_rounds: {}, degree: {} }},",
                    rust_str(&driver.symbol),
                    rust_str(&driver.stage),
                    rust_str(&driver.proof_slot),
                    rust_option_str(driver.kernel.as_deref()),
                    super::plan_tokens::role_optional_relation_kind_expr(
                        "Stage4",
                        &self.role,
                        driver.relation.as_deref()
                    )?,
                    rust_str(&driver.batch),
                    rust_str(&driver.policy),
                    rust_str(&driver.claim_label),
                    rust_str(&driver.round_label),
                    driver.num_rounds,
                    driver.degree
                ))
            })
            .collect::<Result<Vec<_>, EmitError>>()?
            .join("\n");
        push_format(
            &mut source,
            format_args!(
                "pub const STAGE4_SUMCHECK_DRIVERS: &[Stage4SumcheckDriverPlan] = &[\n{drivers}\n];\n"
            ),
        );
        Ok(source)
    }

    fn emit_tail_constants(&self) -> Result<String, EmitError> {
        let mut source = String::new();
        source.push_str(&self.emit_sumcheck_instance_result_constants()?);
        source.push_str(&self.emit_sumcheck_eval_constants());
        source.push_str(&self.emit_point_constants()?);
        source.push_str(&self.emit_opening_claim_constants()?);
        source.push_str(&self.emit_opening_claim_equality_constants()?);
        source.push_str(&self.emit_opening_batch_constants());
        Ok(source)
    }

    fn emit_sumcheck_instance_result_constants(&self) -> Result<String, EmitError> {
        if self.role == Role::Verifier {
            let plan = self.verifier_plan()?;
            return Ok(verifier_plan::emit_sumcheck_instance_result_constants(
                "Stage4",
                "STAGE4",
                &plan.instance_results,
            ));
        }
        let instances = self
            .instance_results
            .iter()
            .map(|instance| {
                Ok(format!(
                    "    Stage4SumcheckInstanceResultPlan {{ symbol: {}, source: {}, claim: {}, relation: {}, index: {}, point_arity: {}, num_rounds: {}, round_offset: {}, point_order: {}, degree: {} }},",
                    rust_str(&instance.symbol),
                    rust_str(&instance.source),
                    rust_str(&instance.claim),
                    super::plan_tokens::role_relation_kind_expr(
                        "Stage4",
                        &self.role,
                        &instance.relation
                    )?,
                    instance.index,
                    instance.point_arity,
                    instance.num_rounds,
                    instance.round_offset,
                    super::plan_tokens::role_sumcheck_point_order_expr(
                        &self.role,
                        &instance.point_order
                    )?,
                    instance.degree
                ))
            })
            .collect::<Result<Vec<_>, EmitError>>()?
            .join("\n");
        Ok(format!(
            "pub const STAGE4_SUMCHECK_INSTANCE_RESULTS: &[Stage4SumcheckInstanceResultPlan] = &[\n{instances}\n];\n\n"
        ))
    }

    fn emit_sumcheck_eval_constants(&self) -> String {
        let evals = self
            .evals
            .iter()
            .map(|eval| {
                format!(
                    "    Stage4SumcheckEvalPlan {{ symbol: {}, source: {}, name: {}, index: {}, oracle: {} }},",
                    rust_str(&eval.symbol),
                    rust_str(&eval.source),
                    rust_str(&eval.name),
                    eval.index,
                    rust_str(&eval.oracle)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!("pub const STAGE4_SUMCHECK_EVALS: &[Stage4SumcheckEvalPlan] = &[\n{evals}\n];\n\n")
    }

    fn emit_verifier_relation_output_constants(&self) -> Result<String, EmitError> {
        super::relation_outputs::emit_verifier_relation_output_constants(
            "Stage4",
            &self.role,
            &self.relation_outputs,
        )
    }

    fn emit_point_constants(&self) -> Result<String, EmitError> {
        if self.role == Role::Verifier {
            let plan = self.verifier_plan()?;
            return Ok(verifier_plan::emit_point_expr_constants(
                "Stage4",
                "STAGE4",
                &plan.point_exprs,
            ));
        }

        let mut source = String::new();
        let slices = self
            .point_slices
            .iter()
            .map(|slice| {
                format!(
                    "    Stage4PointSlicePlan {{ symbol: {}, source: {}, offset: {}, length: {}, input: {} }},",
                    rust_str(&slice.symbol),
                    rust_str(&slice.source),
                    slice.offset,
                    slice.length,
                    rust_str(&slice.input)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        push_format(
            &mut source,
            format_args!(
                "pub const STAGE4_POINT_SLICES: &[Stage4PointSlicePlan] = &[\n{slices}\n];\n\n"
            ),
        );
        for (index, concat) in self.point_concats.iter().enumerate() {
            source.push_str(&emit_str_array(
                &format!("STAGE4_POINT_CONCAT_{index}_INPUTS"),
                &concat.inputs,
            ));
        }
        let concats = self
            .point_concats
            .iter()
            .enumerate()
            .map(|(index, concat)| {
                format!(
                    "    Stage4PointConcatPlan {{ symbol: {}, layout: {}, arity: {}, inputs: STAGE4_POINT_CONCAT_{index}_INPUTS }},",
                    rust_str(&concat.symbol),
                    rust_str(&concat.layout),
                    concat.arity
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        push_format(
            &mut source,
            format_args!(
                "pub const STAGE4_POINT_CONCATS: &[Stage4PointConcatPlan] = &[\n{concats}\n];\n"
            ),
        );
        Ok(source)
    }

    fn emit_opening_claim_constants(&self) -> Result<String, EmitError> {
        if self.role == Role::Verifier {
            let plan = self.verifier_plan()?;
            return Ok(verifier_plan::emit_opening_claim_constants(
                "Stage4",
                "STAGE4",
                &plan.opening_claims,
            ));
        }
        let claims = self
            .opening_claims
            .iter()
            .map(|claim| {
                Ok(format!(
                    "    Stage4OpeningClaimPlan {{ symbol: {}, oracle: {}, domain: {}, point_arity: {}, claim_kind: {}, point_source: {}, eval_source: {} }},",
                    rust_str(&claim.symbol),
                    rust_str(&claim.oracle),
                    rust_str(&claim.domain),
                    claim.point_arity,
                    super::plan_tokens::role_claim_kind_expr("Stage4", &self.role, &claim.claim_kind)?,
                    rust_str(&claim.point_source),
                    rust_str(&claim.eval_source)
                ))
            })
            .collect::<Result<Vec<_>, EmitError>>()?
            .join("\n");
        Ok(format!(
            "pub const STAGE4_OPENING_CLAIMS: &[Stage4OpeningClaimPlan] = &[\n{claims}\n];\n\n"
        ))
    }

    fn emit_opening_claim_equality_constants(&self) -> Result<String, EmitError> {
        if self.role == Role::Verifier {
            let plan = self.verifier_plan()?;
            return Ok(verifier_plan::emit_opening_claim_equality_constants(
                "Stage4",
                "STAGE4",
                &plan.opening_equalities,
            ));
        }
        let equalities = self
            .opening_equalities
            .iter()
            .map(|equality| {
                Ok(format!(
                    "    Stage4OpeningClaimEqualityPlan {{ symbol: {}, mode: {}, lhs: {}, rhs: {} }},",
                    rust_str(&equality.symbol),
                    super::plan_tokens::role_opening_equality_mode_expr("Stage4", &self.role, &equality.mode)?,
                    rust_str(&equality.lhs),
                    rust_str(&equality.rhs)
                ))
            })
            .collect::<Result<Vec<_>, EmitError>>()?
            .join("\n");
        Ok(format!(
            "pub const STAGE4_OPENING_EQUALITIES: &[Stage4OpeningClaimEqualityPlan] = &[\n{equalities}\n];\n\n"
        ))
    }

    fn emit_opening_batch_constants(&self) -> String {
        if self.role == Role::Verifier {
            let batches = self
                .opening_batches
                .iter()
                .map(|batch| {
                    format!(
                        "    Stage4OpeningBatchPlan {{ symbol: {}, stage: {}, proof_slot: {}, policy: {}, count: {}, ordered_claims: {}, claim_operands: {} }},",
                        rust_str(&batch.symbol),
                        rust_str(&batch.stage),
                        rust_str(&batch.proof_slot),
                        rust_str(&batch.policy),
                        batch.count,
                        super::plan_tokens::rust_str_slice_expr(&batch.ordered_claims),
                        super::plan_tokens::rust_str_slice_expr(&batch.claim_operands)
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");
            return format!(
                "pub const STAGE4_OPENING_BATCHES: &[Stage4OpeningBatchPlan] = &[\n{batches}\n];\n"
            );
        }

        let mut source = String::new();
        for (index, batch) in self.opening_batches.iter().enumerate() {
            source.push_str(&emit_str_array(
                &format!("STAGE4_OPENING_BATCH_{index}_ORDERED_CLAIMS"),
                &batch.ordered_claims,
            ));
            source.push_str(&emit_str_array(
                &format!("STAGE4_OPENING_BATCH_{index}_CLAIM_OPERANDS"),
                &batch.claim_operands,
            ));
        }
        let batches = self
            .opening_batches
            .iter()
            .enumerate()
            .map(|(index, batch)| {
                format!(
                    "    Stage4OpeningBatchPlan {{ symbol: {}, stage: {}, proof_slot: {}, policy: {}, count: {}, ordered_claims: STAGE4_OPENING_BATCH_{index}_ORDERED_CLAIMS, claim_operands: STAGE4_OPENING_BATCH_{index}_CLAIM_OPERANDS }},",
                    rust_str(&batch.symbol),
                    rust_str(&batch.stage),
                    rust_str(&batch.proof_slot),
                    rust_str(&batch.policy),
                    batch.count
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        push_format(
            &mut source,
            format_args!(
                "pub const STAGE4_OPENING_BATCHES: &[Stage4OpeningBatchPlan] = &[\n{batches}\n];\n"
            ),
        );
        source
    }

    fn emit_entrypoint(&self) -> &'static str {
        match self.role {
            Role::Prover => {
                "pub fn execute_stage4_prover<E, T>(\n\
                 \x20   executor: &mut E,\n\
                 \x20   transcript: &mut T,\n\
                 ) -> Result<Stage4ExecutionArtifacts<Fr>, Stage4KernelError>\n\
                 where\n\
                 \x20   E: Stage4KernelExecutor<Fr>,\n\
                 \x20   T: Transcript<Challenge = Fr>,\n\
                 {\n\
                 \x20   execute_stage4_prover_with_program(&STAGE4_PROGRAM, executor, transcript)\n\
                 }\n\
                 \n\
                 pub fn execute_stage4_prover_with_program<E, T>(\n\
                 \x20   program: &'static Stage4CpuProgramPlan,\n\
                 \x20   executor: &mut E,\n\
                 \x20   transcript: &mut T,\n\
                 ) -> Result<Stage4ExecutionArtifacts<Fr>, Stage4KernelError>\n\
                 where\n\
                 \x20   E: Stage4KernelExecutor<Fr>,\n\
                 \x20   T: Transcript<Challenge = Fr>,\n\
                 {\n\
                 \x20   execute_stage4_program(program, Stage4ExecutionMode::Prover, executor, transcript)\n\
                 }\n"
            }
            Role::Verifier => {
                r#"pub fn verify_stage4<T>(
    proof: &Stage4Proof<Fr>,
    opening_inputs: &[Stage4OpeningInputValue<Fr>],
    transcript: &mut T,
) -> Result<Stage4ExecutionArtifacts<Fr>, VerifyStage4Error>
where
    T: Transcript<Challenge = Fr>,
{
    verify_stage4_with_program(&STAGE4_PROGRAM, proof, opening_inputs, transcript)
}

pub fn verify_stage4_with_program<T>(
    program: &'static Stage4VerifierProgramPlan,
    proof: &Stage4Proof<Fr>,
    opening_inputs: &[Stage4OpeningInputValue<Fr>],
    transcript: &mut T,
) -> Result<Stage4ExecutionArtifacts<Fr>, VerifyStage4Error>
where
    T: Transcript<Challenge = Fr>,
{
    if proof.sumchecks.len() != program.drivers.len() {
        return Err(VerifyStage4Error::UnexpectedProofCount {
            expected: program.drivers.len(),
            got: proof.sumchecks.len(),
        });
    }
    let mut store =
        bolt_verifier_runtime::ValueStore::with_opening_inputs(opening_inputs, program.opening_inputs)?;
    store.seed_constants(program.field_constants);
    let mut artifacts = Stage4ExecutionArtifacts::default();
    for step in program.steps {
        match step.kind {
            Stage4ProgramStepKind::TranscriptSqueeze => {
                let squeeze =
                    find_plan(program.transcript_squeezes, step.symbol).ok_or(VerifyStage4Error::MissingValue {
                        symbol: step.symbol,
                    })?;
                verify_stage4_squeeze(program, squeeze, &mut store, transcript, &mut artifacts)?;
            }
            Stage4ProgramStepKind::TranscriptAbsorbBytes => {
                let absorb = find_plan(program.transcript_absorb_bytes, step.symbol).ok_or(
                    VerifyStage4Error::MissingValue {
                        symbol: step.symbol,
                    },
                )?;
                absorb_stage4_bytes(absorb, transcript);
            }
            Stage4ProgramStepKind::SumcheckDriver => {
                let driver =
                    find_plan(program.drivers, step.symbol).ok_or(VerifyStage4Error::MissingProof {
                        driver: step.symbol,
                    })?;
                verify_stage4_driver(program, driver, proof, &mut store, transcript, &mut artifacts)?;
            }
        }
    }
    artifacts
        .opening_batches
        .extend(program.opening_batches.iter());
    Ok(artifacts)
}

pub fn stage4_verifier_program() -> &'static Stage4VerifierProgramPlan {
    &STAGE4_PROGRAM
}

fn verify_stage4_squeeze<T>(
    program: &'static Stage4VerifierProgramPlan,
    squeeze: &'static Stage4TranscriptSqueezePlan,
    store: &mut bolt_verifier_runtime::ValueStore<Fr>,
    transcript: &mut T,
    artifacts: &mut Stage4ExecutionArtifacts<Fr>,
) -> Result<(), VerifyStage4Error>
where
    T: Transcript<Challenge = Fr>,
{
    let values = transcript.challenge_vector(squeeze.count);
    store.observe_challenge_vector(squeeze, &values, |input, expected, actual| {
        VerifyStage4Error::InvalidInputLength {
            input,
            expected,
            actual,
        }
    })?;
    store
        .evaluate_available_exprs(program.field_exprs, program.scalar_exprs)
        .map_err(VerifyStage4Error::from)?;
    artifacts.challenge_vectors.push(Stage4ChallengeVector {
        symbol: squeeze.symbol,
        values,
    });
    Ok(())
}

fn absorb_stage4_bytes<T>(absorb: &'static Stage4TranscriptAbsorbBytesPlan, transcript: &mut T)
where
    T: Transcript<Challenge = Fr>,
{
    transcript.append(&LabelWithCount(
        absorb.label.as_bytes(),
        absorb.payload.len() as u64,
    ));
    transcript.append_bytes(absorb.payload.as_bytes());
}

fn verify_stage4_driver<T>(
    program: &'static Stage4VerifierProgramPlan,
    driver: &'static Stage4SumcheckDriverPlan,
    proof: &Stage4Proof<Fr>,
    store: &mut bolt_verifier_runtime::ValueStore<Fr>,
    transcript: &mut T,
    artifacts: &mut Stage4ExecutionArtifacts<Fr>,
) -> Result<(), VerifyStage4Error>
where
    T: Transcript<Challenge = Fr>,
{
    let proof = proof
        .sumchecks
        .get(artifacts.sumchecks.len())
        .ok_or(VerifyStage4Error::MissingProof {
            driver: driver.symbol,
        })?;
    let Some(relation) = driver.relation else {
        return Err(VerifyStage4Error::InvalidProof {
            driver: driver.symbol,
            reason: "missing driver relation",
        });
    };
    let output = match relation {
        Stage4RelationKind::Stage4Batched => {
            verify_batched_stage4(program, driver, proof, store, transcript)?
        }
        relation => return Err(VerifyStage4Error::UnsupportedRelation { relation }),
    };
    artifacts.sumchecks.push(output);
    Ok(())
}

fn verify_batched_stage4<T>(
    program: &'static Stage4VerifierProgramPlan,
    driver: &'static Stage4SumcheckDriverPlan,
    proof: &Stage4SumcheckOutput<Fr>,
    store: &mut bolt_verifier_runtime::ValueStore<Fr>,
    transcript: &mut T,
) -> Result<Stage4SumcheckOutput<Fr>, VerifyStage4Error>
where
    T: Transcript<Challenge = Fr>,
{
    store.evaluate_available_points(
        program.point_exprs,
        |input, expected, actual| VerifyStage4Error::InvalidInputLength {
            input,
            expected,
            actual,
        },
    )?;
    bolt_verifier_runtime::verify_batched_sumcheck(
        driver,
        proof,
        program.claims,
        program.batches,
        program.field_exprs,
        program.scalar_exprs,
        program.opening_inputs,
        program.opening_claims,
        program.opening_batches,
        store,
        transcript,
        |store, evals, point, batching_coeffs| {
            expected_batched_output_claim(program, driver, store, evals, point, batching_coeffs)
        },
        |store, verified| observe_stage4_sumcheck_output(program, store, verified),
        |driver, error| VerifyStage4Error::Sumcheck { driver, error },
    )
}

fn observe_stage4_sumcheck_output<F: Field>(
    program: &'static Stage4VerifierProgramPlan,
    store: &mut bolt_verifier_runtime::ValueStore<F>,
    output: &Stage4SumcheckOutput<F>,
) -> Result<(), VerifyStage4Error> {
    store.observe_sumcheck_output(
        program.instance_results,
        program.evals,
        output,
        |instance, mut point| {
            match instance.point_order {
                bolt_verifier_runtime::SumcheckPointOrder::AsIs => {}
                bolt_verifier_runtime::SumcheckPointOrder::Reverse => point.reverse(),
                bolt_verifier_runtime::SumcheckPointOrder::Stage4RegistersReadWrite => {
                    point = normalize_stage4_registers_rw_point(program, output.driver, &point)?;
                }
                _ => {
                    return Err(VerifyStage4Error::InvalidProof {
                        driver: output.driver,
                        reason: "unsupported point order",
                    });
                }
            }
            Ok(point)
        },
        |input, expected, actual| VerifyStage4Error::InvalidInputLength {
            input,
            expected,
            actual,
        },
        |symbol| VerifyStage4Error::MissingValue { symbol },
    )?;
    store.evaluate_available_points(
        program.point_exprs,
        |input, expected, actual| VerifyStage4Error::InvalidInputLength {
            input,
            expected,
            actual,
        },
    )?;
    store
        .evaluate_available_exprs(program.field_exprs, program.scalar_exprs)
        .map_err(VerifyStage4Error::from)?;
    store.verify_opening_equalities(
        program.opening_equalities,
        |driver, reason| VerifyStage4Error::InvalidProof { driver, reason },
        |symbol| VerifyStage4Error::MissingValue { symbol },
    )
}

fn expected_batched_output_claim(
    program: &'static Stage4VerifierProgramPlan,
    driver: &'static Stage4SumcheckDriverPlan,
    store: &bolt_verifier_runtime::ValueStore<Fr>,
    evals: &[Stage4NamedEval<Fr>],
    point: &[Fr],
    batching_coeffs: &[Fr],
) -> Result<Fr, VerifyStage4Error> {
    let batch = find_batch(program.batches, driver.symbol, driver.batch)?;
    let claims = batch_claims(program.claims, batch)?;
    let mut expected = Fr::from_u64(0);
    for (claim, coefficient) in claims.iter().zip(batching_coeffs) {
        let instance = program
            .instance_results
            .iter()
            .find(|instance| instance.claim == claim.symbol && instance.source == driver.symbol)
            .ok_or(VerifyStage4Error::MissingClaim {
                batch: batch.symbol,
                claim: claim.symbol,
            })?;
        let local_point = point
            .get(instance.round_offset..instance.round_offset + instance.num_rounds)
            .ok_or(VerifyStage4Error::InvalidInputLength {
                input: instance.symbol,
                expected: instance.round_offset + instance.num_rounds,
                actual: point.len(),
            })?;
        let value = bolt_verifier_runtime::evaluate_relation_output_for_instance(
            program.relation_outputs,
            program.field_exprs,
            program.scalar_exprs,
            store,
            instance,
            evals, &[], &[], local_point,
        )?;
        expected += *coefficient * value;
    }
    Ok(expected)
}

fn normalize_stage4_registers_rw_point<F: Field>(
    program: &'static Stage4VerifierProgramPlan,
    driver: &'static str,
    point: &[F],
) -> Result<Vec<F>, VerifyStage4Error> {
    let driver_plan = find_plan(program.drivers, driver).ok_or(VerifyStage4Error::MissingProof {
        driver,
    })?;
    if driver_plan.round_schedule.len() != 2 {
        return Err(VerifyStage4Error::InvalidProof {
            driver,
            reason: "stage4 registers point normalization requires [cycle, address] schedule",
        });
    }
    let cycle_rounds = driver_plan.round_schedule[0];
    let address_rounds = driver_plan.round_schedule[1];
    if point.len() != cycle_rounds + address_rounds {
        return Err(VerifyStage4Error::InvalidInputLength {
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

"#
            }
        }
    }

    fn role_label(&self) -> &'static str {
        match self.role {
            Role::Prover => "prover",
            Role::Verifier => "verifier",
        }
    }

    fn program_plan_type(&self) -> &'static str {
        match self.role {
            Role::Prover => "Stage4CpuProgramPlan",
            Role::Verifier => "Stage4VerifierProgramPlan",
        }
    }
}

fn require_supported_symbol(kind: &str, actual: &str, expected: &str) -> Result<(), EmitError> {
    if actual == expected {
        Ok(())
    } else {
        Err(EmitError::new(format!(
            "unsupported {kind} @{actual}; expected @{expected}"
        )))
    }
}

fn emit_str_array(name: &str, values: &[String]) -> String {
    if values.is_empty() {
        return format!("pub const {name}: &[&str] = &[];\n\n");
    }
    if let [value] = values {
        return format!("pub const {name}: &[&str] = &[{}];\n\n", rust_str(value));
    }
    let entries = values
        .iter()
        .map(|value| format!("    {},", rust_str(value)))
        .collect::<Vec<_>>()
        .join("\n");
    format!("pub const {name}: &[&str] = &[\n{entries}\n];\n\n")
}

fn emit_usize_array(name: &str, values: &[usize]) -> String {
    let entries = values
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    format!("pub const {name}: &[usize] = &[{entries}];\n\n")
}

fn intern_str_array(
    source: &mut String,
    arrays: &mut Vec<(Vec<String>, String)>,
    name_prefix: &str,
    values: &[String],
) -> String {
    if let Some((_, name)) = arrays
        .iter()
        .find(|(existing, _)| existing.as_slice() == values)
    {
        return name.clone();
    }
    let name = format!("{name_prefix}_{}", arrays.len());
    source.push_str(&emit_str_array(&name, values));
    arrays.push((values.to_vec(), name.clone()));
    name
}

fn rust_str(value: &str) -> String {
    format!("{value:?}")
}

fn rust_option_str(value: Option<&str>) -> String {
    value.map_or_else(
        || "None".to_owned(),
        |value| format!("Some({})", rust_str(value)),
    )
}

fn verify_count(kind: &str, symbol: &str, expected: usize, actual: usize) -> Result<(), EmitError> {
    if expected == actual {
        Ok(())
    } else {
        Err(EmitError::new(format!(
            "{kind} @{symbol} count mismatch: expected {expected}, got {actual}"
        )))
    }
}

fn symbols<'a>(values: impl Iterator<Item = &'a String>) -> BTreeSet<String> {
    values.cloned().collect()
}
