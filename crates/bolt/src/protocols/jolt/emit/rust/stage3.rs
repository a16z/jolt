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
    self, RelationOutputAst as Stage3RelationOutputAst,
    RelationOutputFieldExprPlan as Stage3RelationOutputFieldExprPlan,
    RelationOutputPlan as Stage3RelationOutputPlan,
    StructuredPolynomialEvalPlan as Stage3StructuredPolynomialEvalPlan,
};
use crate::protocols::jolt::verifier_sumcheck_rows;
use crate::protocols::jolt::verifier_value_rows;
use crate::protocols::jolt::verifier_values;
use crate::schema::verify_cpu_schema;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3CpuProgram {
    pub role: Role,
    pub(crate) verifier_plan: Option<VerifierStagePlan>,
    pub params: Stage3Params,
    pub steps: Vec<verifier_program_rows::CpuProgramStepPlan>,
    pub transcript_squeezes: Vec<verifier_program_rows::CpuTranscriptSqueezePlan>,
    pub opening_inputs: Vec<verifier_program_rows::CpuOpeningInputPlan>,
    pub field_constants: Vec<verifier_value_rows::CpuFieldConstantPlan>,
    pub field_exprs: Vec<verifier_value_rows::CpuFieldExprPlan>,
    pub scalar_exprs: Vec<verifier_value_rows::CpuScalarExprPlan>,
    pub kernels: Vec<Stage3KernelPlan>,
    pub claims: Vec<verifier_sumcheck_rows::CpuSumcheckClaimPlan>,
    pub batches: Vec<verifier_sumcheck_rows::CpuSumcheckBatchPlan>,
    pub drivers: Vec<verifier_sumcheck_rows::CpuSumcheckDriverPlan>,
    pub instance_results: Vec<verifier_sumcheck_rows::CpuSumcheckInstanceResultPlan>,
    pub evals: Vec<verifier_sumcheck_rows::CpuSumcheckEvalPlan>,
    pub relation_output_values: Vec<Stage3StructuredPolynomialEvalPlan>,
    pub relation_outputs: Vec<Stage3RelationOutputPlan>,
    pub point_slices: Vec<verifier_point_rows::CpuPointSlicePlan>,
    pub point_concats: Vec<verifier_point_rows::CpuPointConcatPlan>,
    pub opening_claims: Vec<verifier_opening_rows::CpuOpeningClaimPlan>,
    pub opening_equalities: Vec<verifier_opening_rows::CpuOpeningClaimEqualityPlan>,
    pub opening_batches: Vec<verifier_opening_rows::CpuOpeningBatchPlan>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3Params {
    pub field: String,
    pub pcs: String,
    pub transcript: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3KernelPlan {
    pub symbol: String,
    pub relation: String,
    pub kind: String,
    pub backend: String,
    pub abi: String,
}

fn stage3_scalar_expr(
    expr: Stage3RelationOutputFieldExprPlan,
) -> verifier_value_rows::CpuScalarExprPlan {
    verifier_value_rows::CpuScalarExprPlan::op(expr.symbol, expr.formula, expr.operands)
}

verifier_plan::impl_verifier_plan_source_traits!(program = Stage3CpuProgram,);

pub fn stage3_cpu_program(module: &BoltModule<'_, Cpu>) -> Result<Stage3CpuProgram, EmitError> {
    verify_cpu_schema(module)?;
    let program = Stage3CpuProgram::from_module(module)?;
    program.verify_supported_target()?;
    Ok(program)
}

pub fn emit_stage3_rust(module: &BoltModule<'_, Cpu>) -> Result<RustSourceFile, EmitError> {
    let program = stage3_cpu_program(module)?;

    Ok(RustSourceFile {
        filename: program.filename().to_owned(),
        source: program.emit_source()?,
    })
}

impl Stage3CpuProgram {
    fn from_module(module: &BoltModule<'_, Cpu>) -> Result<Self, EmitError> {
        let mut params = None;
        let mut steps = Vec::new();
        let mut transcript_squeezes = Vec::new();
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
                    params = Some(Stage3Params {
                        field: symbol_attr(op, "field")?,
                        pcs: symbol_attr(op, "pcs")?,
                        transcript: symbol_attr(op, "transcript")?,
                    });
                }
                "cpu.kernel" => {
                    kernels.push(Stage3KernelPlan {
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
                "cpu.poly_lagrange_basis_eval" => {
                    field_exprs
                        .push(verifier_value_rows::CpuFieldExprPlan::from_lagrange_basis_eval(op)?);
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
                    relation_output_asts.push(Stage3RelationOutputAst {
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
                    .map(stage3_scalar_expr),
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
                "stage3",
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
            .ok_or_else(|| EmitError::new("missing stage3 verifier plan"))
    }

    fn verify_supported_target(&self) -> Result<(), EmitError> {
        require_supported_symbol("field", &self.params.field, "bn254_fr")?;
        require_supported_symbol("pcs", &self.params.pcs, "dory")?;
        require_supported_symbol("transcript", &self.params.transcript, "blake2b_transcript")?;
        self.verify_transcript_squeezes()?;
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

    fn verify_transcript_squeezes(&self) -> Result<(), EmitError> {
        for squeeze in &self.transcript_squeezes {
            if !matches!(
                squeeze.kind.as_str(),
                "challenge_scalar" | "challenge_vector"
            ) {
                return Err(EmitError::new(format!(
                    "stage3 transcript squeeze @{} has unsupported kind `{}`",
                    squeeze.symbol, squeeze.kind
                )));
            }
            if squeeze.count == 0 {
                return Err(EmitError::new(format!(
                    "stage3 transcript squeeze @{} has zero count",
                    squeeze.symbol
                )));
            }
        }
        Ok(())
    }

    fn verify_field_flow(&self) -> Result<(), EmitError> {
        for constant in &self.field_constants {
            require_supported_symbol("field constant field", &constant.field, "bn254_fr")?;
        }
        let verifier_plan = if self.role == Role::Verifier {
            Some(self.verifier_plan()?)
        } else {
            None
        };
        let verifier_scalar_values = verifier_plan.map(|plan| plan.scalar_values());
        let field_values = verifier_scalar_values.as_ref().map_or_else(
            || self.cpu_field_value_sources(),
            |values| values.source_set(),
        );
        let field_vector_values = verifier_plan.map(|plan| plan.field_vector_values());
        let verifier_point_values = verifier_plan.map(|plan| plan.point_values());
        let point_values = verifier_point_values
            .as_ref()
            .map(|values| values.source_set());
        super::plan_tokens::verify_field_expr_flow(
            super::plan_tokens::FieldExprFlowVerification {
                cpu_exprs: &self.field_exprs,
                verifier_exprs: verifier_plan.map(|plan| plan.field_exprs.as_slice()),
                field_values: &field_values,
                verifier_field_values: verifier_scalar_values.as_ref(),
            },
        )?;
        super::plan_tokens::verify_scalar_expr_flow(
            super::plan_tokens::ScalarExprFlowVerification {
                stage: "stage3",
                cpu_exprs: &self.scalar_exprs,
                verifier_exprs: verifier_plan.map(|plan| plan.scalar_exprs.as_slice()),
                field_values: &field_values,
                verifier_field_values: verifier_scalar_values.as_ref(),
                field_vector_values: field_vector_values.as_ref(),
                point_values: point_values.as_ref(),
                verifier_point_values: verifier_point_values.as_ref(),
            },
        )?;
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
                    "stage3 kernel @{} targets unsupported backend `{}`",
                    kernel.symbol, kernel.backend
                )));
            }
            if kernel.kind != "sumcheck" {
                return Err(EmitError::new(format!(
                    "stage3 kernel @{} has unsupported kind `{}`",
                    kernel.symbol, kernel.kind
                )));
            }
            let expected_abi = match kernel.relation.as_str() {
                "jolt.stage3.spartan_shift" => "jolt_stage3_spartan_shift",
                "jolt.stage3.instruction_input" => "jolt_stage3_instruction_input",
                "jolt.stage3.registers_claim_reduction" => "jolt_stage3_registers_claim_reduction",
                "jolt.stage3.batched" => "jolt_stage3_batched",
                _ => {
                    return Err(EmitError::new(format!(
                        "unsupported stage3 kernel relation @{}",
                        kernel.relation
                    )));
                }
            };
            if kernel.abi != expected_abi {
                return Err(EmitError::new(format!(
                    "stage3 kernel @{} ABI `{}` does not match relation @{}",
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
                "verifier stage3 program must not contain kernels",
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
        let field_values = plan.scalar_values();
        let field_vector_values = plan.field_vector_values();
        let point_values = plan.point_value_sources();
        verifier_relation_outputs::verify_relation_outputs(
            "stage3",
            verifier_relation_outputs::RelationOutputVerification {
                relation_output_values: &self.relation_output_values,
                relation_output_eval_families: &[],
                relation_output_product_families: &[],
                relation_output_function_families: &[],
                relation_outputs: &self.relation_outputs,
                relations: &relations,
                field_values: &field_values,
                field_vector_values: &field_vector_values,
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

    fn emit_source(&self) -> Result<String, EmitError> {
        match self.role {
            Role::Prover => self.emit_prover_source(),
            Role::Verifier => self.emit_verifier_source(),
        }
    }

    fn emit_prover_source(&self) -> Result<String, EmitError> {
        let mut source = String::new();
        source.push_str("#![allow(dead_code)]\n\n");
        source.push_str(Self::emit_prover_imports());
        source.push_str("\n\n");
        source.push_str(Self::emit_prover_types());
        source.push('\n');
        source.push_str(&self.emit_prover_constants()?);
        source.push('\n');
        source.push_str(Self::emit_prover_entrypoint());
        Ok(source)
    }

    fn emit_verifier_source(&self) -> Result<String, EmitError> {
        let mut source = String::new();
        source.push_str("#![allow(dead_code)]\n\n");
        source.push_str(Self::emit_verifier_imports());
        source.push_str("\n\n");
        source.push_str(Self::emit_verifier_types());
        source.push('\n');
        source.push_str(&self.emit_verifier_constants()?);
        source.push('\n');
        source.push_str(Self::emit_verifier_entrypoint());
        Ok(source)
    }

    fn filename(&self) -> &'static str {
        match self.role {
            Role::Prover => "prove_stage3.rs",
            Role::Verifier => "verify_stage3.rs",
        }
    }

    fn emit_prover_imports() -> &'static str {
        "use jolt_field::Fr;\n\
         use jolt_kernels::stage3::{execute_stage3_program, Stage3CpuProgramPlan, Stage3ExecutionArtifacts, Stage3ExecutionMode, Stage3FieldConstantPlan, Stage3FieldExprPlan, Stage3KernelError, Stage3KernelExecutor, Stage3KernelPlan, Stage3OpeningBatchPlan, Stage3OpeningClaimEqualityPlan, Stage3OpeningClaimPlan, Stage3OpeningInputPlan, Stage3Params, Stage3PointConcatPlan, Stage3PointSlicePlan, Stage3ProgramStepPlan, Stage3SumcheckBatchPlan, Stage3SumcheckClaimPlan, Stage3SumcheckDriverPlan, Stage3SumcheckEvalPlan, Stage3SumcheckInstanceResultPlan, Stage3TranscriptSqueezePlan};\n\
         use jolt_transcript::{Blake2bTranscript, Transcript};"
    }

    fn emit_prover_types() -> &'static str {
        "pub type DefaultStage3Transcript = Blake2bTranscript<Fr>;\n"
    }

    fn emit_verifier_imports() -> &'static str {
        "use bolt_verifier_runtime::find_plan;\n\
         use jolt_field::{Field, Fr};\n\
         use jolt_sumcheck::SumcheckError;\n\
         use jolt_transcript::{Blake2bTranscript, Transcript};"
    }

    fn emit_verifier_types() -> &'static str {
        r"pub type DefaultStage3Transcript = Blake2bTranscript<Fr>;

pub type Stage3NamedEval<F> = bolt_verifier_runtime::StageNamedEval<F>;
pub type Stage3SumcheckOutput<F> = bolt_verifier_runtime::StageSumcheckOutput<F>;
pub type Stage3ChallengeVector<F> = bolt_verifier_runtime::StageChallengeVector<F>;
pub type Stage3ExecutionArtifacts<F> = bolt_verifier_runtime::StageExecutionArtifacts<F>;
pub type Stage3Proof<F> = bolt_verifier_runtime::StageProof<F>;
pub type Stage3OpeningInputValue<F> = bolt_verifier_runtime::StageOpeningInputValue<F>;
pub type Stage3VerifierProgramPlan = bolt_verifier_runtime::StageVerifierProgramPlan<Stage3RelationKind>;
pub type Stage3SumcheckClaimPlan = bolt_verifier_runtime::SumcheckClaimPlan<Stage3RelationKind>;
pub type Stage3SumcheckDriverPlan = bolt_verifier_runtime::SumcheckDriverPlan<Stage3RelationKind>;
pub type Stage3SumcheckInstanceResultPlan = bolt_verifier_runtime::SumcheckInstanceResultPlan<Stage3RelationKind>;
pub type Stage3RelationOutputPlan = bolt_verifier_runtime::RelationOutputPlan<Stage3RelationKind>;

pub use super::jolt_relations::JoltRelationKind as Stage3RelationKind;
pub use bolt_verifier_runtime::{
    ClaimKind as Stage3ClaimKind, FieldConstantPlan as Stage3FieldConstantPlan,
    FieldExprKind as Stage3FieldExprKind,
    FieldExprPlan as Stage3FieldExprPlan,
    ScalarExprKind as Stage3ScalarExprKind,
    ScalarExprPlan as Stage3ScalarExprPlan,
    OpeningBatchPlan as Stage3OpeningBatchPlan,
    OpeningClaimEqualityPlan as Stage3OpeningClaimEqualityPlan,
    OpeningClaimPlan as Stage3OpeningClaimPlan, OpeningInputPlan as Stage3OpeningInputPlan,
    PointExprKind as Stage3PointExprKind, PointExprPlan as Stage3PointExprPlan,
    OpeningEqualityMode as Stage3OpeningEqualityMode,
    ProgramStepKind as Stage3ProgramStepKind, ProgramStepPlan as Stage3ProgramStepPlan,
    StageParams as Stage3Params,
    SumcheckBatchPlan as Stage3SumcheckBatchPlan, SumcheckEvalPlan as Stage3SumcheckEvalPlan,
    TranscriptSqueezeKind as Stage3TranscriptSqueezeKind,
    TranscriptSqueezePlan as Stage3TranscriptSqueezePlan,
};

#[derive(Debug)]
pub enum VerifyStage3Error {
    UnexpectedProofCount { expected: usize, got: usize },
    MissingProof { driver: &'static str },
    MissingBatch { driver: &'static str, batch: &'static str },
    MissingClaim { batch: &'static str, claim: &'static str },
    MissingValue { symbol: &'static str },
    InvalidInputLength { input: &'static str, expected: usize, actual: usize },
    InvalidProof { driver: &'static str, reason: &'static str },
    UnsupportedRelation { relation: Stage3RelationKind },
    Sumcheck { driver: &'static str, error: SumcheckError<Fr> },
}

bolt_verifier_runtime::impl_runtime_plan_error_conversion!(VerifyStage3Error);
"
    }

    fn emit_prover_constants(&self) -> Result<String, EmitError> {
        let mut source = self.emit_shared_constants()?;
        source.push_str(&self.emit_kernel_constants());
        source.push_str(&self.emit_prover_sumcheck_claim_constants()?);
        source.push_str(&self.emit_sumcheck_batch_constants());
        source.push_str(&self.emit_prover_sumcheck_driver_constants()?);
        source.push_str(&self.emit_tail_constants()?);
        source.push_str(
            "pub const STAGE3_PROGRAM: Stage3CpuProgramPlan = Stage3CpuProgramPlan {\n\
             \x20   params: STAGE3_PARAMS,\n\
             \x20   steps: STAGE3_PROGRAM_STEPS,\n\
             \x20   transcript_squeezes: STAGE3_TRANSCRIPT_SQUEEZES,\n\
             \x20   opening_inputs: STAGE3_OPENING_INPUTS,\n\
             \x20   field_constants: STAGE3_FIELD_CONSTANTS,\n\
             \x20   field_exprs: STAGE3_FIELD_EXPRS,\n\
             \x20   kernels: STAGE3_KERNELS,\n\
             \x20   claims: STAGE3_SUMCHECK_CLAIMS,\n\
             \x20   batches: STAGE3_SUMCHECK_BATCHES,\n\
             \x20   drivers: STAGE3_SUMCHECK_DRIVERS,\n\
             \x20   instance_results: STAGE3_SUMCHECK_INSTANCE_RESULTS,\n\
             \x20   evals: STAGE3_SUMCHECK_EVALS,\n\
             \x20   point_slices: STAGE3_POINT_SLICES,\n\
             \x20   point_concats: STAGE3_POINT_CONCATS,\n\
             \x20   opening_claims: STAGE3_OPENING_CLAIMS,\n\
             \x20   opening_equalities: STAGE3_OPENING_EQUALITIES,\n\
             \x20   opening_batches: STAGE3_OPENING_BATCHES,\n\
             };\n",
        );
        Ok(source)
    }

    fn emit_verifier_constants(&self) -> Result<String, EmitError> {
        let mut source = self.emit_shared_constants()?;
        source.push_str(&self.emit_verifier_sumcheck_claim_constants()?);
        source.push_str(&self.emit_sumcheck_batch_constants());
        source.push_str(&self.emit_verifier_sumcheck_driver_constants()?);
        source.push_str(&self.emit_tail_constants()?);
        source.push_str(&self.emit_verifier_relation_output_constants()?);
        source.push_str(
            "pub const STAGE3_PROGRAM: Stage3VerifierProgramPlan = Stage3VerifierProgramPlan {\n\
             \x20   params: STAGE3_PARAMS,\n\
             \x20   steps: STAGE3_PROGRAM_STEPS,\n\
             \x20   transcript_squeezes: STAGE3_TRANSCRIPT_SQUEEZES,\n\
             \x20   opening_inputs: STAGE3_OPENING_INPUTS,\n\
             \x20   field_constants: STAGE3_FIELD_CONSTANTS,\n\
             \x20   field_exprs: STAGE3_FIELD_EXPRS,\n\
             \x20   scalar_exprs: STAGE3_SCALAR_EXPRS,\n\
             \x20   claims: STAGE3_SUMCHECK_CLAIMS,\n\
             \x20   batches: STAGE3_SUMCHECK_BATCHES,\n\
             \x20   drivers: STAGE3_SUMCHECK_DRIVERS,\n\
             \x20   instance_results: STAGE3_SUMCHECK_INSTANCE_RESULTS,\n\
             \x20   evals: STAGE3_SUMCHECK_EVALS,\n\
             \x20   relation_outputs: STAGE3_RELATION_OUTPUTS,\n\
             \x20   point_exprs: STAGE3_POINT_EXPRS,\n\
             \x20   opening_claims: STAGE3_OPENING_CLAIMS,\n\
             \x20   opening_equalities: STAGE3_OPENING_EQUALITIES,\n\
             \x20   opening_batches: STAGE3_OPENING_BATCHES,\n\
             };\n",
        );
        Ok(source)
    }

    fn emit_shared_constants(&self) -> Result<String, EmitError> {
        let mut source = String::new();
        push_format(
            &mut source,
            format_args!(
                "pub const STAGE3_PARAMS: Stage3Params = Stage3Params {{ field: {}, pcs: {}, transcript: {} }};\n",
                rust_str(&self.params.field),
                rust_str(&self.params.pcs),
                rust_str(&self.params.transcript)
            ),
        );
        source.push_str(&self.emit_program_step_constants()?);
        source.push_str(&self.emit_transcript_squeeze_constants()?);
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
                "Stage3",
                "STAGE3",
                &plan.steps,
            ));
        }
        let steps = self
            .steps
            .iter()
            .map(|step| {
                Ok(format!(
                    "    Stage3ProgramStepPlan {{ kind: {}, symbol: {} }},",
                    super::plan_tokens::role_program_step_kind_expr(
                        "Stage3", &self.role, &step.kind
                    )?,
                    rust_str(&step.symbol),
                ))
            })
            .collect::<Result<Vec<_>, EmitError>>()?
            .join("\n");
        Ok(format!(
            "pub const STAGE3_PROGRAM_STEPS: &[Stage3ProgramStepPlan] = &[\n{steps}\n];\n\n"
        ))
    }

    fn emit_transcript_squeeze_constants(&self) -> Result<String, EmitError> {
        if self.role == Role::Verifier {
            let plan = self.verifier_plan()?;
            return Ok(verifier_plan::emit_transcript_squeeze_constants(
                "Stage3",
                "STAGE3",
                &plan.transcript_squeezes,
            ));
        }
        let squeezes = self
            .transcript_squeezes
            .iter()
            .map(|squeeze| {
                Ok(format!(
                    "    Stage3TranscriptSqueezePlan {{ symbol: {}, label: {}, kind: {}, count: {} }},",
                    rust_str(&squeeze.symbol),
                    rust_str(&squeeze.label),
                    super::plan_tokens::role_transcript_squeeze_kind_expr(
                        "Stage3",
                        &self.role,
                        &squeeze.kind
                    )?,
                    squeeze.count,
                ))
            })
            .collect::<Result<Vec<_>, EmitError>>()?
            .join("\n");
        Ok(format!(
            "pub const STAGE3_TRANSCRIPT_SQUEEZES: &[Stage3TranscriptSqueezePlan] = &[\n{squeezes}\n];\n\n"
        ))
    }

    fn emit_opening_input_constants(&self) -> Result<String, EmitError> {
        if self.role == Role::Verifier {
            let plan = self.verifier_plan()?;
            return Ok(verifier_plan::emit_opening_input_constants(
                "Stage3",
                "STAGE3",
                &plan.opening_inputs,
            ));
        }
        let inputs = self
            .opening_inputs
            .iter()
            .map(|input| {
                Ok(format!(
                    "    Stage3OpeningInputPlan {{ symbol: {}, source_stage: {}, source_claim: {}, oracle: {}, domain: {}, point_arity: {}, claim_kind: {} }},",
                    rust_str(&input.symbol),
                    rust_str(&input.source_stage),
                    rust_str(&input.source_claim),
                    rust_str(&input.oracle),
                    rust_str(&input.domain),
                    input.point_arity,
                    super::plan_tokens::role_claim_kind_expr("Stage3", &self.role, &input.claim_kind)?
                ))
            })
            .collect::<Result<Vec<_>, EmitError>>()?
            .join("\n");
        Ok(format!(
            "pub const STAGE3_OPENING_INPUTS: &[Stage3OpeningInputPlan] = &[\n{inputs}\n];\n\n"
        ))
    }

    fn emit_field_constant_constants(&self) -> String {
        let constants = self
            .field_constants
            .iter()
            .map(|constant| {
                format!(
                    "    Stage3FieldConstantPlan {{ symbol: {}, field: {}, value: {} }},",
                    rust_str(&constant.symbol),
                    rust_str(&constant.field),
                    constant.value
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "pub const STAGE3_FIELD_CONSTANTS: &[Stage3FieldConstantPlan] = &[\n{constants}\n];\n\n"
        )
    }

    fn emit_field_expr_constants(&self) -> Result<String, EmitError> {
        if self.role == Role::Verifier {
            let plan = self.verifier_plan()?;
            return Ok(verifier_plan::emit_field_expr_constants(
                "Stage3",
                "STAGE3",
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
                "STAGE3_FIELD_EXPR_OPERANDS",
                &expr.operands,
            );
            let operand_names = intern_str_array(
                &mut source,
                &mut arrays,
                "STAGE3_FIELD_EXPR_OPERANDS",
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
                    "    Stage3FieldExprPlan {{ symbol: {}, kind: {}, formula: {}, operand_names: {operand_names}, operands: {operands} }},",
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
                "pub const STAGE3_FIELD_EXPRS: &[Stage3FieldExprPlan] = &[\n{exprs}\n];\n"
            ),
        );
        Ok(source)
    }

    fn emit_scalar_expr_constants(&self) -> Result<String, EmitError> {
        let plan = self.verifier_plan()?;
        Ok(verifier_plan::emit_scalar_expr_constants(
            "Stage3",
            "STAGE3",
            &plan.scalar_exprs,
        ))
    }

    fn emit_kernel_constants(&self) -> String {
        let kernels = self
            .kernels
            .iter()
            .map(|kernel| {
                format!(
                    "    Stage3KernelPlan {{ symbol: {}, relation: {}, kind: {}, backend: {}, abi: {} }},",
                    rust_str(&kernel.symbol),
                    rust_str(&kernel.relation),
                    rust_str(&kernel.kind),
                    rust_str(&kernel.backend),
                    rust_str(&kernel.abi)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!("pub const STAGE3_KERNELS: &[Stage3KernelPlan] = &[\n{kernels}\n];\n\n")
    }

    fn emit_prover_sumcheck_claim_constants(&self) -> Result<String, EmitError> {
        self.emit_sumcheck_claim_constants(true)
    }

    fn emit_verifier_sumcheck_claim_constants(&self) -> Result<String, EmitError> {
        self.emit_sumcheck_claim_constants(false)
    }

    fn emit_sumcheck_claim_constants(&self, prover: bool) -> Result<String, EmitError> {
        if !prover {
            let plan = self.verifier_plan()?;
            return Ok(verifier_plan::emit_sumcheck_claim_constants(
                "Stage3",
                "STAGE3",
                &plan.claims,
            ));
        }
        let mut source = String::new();
        for (index, claim) in self.claims.iter().enumerate() {
            source.push_str(&emit_str_array(
                &format!("STAGE3_SUMCHECK_CLAIM_{index}_INPUT_OPENINGS"),
                &claim.input_openings,
            ));
        }
        let mut claims = Vec::new();
        for (index, claim) in self.claims.iter().enumerate() {
            let kernel = claim
                .kernel
                .as_deref()
                .ok_or_else(|| missing_role_binding("prover claim kernel", &claim.symbol))?;
            claims.push(format!(
                        "    Stage3SumcheckClaimPlan {{ symbol: {}, stage: {}, domain: {}, num_rounds: {}, degree: {}, claim: {}, kernel: Some({}), relation: None, claim_value: {}, input_openings: STAGE3_SUMCHECK_CLAIM_{index}_INPUT_OPENINGS }},",
                        rust_str(&claim.symbol),
                        rust_str(&claim.stage),
                        rust_str(&claim.domain),
                        claim.num_rounds,
                        claim.degree,
                        rust_str(&claim.claim),
                        rust_str(kernel),
                        rust_str(&claim.claim_value)
                    ));
        }
        let claims = claims.join("\n");
        push_format(
            &mut source,
            format_args!(
                "pub const STAGE3_SUMCHECK_CLAIMS: &[Stage3SumcheckClaimPlan] = &[\n{claims}\n];\n"
            ),
        );
        Ok(source)
    }

    fn emit_sumcheck_batch_constants(&self) -> String {
        if self.role == Role::Verifier {
            let mut source = String::new();
            for (index, batch) in self.batches.iter().enumerate() {
                source.push_str(&emit_usize_array(
                    &format!("STAGE3_SUMCHECK_BATCH_{index}_ROUND_SCHEDULE"),
                    &batch.round_schedule,
                ));
            }
            let batches = self
                .batches
                .iter()
                .enumerate()
                .map(|(index, batch)| {
                    format!(
                        "    Stage3SumcheckBatchPlan {{ symbol: {}, stage: {}, proof_slot: {}, policy: {}, count: {}, claim_operands: {}, claim_label: {}, round_label: {}, round_schedule: STAGE3_SUMCHECK_BATCH_{index}_ROUND_SCHEDULE }},",
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
                    "pub const STAGE3_SUMCHECK_BATCHES: &[Stage3SumcheckBatchPlan] = &[\n{batches}\n];\n"
                ),
            );
            return source;
        }

        let mut source = String::new();
        for (index, batch) in self.batches.iter().enumerate() {
            source.push_str(&emit_str_array(
                &format!("STAGE3_SUMCHECK_BATCH_{index}_ORDERED_CLAIMS"),
                &batch.ordered_claims,
            ));
            source.push_str(&emit_str_array(
                &format!("STAGE3_SUMCHECK_BATCH_{index}_CLAIM_OPERANDS"),
                &batch.claim_operands,
            ));
            source.push_str(&emit_usize_array(
                &format!("STAGE3_SUMCHECK_BATCH_{index}_ROUND_SCHEDULE"),
                &batch.round_schedule,
            ));
        }
        let batches = self
            .batches
            .iter()
            .enumerate()
            .map(|(index, batch)| {
                format!(
                    "    Stage3SumcheckBatchPlan {{ symbol: {}, stage: {}, proof_slot: {}, policy: {}, count: {}, ordered_claims: STAGE3_SUMCHECK_BATCH_{index}_ORDERED_CLAIMS, claim_operands: STAGE3_SUMCHECK_BATCH_{index}_CLAIM_OPERANDS, claim_label: {}, round_label: {}, round_schedule: STAGE3_SUMCHECK_BATCH_{index}_ROUND_SCHEDULE }},",
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
                "pub const STAGE3_SUMCHECK_BATCHES: &[Stage3SumcheckBatchPlan] = &[\n{batches}\n];\n"
            ),
        );
        source
    }

    fn emit_prover_sumcheck_driver_constants(&self) -> Result<String, EmitError> {
        self.emit_sumcheck_driver_constants(true)
    }

    fn emit_verifier_sumcheck_driver_constants(&self) -> Result<String, EmitError> {
        self.emit_sumcheck_driver_constants(false)
    }

    fn emit_sumcheck_driver_constants(&self, prover: bool) -> Result<String, EmitError> {
        if !prover {
            let plan = self.verifier_plan()?;
            return Ok(verifier_plan::emit_sumcheck_driver_constants(
                "Stage3",
                "STAGE3",
                &plan.drivers,
            ));
        }
        let mut source = String::new();
        for (index, driver) in self.drivers.iter().enumerate() {
            source.push_str(&emit_usize_array(
                &format!("STAGE3_SUMCHECK_DRIVER_{index}_ROUND_SCHEDULE"),
                &driver.round_schedule,
            ));
        }
        let mut drivers = Vec::new();
        for (index, driver) in self.drivers.iter().enumerate() {
            let kernel = driver
                .kernel
                .as_deref()
                .ok_or_else(|| missing_role_binding("prover driver kernel", &driver.symbol))?;
            drivers.push(format!(
                        "    Stage3SumcheckDriverPlan {{ symbol: {}, stage: {}, proof_slot: {}, kernel: Some({}), relation: None, batch: {}, policy: {}, round_schedule: STAGE3_SUMCHECK_DRIVER_{index}_ROUND_SCHEDULE, claim_label: {}, round_label: {}, num_rounds: {}, degree: {} }},",
                        rust_str(&driver.symbol),
                        rust_str(&driver.stage),
                        rust_str(&driver.proof_slot),
                        rust_str(kernel),
                        rust_str(&driver.batch),
                        rust_str(&driver.policy),
                        rust_str(&driver.claim_label),
                        rust_str(&driver.round_label),
                        driver.num_rounds,
                        driver.degree
                    ));
        }
        let drivers = drivers.join("\n");
        push_format(
            &mut source,
            format_args!(
                "pub const STAGE3_SUMCHECK_DRIVERS: &[Stage3SumcheckDriverPlan] = &[\n{drivers}\n];\n"
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
                "Stage3",
                "STAGE3",
                &plan.instance_results,
            ));
        }
        let instances = self
            .instance_results
            .iter()
            .map(|instance| {
                Ok(format!(
                    "    Stage3SumcheckInstanceResultPlan {{ symbol: {}, source: {}, claim: {}, relation: {}, index: {}, point_arity: {}, num_rounds: {}, round_offset: {}, point_order: {}, degree: {} }},",
                    rust_str(&instance.symbol),
                    rust_str(&instance.source),
                    rust_str(&instance.claim),
                    super::plan_tokens::role_relation_kind_expr(
                        "Stage3",
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
            "pub const STAGE3_SUMCHECK_INSTANCE_RESULTS: &[Stage3SumcheckInstanceResultPlan] = &[\n{instances}\n];\n\n"
        ))
    }

    fn emit_sumcheck_eval_constants(&self) -> String {
        let evals = self
            .evals
            .iter()
            .map(|eval| {
                format!(
                    "    Stage3SumcheckEvalPlan {{ symbol: {}, source: {}, name: {}, index: {}, oracle: {} }},",
                    rust_str(&eval.symbol),
                    rust_str(&eval.source),
                    rust_str(&eval.name),
                    eval.index,
                    rust_str(&eval.oracle)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!("pub const STAGE3_SUMCHECK_EVALS: &[Stage3SumcheckEvalPlan] = &[\n{evals}\n];\n\n")
    }

    fn emit_verifier_relation_output_constants(&self) -> Result<String, EmitError> {
        super::relation_outputs::emit_verifier_relation_output_constants(
            "Stage3",
            &self.role,
            &self.relation_outputs,
        )
    }

    fn emit_point_constants(&self) -> Result<String, EmitError> {
        if self.role == Role::Verifier {
            let plan = self.verifier_plan()?;
            return Ok(verifier_plan::emit_point_expr_constants(
                "Stage3",
                "STAGE3",
                &plan.point_exprs,
            ));
        }

        let mut source = String::new();
        let slices = self
            .point_slices
            .iter()
            .map(|slice| {
                format!(
                    "    Stage3PointSlicePlan {{ symbol: {}, source: {}, offset: {}, length: {}, input: {} }},",
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
                "pub const STAGE3_POINT_SLICES: &[Stage3PointSlicePlan] = &[\n{slices}\n];\n\n"
            ),
        );
        for (index, concat) in self.point_concats.iter().enumerate() {
            source.push_str(&emit_str_array(
                &format!("STAGE3_POINT_CONCAT_{index}_INPUTS"),
                &concat.inputs,
            ));
        }
        let concats = self
            .point_concats
            .iter()
            .enumerate()
            .map(|(index, concat)| {
                format!(
                    "    Stage3PointConcatPlan {{ symbol: {}, layout: {}, arity: {}, inputs: STAGE3_POINT_CONCAT_{index}_INPUTS }},",
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
                "pub const STAGE3_POINT_CONCATS: &[Stage3PointConcatPlan] = &[\n{concats}\n];\n"
            ),
        );
        Ok(source)
    }

    fn emit_opening_claim_constants(&self) -> Result<String, EmitError> {
        if self.role == Role::Verifier {
            let plan = self.verifier_plan()?;
            return Ok(verifier_plan::emit_opening_claim_constants(
                "Stage3",
                "STAGE3",
                &plan.opening_claims,
            ));
        }
        let claims = self
            .opening_claims
            .iter()
            .map(|claim| {
                Ok(format!(
                    "    Stage3OpeningClaimPlan {{ symbol: {}, oracle: {}, domain: {}, point_arity: {}, claim_kind: {}, point_source: {}, eval_source: {} }},",
                    rust_str(&claim.symbol),
                    rust_str(&claim.oracle),
                    rust_str(&claim.domain),
                    claim.point_arity,
                    super::plan_tokens::role_claim_kind_expr("Stage3", &self.role, &claim.claim_kind)?,
                    rust_str(&claim.point_source),
                    rust_str(&claim.eval_source)
                ))
            })
            .collect::<Result<Vec<_>, EmitError>>()?
            .join("\n");
        Ok(format!(
            "pub const STAGE3_OPENING_CLAIMS: &[Stage3OpeningClaimPlan] = &[\n{claims}\n];\n\n"
        ))
    }

    fn emit_opening_claim_equality_constants(&self) -> Result<String, EmitError> {
        if self.role == Role::Verifier {
            let plan = self.verifier_plan()?;
            return Ok(verifier_plan::emit_opening_claim_equality_constants(
                "Stage3",
                "STAGE3",
                &plan.opening_equalities,
            ));
        }
        let equalities = self
            .opening_equalities
            .iter()
            .map(|equality| {
                Ok(format!(
                    "    Stage3OpeningClaimEqualityPlan {{ symbol: {}, mode: {}, lhs: {}, rhs: {} }},",
                    rust_str(&equality.symbol),
                    super::plan_tokens::role_opening_equality_mode_expr(
                        "Stage3",
                        &self.role,
                        &equality.mode
                    )?,
                    rust_str(&equality.lhs),
                    rust_str(&equality.rhs)
                ))
            })
            .collect::<Result<Vec<_>, EmitError>>()?
            .join("\n");
        Ok(format!(
            "pub const STAGE3_OPENING_EQUALITIES: &[Stage3OpeningClaimEqualityPlan] = &[\n{equalities}\n];\n\n"
        ))
    }

    fn emit_opening_batch_constants(&self) -> String {
        if self.role == Role::Verifier {
            let batches = self
                .opening_batches
                .iter()
                .map(|batch| {
                    format!(
                        "    Stage3OpeningBatchPlan {{ symbol: {}, stage: {}, proof_slot: {}, policy: {}, count: {}, ordered_claims: {}, claim_operands: {} }},",
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
                "pub const STAGE3_OPENING_BATCHES: &[Stage3OpeningBatchPlan] = &[\n{batches}\n];\n"
            );
        }

        let mut source = String::new();
        for (index, batch) in self.opening_batches.iter().enumerate() {
            source.push_str(&emit_str_array(
                &format!("STAGE3_OPENING_BATCH_{index}_ORDERED_CLAIMS"),
                &batch.ordered_claims,
            ));
            source.push_str(&emit_str_array(
                &format!("STAGE3_OPENING_BATCH_{index}_CLAIM_OPERANDS"),
                &batch.claim_operands,
            ));
        }
        let batches = self
            .opening_batches
            .iter()
            .enumerate()
            .map(|(index, batch)| {
                format!(
                    "    Stage3OpeningBatchPlan {{ symbol: {}, stage: {}, proof_slot: {}, policy: {}, count: {}, ordered_claims: STAGE3_OPENING_BATCH_{index}_ORDERED_CLAIMS, claim_operands: STAGE3_OPENING_BATCH_{index}_CLAIM_OPERANDS }},",
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
                "pub const STAGE3_OPENING_BATCHES: &[Stage3OpeningBatchPlan] = &[\n{batches}\n];\n"
            ),
        );
        source
    }

    fn emit_prover_entrypoint() -> &'static str {
        "pub fn execute_stage3_prover<E, T>(\n\
         \x20   executor: &mut E,\n\
         \x20   transcript: &mut T,\n\
         ) -> Result<Stage3ExecutionArtifacts<Fr>, Stage3KernelError>\n\
         where\n\
         \x20   E: Stage3KernelExecutor<Fr>,\n\
         \x20   T: Transcript<Challenge = Fr>,\n\
         {\n\
         \x20   execute_stage3_prover_with_program(&STAGE3_PROGRAM, executor, transcript)\n\
         }\n\
         \n\
         pub fn execute_stage3_prover_with_program<E, T>(\n\
         \x20   program: &'static Stage3CpuProgramPlan,\n\
         \x20   executor: &mut E,\n\
         \x20   transcript: &mut T,\n\
         ) -> Result<Stage3ExecutionArtifacts<Fr>, Stage3KernelError>\n\
         where\n\
         \x20   E: Stage3KernelExecutor<Fr>,\n\
         \x20   T: Transcript<Challenge = Fr>,\n\
         {\n\
         \x20   execute_stage3_program(program, Stage3ExecutionMode::Prover, executor, transcript)\n\
         }\n"
    }

    fn emit_verifier_entrypoint() -> &'static str {
        r#"pub fn verify_stage3<T>(
    proof: &Stage3Proof<Fr>,
    opening_inputs: &[Stage3OpeningInputValue<Fr>],
    transcript: &mut T,
) -> Result<Stage3ExecutionArtifacts<Fr>, VerifyStage3Error>
where
    T: Transcript<Challenge = Fr>,
{
    verify_stage3_with_program(&STAGE3_PROGRAM, proof, opening_inputs, transcript)
}

pub fn verify_stage3_with_program<T>(
    program: &'static Stage3VerifierProgramPlan,
    proof: &Stage3Proof<Fr>,
    opening_inputs: &[Stage3OpeningInputValue<Fr>],
    transcript: &mut T,
) -> Result<Stage3ExecutionArtifacts<Fr>, VerifyStage3Error>
where
    T: Transcript<Challenge = Fr>,
{
    if proof.sumchecks.len() != program.drivers.len() {
        return Err(VerifyStage3Error::UnexpectedProofCount {
            expected: program.drivers.len(),
            got: proof.sumchecks.len(),
        });
    }
    let mut store =
        bolt_verifier_runtime::ValueStore::with_opening_inputs(opening_inputs, program.opening_inputs)?;
    store.seed_constants(program.field_constants);
    let mut artifacts = Stage3ExecutionArtifacts::default();
    for step in program.steps {
        match step.kind {
            Stage3ProgramStepKind::TranscriptSqueeze => {
                let squeeze =
                    find_plan(program.transcript_squeezes, step.symbol).ok_or(VerifyStage3Error::MissingValue {
                        symbol: step.symbol,
                    })?;
                verify_stage3_squeeze(program, squeeze, &mut store, transcript, &mut artifacts)?;
            }
            Stage3ProgramStepKind::SumcheckDriver => {
                let driver =
                    find_plan(program.drivers, step.symbol).ok_or(VerifyStage3Error::MissingProof {
                        driver: step.symbol,
                    })?;
                verify_stage3_driver(program, driver, proof, &mut store, transcript, &mut artifacts)?;
            }
            Stage3ProgramStepKind::TranscriptAbsorbBytes => {
                return Err(VerifyStage3Error::InvalidProof {
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

pub fn stage3_verifier_program() -> &'static Stage3VerifierProgramPlan {
    &STAGE3_PROGRAM
}

fn verify_stage3_squeeze<T>(
    program: &'static Stage3VerifierProgramPlan,
    squeeze: &'static Stage3TranscriptSqueezePlan,
    store: &mut bolt_verifier_runtime::ValueStore<Fr>,
    transcript: &mut T,
    artifacts: &mut Stage3ExecutionArtifacts<Fr>,
) -> Result<(), VerifyStage3Error>
where
    T: Transcript<Challenge = Fr>,
{
    let values = transcript.challenge_vector(squeeze.count);
    store.observe_challenge_vector(squeeze, &values, |input, expected, actual| {
        VerifyStage3Error::InvalidInputLength {
            input,
            expected,
            actual,
        }
    })?;
    store
        .evaluate_available_exprs(program.field_exprs, program.scalar_exprs)
        .map_err(VerifyStage3Error::from)?;
    artifacts.challenge_vectors.push(Stage3ChallengeVector {
        symbol: squeeze.symbol,
        values,
    });
    Ok(())
}

fn verify_stage3_driver<T>(
    program: &'static Stage3VerifierProgramPlan,
    driver: &'static Stage3SumcheckDriverPlan,
    proof: &Stage3Proof<Fr>,
    store: &mut bolt_verifier_runtime::ValueStore<Fr>,
    transcript: &mut T,
    artifacts: &mut Stage3ExecutionArtifacts<Fr>,
) -> Result<(), VerifyStage3Error>
where
    T: Transcript<Challenge = Fr>,
{
    let proof = proof
        .sumchecks
        .get(artifacts.sumchecks.len())
        .ok_or(VerifyStage3Error::MissingProof {
            driver: driver.symbol,
        })?;
    let Some(relation) = driver.relation else {
        return Err(VerifyStage3Error::InvalidProof {
            driver: driver.symbol,
            reason: "missing driver relation",
        });
    };
    let output = match relation {
        Stage3RelationKind::Stage3Batched => {
            verify_batched_stage3(program, driver, proof, store, transcript)?
        }
        relation => return Err(VerifyStage3Error::UnsupportedRelation { relation }),
    };
    artifacts.sumchecks.push(output);
    Ok(())
}

fn verify_batched_stage3<T>(
    program: &'static Stage3VerifierProgramPlan,
    driver: &'static Stage3SumcheckDriverPlan,
    proof: &Stage3SumcheckOutput<Fr>,
    store: &mut bolt_verifier_runtime::ValueStore<Fr>,
    transcript: &mut T,
) -> Result<Stage3SumcheckOutput<Fr>, VerifyStage3Error>
where
    T: Transcript<Challenge = Fr>,
{
    store.evaluate_available_points(
        program.point_exprs,
        |input, expected, actual| VerifyStage3Error::InvalidInputLength {
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
            bolt_verifier_runtime::evaluate_relation_output_batch(
                driver,
                program.batches,
                program.claims,
                program.instance_results,
                program.relation_outputs,
                program.field_exprs,
                program.scalar_exprs,
                &[],
                store,
                evals,
                point,
                batching_coeffs,
                |_, _, _| Ok::<_, VerifyStage3Error>(bolt_verifier_runtime::RelationOutputInputs::empty()),
            )
        },
        |store, verified| observe_stage3_sumcheck_output(program, store, verified),
        |driver, error| VerifyStage3Error::Sumcheck { driver, error },
    )
}

fn observe_stage3_sumcheck_output<F: Field>(
    program: &'static Stage3VerifierProgramPlan,
    store: &mut bolt_verifier_runtime::ValueStore<F>,
    output: &Stage3SumcheckOutput<F>,
) -> Result<(), VerifyStage3Error> {
    store.observe_sumcheck_output(
        program.instance_results,
        program.evals,
        output,
        |instance, mut point| {
            match instance.point_order {
                bolt_verifier_runtime::SumcheckPointOrder::AsIs => {}
                bolt_verifier_runtime::SumcheckPointOrder::Reverse => point.reverse(),
                _ => {
                    return Err(VerifyStage3Error::InvalidProof {
                        driver: output.driver,
                        reason: "unsupported point order",
                    });
                }
            }
            Ok(point)
        },
        |input, expected, actual| VerifyStage3Error::InvalidInputLength {
            input,
            expected,
            actual,
        },
        |symbol| VerifyStage3Error::MissingValue { symbol },
    )?;
    store.evaluate_available_points(
        program.point_exprs,
        |input, expected, actual| VerifyStage3Error::InvalidInputLength {
            input,
            expected,
            actual,
        },
    )?;
    store
        .evaluate_available_exprs(program.field_exprs, program.scalar_exprs)
        .map_err(VerifyStage3Error::from)?;
    store.verify_opening_equalities(
        program.opening_equalities,
        |driver, reason| VerifyStage3Error::InvalidProof { driver, reason },
        |symbol| VerifyStage3Error::MissingValue { symbol },
    )
}

"#
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

fn verify_count(kind: &str, symbol: &str, expected: usize, actual: usize) -> Result<(), EmitError> {
    if expected == actual {
        Ok(())
    } else {
        Err(EmitError::new(format!(
            "{kind} @{symbol} count mismatch: expected {expected}, got {actual}"
        )))
    }
}

fn missing_role_binding(kind: &str, symbol: &str) -> EmitError {
    EmitError::new(format!("missing {kind} for `{symbol}`"))
}

fn symbols<'a>(values: impl Iterator<Item = &'a String>) -> BTreeSet<String> {
    values.cloned().collect()
}
