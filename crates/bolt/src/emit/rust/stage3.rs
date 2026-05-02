use std::collections::{BTreeMap, BTreeSet};

use melior::ir::block::BlockLike;
use melior::ir::operation::{OperationLike, OperationResult};
use melior::ir::{Attribute, OperationRef};

use crate::emit::rust::{push_format, EmitError, RustSourceFile};
use crate::ir::{string_attribute_value, symbol_attribute_value, BoltModule, Cpu, Role};
use crate::schema::verify_cpu_schema;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3CpuProgram {
    pub role: Role,
    pub params: Stage3Params,
    pub steps: Vec<Stage3ProgramStepPlan>,
    pub transcript_squeezes: Vec<Stage3TranscriptSqueezePlan>,
    pub opening_inputs: Vec<Stage3OpeningInputPlan>,
    pub field_constants: Vec<Stage3FieldConstantPlan>,
    pub field_exprs: Vec<Stage3FieldExprPlan>,
    pub kernels: Vec<Stage3KernelPlan>,
    pub claims: Vec<Stage3SumcheckClaimPlan>,
    pub batches: Vec<Stage3SumcheckBatchPlan>,
    pub drivers: Vec<Stage3SumcheckDriverPlan>,
    pub instance_results: Vec<Stage3SumcheckInstanceResultPlan>,
    pub evals: Vec<Stage3SumcheckEvalPlan>,
    pub point_slices: Vec<Stage3PointSlicePlan>,
    pub point_concats: Vec<Stage3PointConcatPlan>,
    pub opening_claims: Vec<Stage3OpeningClaimPlan>,
    pub opening_equalities: Vec<Stage3OpeningClaimEqualityPlan>,
    pub opening_batches: Vec<Stage3OpeningBatchPlan>,
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3TranscriptSqueezePlan {
    pub symbol: String,
    pub label: String,
    pub kind: String,
    pub count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3ProgramStepPlan {
    pub kind: String,
    pub symbol: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3OpeningInputPlan {
    pub symbol: String,
    pub source_stage: String,
    pub source_claim: String,
    pub oracle: String,
    pub domain: String,
    pub point_arity: usize,
    pub claim_kind: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3FieldConstantPlan {
    pub symbol: String,
    pub field: String,
    pub value: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3FieldExprPlan {
    pub symbol: String,
    pub kind: String,
    pub formula: String,
    pub operand_names: Vec<String>,
    pub operands: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3SumcheckClaimPlan {
    pub symbol: String,
    pub stage: String,
    pub domain: String,
    pub num_rounds: usize,
    pub degree: usize,
    pub claim: String,
    pub kernel: Option<String>,
    pub relation: Option<String>,
    pub claim_value: String,
    pub input_openings: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3SumcheckBatchPlan {
    pub symbol: String,
    pub stage: String,
    pub proof_slot: String,
    pub policy: String,
    pub count: usize,
    pub ordered_claims: Vec<String>,
    pub claim_operands: Vec<String>,
    pub claim_label: String,
    pub round_label: String,
    pub round_schedule: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3SumcheckDriverPlan {
    pub symbol: String,
    pub stage: String,
    pub proof_slot: String,
    pub kernel: Option<String>,
    pub relation: Option<String>,
    pub batch: String,
    pub policy: String,
    pub round_schedule: Vec<usize>,
    pub claim_label: String,
    pub round_label: String,
    pub num_rounds: usize,
    pub degree: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3SumcheckInstanceResultPlan {
    pub symbol: String,
    pub source: String,
    pub claim: String,
    pub relation: String,
    pub index: usize,
    pub point_arity: usize,
    pub num_rounds: usize,
    pub round_offset: usize,
    pub point_order: String,
    pub degree: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3SumcheckEvalPlan {
    pub symbol: String,
    pub source: String,
    pub name: String,
    pub index: usize,
    pub oracle: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3PointSlicePlan {
    pub symbol: String,
    pub source: String,
    pub offset: usize,
    pub length: usize,
    pub input: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3PointConcatPlan {
    pub symbol: String,
    pub layout: String,
    pub arity: usize,
    pub inputs: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3OpeningClaimPlan {
    pub symbol: String,
    pub oracle: String,
    pub domain: String,
    pub point_arity: usize,
    pub claim_kind: String,
    pub point_source: String,
    pub eval_source: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3OpeningClaimEqualityPlan {
    pub symbol: String,
    pub mode: String,
    pub lhs: String,
    pub rhs: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3OpeningBatchPlan {
    pub symbol: String,
    pub stage: String,
    pub proof_slot: String,
    pub policy: String,
    pub count: usize,
    pub ordered_claims: Vec<String>,
    pub claim_operands: Vec<String>,
}

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
        let mut kernels = Vec::new();
        let mut claims = Vec::new();
        let mut batches = Vec::new();
        let mut drivers = Vec::new();
        let mut instance_results = Vec::new();
        let mut evals = Vec::new();
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
                    let symbol = string_attr(op, "sym_name")?;
                    steps.push(Stage3ProgramStepPlan {
                        kind: "transcript_squeeze".to_owned(),
                        symbol: symbol.clone(),
                    });
                    transcript_squeezes.push(Stage3TranscriptSqueezePlan {
                        symbol,
                        label: string_attr(op, "label")?,
                        kind: string_attr(op, "kind")?,
                        count: int_attr(op, "count")?,
                    });
                }
                "cpu.opening_input" => {
                    opening_inputs.push(Stage3OpeningInputPlan {
                        symbol: string_attr(op, "sym_name")?,
                        source_stage: symbol_attr(op, "source_stage")?,
                        source_claim: symbol_attr(op, "source_claim")?,
                        oracle: symbol_attr(op, "oracle")?,
                        domain: symbol_attr(op, "domain")?,
                        point_arity: int_attr(op, "point_arity")?,
                        claim_kind: string_attr(op, "claim_kind")?,
                    });
                }
                "cpu.field_const" => {
                    field_constants.push(Stage3FieldConstantPlan {
                        symbol: string_attr(op, "sym_name")?,
                        field: symbol_attr(op, "field")?,
                        value: int_attr(op, "value")?,
                    });
                }
                "cpu.field_zero" => {
                    field_constants.push(Stage3FieldConstantPlan {
                        symbol: string_attr(op, "sym_name")?,
                        field: symbol_attr(op, "field")?,
                        value: 0,
                    });
                }
                "cpu.field_one" => {
                    field_constants.push(Stage3FieldConstantPlan {
                        symbol: string_attr(op, "sym_name")?,
                        field: symbol_attr(op, "field")?,
                        value: 1,
                    });
                }
                "cpu.field_add" | "cpu.field_sub" | "cpu.field_mul" | "cpu.field_neg" => {
                    let operands = operand_symbols(op, 0)?;
                    field_exprs.push(Stage3FieldExprPlan {
                        symbol: string_attr(op, "sym_name")?,
                        kind: "op".to_owned(),
                        formula: operation_name(op).replace("cpu.field_", "field."),
                        operand_names: operands.clone(),
                        operands,
                    });
                }
                "cpu.field_pow" => {
                    let exponent = int_attr(op, "exponent")?;
                    let operands = operand_symbols(op, 0)?;
                    field_exprs.push(Stage3FieldExprPlan {
                        symbol: string_attr(op, "sym_name")?,
                        kind: "op".to_owned(),
                        formula: format!("field.pow:{exponent}"),
                        operand_names: operands.clone(),
                        operands,
                    });
                }
                "cpu.poly_lagrange_basis_eval" => {
                    let domain_start = signed_int_attr(op, "domain_start")?;
                    let domain_size = int_attr(op, "domain_size")?;
                    let index = int_attr(op, "index")?;
                    let operands = operand_symbols(op, 0)?;
                    field_exprs.push(Stage3FieldExprPlan {
                        symbol: string_attr(op, "sym_name")?,
                        kind: "op".to_owned(),
                        formula: format!(
                            "poly.lagrange_basis_eval:{domain_start}:{domain_size}:{index}"
                        ),
                        operand_names: operands.clone(),
                        operands,
                    });
                }
                "cpu.sumcheck_claim" => {
                    claims.push(Stage3SumcheckClaimPlan {
                        symbol: string_attr(op, "sym_name")?,
                        stage: symbol_attr(op, "stage")?,
                        domain: symbol_attr(op, "domain")?,
                        num_rounds: int_attr(op, "num_rounds")?,
                        degree: int_attr(op, "degree")?,
                        claim: symbol_attr(op, "claim")?,
                        kernel: Some(symbol_attr(op, "kernel")?),
                        relation: None,
                        claim_value: operand_symbol(op, 0)?,
                        input_openings: operand_symbols(op, 1)?,
                    });
                }
                "cpu.sumcheck_verify_claim" => {
                    claims.push(Stage3SumcheckClaimPlan {
                        symbol: string_attr(op, "sym_name")?,
                        stage: symbol_attr(op, "stage")?,
                        domain: symbol_attr(op, "domain")?,
                        num_rounds: int_attr(op, "num_rounds")?,
                        degree: int_attr(op, "degree")?,
                        claim: symbol_attr(op, "claim")?,
                        kernel: None,
                        relation: Some(symbol_attr(op, "relation")?),
                        claim_value: operand_symbol(op, 0)?,
                        input_openings: operand_symbols(op, 1)?,
                    });
                }
                "cpu.sumcheck_batch" => {
                    batches.push(Stage3SumcheckBatchPlan {
                        symbol: string_attr(op, "sym_name")?,
                        stage: symbol_attr(op, "stage")?,
                        proof_slot: symbol_attr(op, "proof_slot")?,
                        policy: string_attr(op, "policy")?,
                        count: int_attr(op, "count")?,
                        ordered_claims: symbol_array_attr(op, "ordered_claims")?,
                        claim_operands: operand_symbols(op, 0)?,
                        claim_label: string_attr(op, "claim_label")?,
                        round_label: string_attr(op, "round_label")?,
                        round_schedule: int_array_attr(op, "round_schedule")?,
                    });
                }
                "cpu.sumcheck_driver" => {
                    let symbol = string_attr(op, "sym_name")?;
                    steps.push(Stage3ProgramStepPlan {
                        kind: "sumcheck_driver".to_owned(),
                        symbol: symbol.clone(),
                    });
                    drivers.push(Stage3SumcheckDriverPlan {
                        symbol,
                        stage: symbol_attr(op, "stage")?,
                        proof_slot: symbol_attr(op, "proof_slot")?,
                        kernel: Some(symbol_attr(op, "kernel")?),
                        relation: None,
                        batch: operand_symbol(op, 1)?,
                        policy: string_attr(op, "policy")?,
                        round_schedule: int_array_attr(op, "round_schedule")?,
                        claim_label: string_attr(op, "claim_label")?,
                        round_label: string_attr(op, "round_label")?,
                        num_rounds: int_attr(op, "num_rounds")?,
                        degree: int_attr(op, "degree")?,
                    });
                }
                "cpu.sumcheck_verify" => {
                    let symbol = string_attr(op, "sym_name")?;
                    steps.push(Stage3ProgramStepPlan {
                        kind: "sumcheck_driver".to_owned(),
                        symbol: symbol.clone(),
                    });
                    drivers.push(Stage3SumcheckDriverPlan {
                        symbol,
                        stage: symbol_attr(op, "stage")?,
                        proof_slot: symbol_attr(op, "proof_slot")?,
                        kernel: None,
                        relation: Some(symbol_attr(op, "relation")?),
                        batch: operand_symbol(op, 1)?,
                        policy: string_attr(op, "policy")?,
                        round_schedule: int_array_attr(op, "round_schedule")?,
                        claim_label: string_attr(op, "claim_label")?,
                        round_label: string_attr(op, "round_label")?,
                        num_rounds: int_attr(op, "num_rounds")?,
                        degree: int_attr(op, "degree")?,
                    });
                }
                "cpu.sumcheck_instance_result" => {
                    instance_results.push(Stage3SumcheckInstanceResultPlan {
                        symbol: string_attr(op, "sym_name")?,
                        source: symbol_attr(op, "source")?,
                        claim: symbol_attr(op, "claim")?,
                        relation: symbol_attr(op, "relation")?,
                        index: int_attr(op, "index")?,
                        point_arity: int_attr(op, "point_arity")?,
                        num_rounds: int_attr(op, "num_rounds")?,
                        round_offset: int_attr(op, "round_offset")?,
                        point_order: string_attr(op, "point_order")?,
                        degree: int_attr(op, "degree")?,
                    });
                }
                "cpu.sumcheck_eval" => {
                    evals.push(Stage3SumcheckEvalPlan {
                        symbol: string_attr(op, "sym_name")?,
                        source: symbol_attr(op, "source")?,
                        name: symbol_attr(op, "name")?,
                        index: int_attr(op, "index")?,
                        oracle: symbol_attr(op, "oracle")?,
                    });
                }
                "cpu.point_slice" => {
                    point_slices.push(Stage3PointSlicePlan {
                        symbol: string_attr(op, "sym_name")?,
                        source: symbol_attr(op, "source")?,
                        offset: int_attr(op, "offset")?,
                        length: int_attr(op, "length")?,
                        input: operand_symbol(op, 0)?,
                    });
                }
                "cpu.point_concat" => {
                    point_concats.push(Stage3PointConcatPlan {
                        symbol: string_attr(op, "sym_name")?,
                        layout: string_attr(op, "layout")?,
                        arity: int_attr(op, "arity")?,
                        inputs: operand_symbols(op, 0)?,
                    });
                }
                "cpu.opening_claim" => {
                    opening_claims.push(Stage3OpeningClaimPlan {
                        symbol: string_attr(op, "sym_name")?,
                        oracle: symbol_attr(op, "oracle")?,
                        domain: symbol_attr(op, "domain")?,
                        point_arity: int_attr(op, "point_arity")?,
                        claim_kind: string_attr(op, "claim_kind")?,
                        point_source: operand_symbol(op, 0)?,
                        eval_source: operand_symbol(op, 1)?,
                    });
                }
                "cpu.opening_claim_equal" => {
                    opening_equalities.push(Stage3OpeningClaimEqualityPlan {
                        symbol: string_attr(op, "sym_name")?,
                        mode: string_attr(op, "mode")?,
                        lhs: operand_symbol(op, 0)?,
                        rhs: operand_symbol(op, 1)?,
                    });
                }
                "cpu.opening_batch" => {
                    opening_batches.push(Stage3OpeningBatchPlan {
                        symbol: string_attr(op, "sym_name")?,
                        stage: symbol_attr(op, "stage")?,
                        proof_slot: symbol_attr(op, "proof_slot")?,
                        policy: string_attr(op, "policy")?,
                        count: int_attr(op, "count")?,
                        ordered_claims: symbol_array_attr(op, "ordered_claims")?,
                        claim_operands: operand_symbols(op, 0)?,
                    });
                }
                _ => {}
            }
        }

        Ok(Self {
            params: params.ok_or_else(|| EmitError::new("missing cpu.params"))?,
            role: module
                .role()
                .ok_or_else(|| EmitError::new("missing cpu party role"))?,
            steps,
            transcript_squeezes,
            opening_inputs,
            field_constants,
            field_exprs,
            kernels,
            claims,
            batches,
            drivers,
            instance_results,
            evals,
            point_slices,
            point_concats,
            opening_claims,
            opening_equalities,
            opening_batches,
        })
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
        let field_values = self.field_value_symbols();
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

    fn field_value_symbols(&self) -> BTreeSet<String> {
        let mut values = symbols(self.opening_inputs.iter().map(|input| &input.symbol));
        values.extend(symbols(
            self.field_constants.iter().map(|constant| &constant.symbol),
        ));
        values.extend(symbols(
            self.transcript_squeezes
                .iter()
                .filter(|squeeze| matches!(squeeze.kind.as_str(), "challenge_scalar" | "scalar"))
                .map(|squeeze| &squeeze.symbol),
        ));
        values.extend(symbols(self.field_exprs.iter().map(|expr| &expr.symbol)));
        values.extend(symbols(self.evals.iter().map(|eval| &eval.symbol)));
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

    fn verify_opening_flow(&self) -> Result<(), EmitError> {
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
        let eval_sources = self.field_value_symbols();
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
         use jolt_transcript::Blake2bTranscript;"
    }

    fn emit_prover_types() -> &'static str {
        "pub type DefaultStage3Transcript = Blake2bTranscript<Fr>;\n"
    }

    fn emit_verifier_imports() -> &'static str {
        "use jolt_field::{Field, Fr};\n\
         use jolt_poly::{EqPlusOnePolynomial, EqPolynomial};\n\
         use jolt_sumcheck::{CompressedLabeledRoundPoly, SumcheckClaim, SumcheckError, SumcheckProof, SumcheckVerifier};\n\
         use jolt_transcript::{Blake2bTranscript, Label, Transcript};"
    }

    fn emit_verifier_types() -> &'static str {
        r"pub type DefaultStage3Transcript = Blake2bTranscript<Fr>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage3Params {
    pub field: &'static str,
    pub pcs: &'static str,
    pub transcript: &'static str,
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
    pub relation: &'static str,
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
    pub relation: &'static str,
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
pub struct Stage3VerifierProgramPlan {
    pub params: Stage3Params,
    pub steps: &'static [Stage3ProgramStepPlan],
    pub transcript_squeezes: &'static [Stage3TranscriptSqueezePlan],
    pub opening_inputs: &'static [Stage3OpeningInputPlan],
    pub field_constants: &'static [Stage3FieldConstantPlan],
    pub field_exprs: &'static [Stage3FieldExprPlan],
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

#[derive(Clone, Debug)]
pub struct Stage3OpeningInputValue<F: Field> {
    pub symbol: &'static str,
    pub point: Vec<F>,
    pub eval: F,
}

#[derive(Clone, Debug, Default)]
struct Stage3ValueStore<F: Field> {
    scalars: Vec<(&'static str, F)>,
    points: Vec<(&'static str, Vec<F>)>,
}

#[derive(Debug)]
pub enum VerifyStage3Error {
    UnexpectedProofCount { expected: usize, got: usize },
    MissingProof { driver: &'static str },
    MissingBatch { driver: &'static str, batch: &'static str },
    MissingClaim { batch: &'static str, claim: &'static str },
    MissingValue { symbol: &'static str },
    InvalidInputLength { input: &'static str, expected: usize, actual: usize },
    InvalidProof { driver: &'static str, reason: &'static str },
    UnsupportedFieldExpr { symbol: &'static str, formula: &'static str },
    UnsupportedRelation { relation: &'static str },
    Sumcheck { driver: &'static str, error: SumcheckError<Fr> },
}
"
    }

    fn emit_prover_constants(&self) -> Result<String, EmitError> {
        let mut source = self.emit_shared_constants();
        source.push_str(&self.emit_kernel_constants());
        source.push_str(&self.emit_prover_sumcheck_claim_constants()?);
        source.push_str(&self.emit_sumcheck_batch_constants());
        source.push_str(&self.emit_prover_sumcheck_driver_constants()?);
        source.push_str(&self.emit_tail_constants());
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
        let mut source = self.emit_shared_constants();
        source.push_str(&self.emit_verifier_sumcheck_claim_constants()?);
        source.push_str(&self.emit_sumcheck_batch_constants());
        source.push_str(&self.emit_verifier_sumcheck_driver_constants()?);
        source.push_str(&self.emit_tail_constants());
        source.push_str(
            "pub const STAGE3_PROGRAM: Stage3VerifierProgramPlan = Stage3VerifierProgramPlan {\n\
             \x20   params: STAGE3_PARAMS,\n\
             \x20   steps: STAGE3_PROGRAM_STEPS,\n\
             \x20   transcript_squeezes: STAGE3_TRANSCRIPT_SQUEEZES,\n\
             \x20   opening_inputs: STAGE3_OPENING_INPUTS,\n\
             \x20   field_constants: STAGE3_FIELD_CONSTANTS,\n\
             \x20   field_exprs: STAGE3_FIELD_EXPRS,\n\
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

    fn emit_shared_constants(&self) -> String {
        let mut source = String::new();
        push_format(
            &mut source,
            format_args!(
                "pub const STAGE3_PARAMS: Stage3Params = Stage3Params {{\n\
             \x20   field: {},\n\
             \x20   pcs: {},\n\
             \x20   transcript: {},\n\
             }};\n",
                rust_str(&self.params.field),
                rust_str(&self.params.pcs),
                rust_str(&self.params.transcript)
            ),
        );
        source.push_str(&self.emit_program_step_constants());
        source.push_str(&self.emit_transcript_squeeze_constants());
        source.push_str(&self.emit_opening_input_constants());
        source.push_str(&self.emit_field_constant_constants());
        source.push_str(&self.emit_field_expr_constants());
        source
    }

    fn emit_program_step_constants(&self) -> String {
        let steps = self
            .steps
            .iter()
            .map(|step| {
                format!(
                    "    Stage3ProgramStepPlan {{ kind: {}, symbol: {} }},",
                    rust_str(&step.kind),
                    rust_str(&step.symbol),
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!("pub const STAGE3_PROGRAM_STEPS: &[Stage3ProgramStepPlan] = &[\n{steps}\n];\n\n")
    }

    fn emit_transcript_squeeze_constants(&self) -> String {
        let squeezes = self
            .transcript_squeezes
            .iter()
            .map(|squeeze| {
                format!(
                    "    Stage3TranscriptSqueezePlan {{ symbol: {}, label: {}, kind: {}, count: {} }},",
                    rust_str(&squeeze.symbol),
                    rust_str(&squeeze.label),
                    rust_str(&squeeze.kind),
                    squeeze.count,
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "pub const STAGE3_TRANSCRIPT_SQUEEZES: &[Stage3TranscriptSqueezePlan] = &[\n{squeezes}\n];\n\n"
        )
    }

    fn emit_opening_input_constants(&self) -> String {
        let inputs = self
            .opening_inputs
            .iter()
            .map(|input| {
                format!(
                    "    Stage3OpeningInputPlan {{ symbol: {}, source_stage: {}, source_claim: {}, oracle: {}, domain: {}, point_arity: {}, claim_kind: {} }},",
                    rust_str(&input.symbol),
                    rust_str(&input.source_stage),
                    rust_str(&input.source_claim),
                    rust_str(&input.oracle),
                    rust_str(&input.domain),
                    input.point_arity,
                    rust_str(&input.claim_kind)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!("pub const STAGE3_OPENING_INPUTS: &[Stage3OpeningInputPlan] = &[\n{inputs}\n];\n\n")
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

    fn emit_field_expr_constants(&self) -> String {
        let mut source = String::new();
        for (index, expr) in self.field_exprs.iter().enumerate() {
            source.push_str(&emit_str_array(
                &format!("STAGE3_FIELD_EXPR_{index}_OPERAND_NAMES"),
                &expr.operand_names,
            ));
            source.push_str(&emit_str_array(
                &format!("STAGE3_FIELD_EXPR_{index}_OPERANDS"),
                &expr.operands,
            ));
        }
        let exprs = self
            .field_exprs
            .iter()
            .enumerate()
            .map(|(index, expr)| {
                format!(
                    "    Stage3FieldExprPlan {{ symbol: {}, kind: {}, formula: {}, operand_names: STAGE3_FIELD_EXPR_{index}_OPERAND_NAMES, operands: STAGE3_FIELD_EXPR_{index}_OPERANDS }},",
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
        source
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
        let mut source = String::new();
        for (index, claim) in self.claims.iter().enumerate() {
            source.push_str(&emit_str_array(
                &format!("STAGE3_SUMCHECK_CLAIM_{index}_INPUT_OPENINGS"),
                &claim.input_openings,
            ));
        }
        let mut claims = Vec::new();
        for (index, claim) in self.claims.iter().enumerate() {
            if prover {
                let kernel = claim
                    .kernel
                    .as_deref()
                    .ok_or_else(|| missing_role_binding("prover claim kernel", &claim.symbol))?;
                claims.push(format!(
                        "    Stage3SumcheckClaimPlan {{ symbol: {}, stage: {}, domain: {}, num_rounds: {}, degree: {}, claim: {}, kernel: {}, claim_value: {}, input_openings: STAGE3_SUMCHECK_CLAIM_{index}_INPUT_OPENINGS }},",
                        rust_str(&claim.symbol),
                        rust_str(&claim.stage),
                        rust_str(&claim.domain),
                        claim.num_rounds,
                        claim.degree,
                        rust_str(&claim.claim),
                        rust_str(kernel),
                        rust_str(&claim.claim_value)
                    ));
            } else {
                let relation = claim.relation.as_deref().ok_or_else(|| {
                    missing_role_binding("verifier claim relation", &claim.symbol)
                })?;
                claims.push(format!(
                        "    Stage3SumcheckClaimPlan {{ symbol: {}, stage: {}, domain: {}, num_rounds: {}, degree: {}, claim: {}, relation: {}, claim_value: {}, input_openings: STAGE3_SUMCHECK_CLAIM_{index}_INPUT_OPENINGS }},",
                        rust_str(&claim.symbol),
                        rust_str(&claim.stage),
                        rust_str(&claim.domain),
                        claim.num_rounds,
                        claim.degree,
                        rust_str(&claim.claim),
                        rust_str(relation),
                        rust_str(&claim.claim_value)
                    ));
            }
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
        let mut source = String::new();
        for (index, driver) in self.drivers.iter().enumerate() {
            source.push_str(&emit_usize_array(
                &format!("STAGE3_SUMCHECK_DRIVER_{index}_ROUND_SCHEDULE"),
                &driver.round_schedule,
            ));
        }
        let mut drivers = Vec::new();
        for (index, driver) in self.drivers.iter().enumerate() {
            if prover {
                let kernel = driver
                    .kernel
                    .as_deref()
                    .ok_or_else(|| missing_role_binding("prover driver kernel", &driver.symbol))?;
                drivers.push(format!(
                        "    Stage3SumcheckDriverPlan {{ symbol: {}, stage: {}, proof_slot: {}, kernel: {}, batch: {}, policy: {}, round_schedule: STAGE3_SUMCHECK_DRIVER_{index}_ROUND_SCHEDULE, claim_label: {}, round_label: {}, num_rounds: {}, degree: {} }},",
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
            } else {
                let relation = driver.relation.as_deref().ok_or_else(|| {
                    missing_role_binding("verifier driver relation", &driver.symbol)
                })?;
                drivers.push(format!(
                        "    Stage3SumcheckDriverPlan {{ symbol: {}, stage: {}, proof_slot: {}, relation: {}, batch: {}, policy: {}, round_schedule: STAGE3_SUMCHECK_DRIVER_{index}_ROUND_SCHEDULE, claim_label: {}, round_label: {}, num_rounds: {}, degree: {} }},",
                        rust_str(&driver.symbol),
                        rust_str(&driver.stage),
                        rust_str(&driver.proof_slot),
                        rust_str(relation),
                        rust_str(&driver.batch),
                        rust_str(&driver.policy),
                        rust_str(&driver.claim_label),
                        rust_str(&driver.round_label),
                        driver.num_rounds,
                        driver.degree
                    ));
            }
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

    fn emit_tail_constants(&self) -> String {
        let mut source = String::new();
        source.push_str(&self.emit_sumcheck_instance_result_constants());
        source.push_str(&self.emit_sumcheck_eval_constants());
        source.push_str(&self.emit_point_slice_constants());
        source.push_str(&self.emit_point_concat_constants());
        source.push_str(&self.emit_opening_claim_constants());
        source.push_str(&self.emit_opening_claim_equality_constants());
        source.push_str(&self.emit_opening_batch_constants());
        source
    }

    fn emit_sumcheck_instance_result_constants(&self) -> String {
        let instances = self
            .instance_results
            .iter()
            .map(|instance| {
                format!(
                    "    Stage3SumcheckInstanceResultPlan {{ symbol: {}, source: {}, claim: {}, relation: {}, index: {}, point_arity: {}, num_rounds: {}, round_offset: {}, point_order: {}, degree: {} }},",
                    rust_str(&instance.symbol),
                    rust_str(&instance.source),
                    rust_str(&instance.claim),
                    rust_str(&instance.relation),
                    instance.index,
                    instance.point_arity,
                    instance.num_rounds,
                    instance.round_offset,
                    rust_str(&instance.point_order),
                    instance.degree
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "pub const STAGE3_SUMCHECK_INSTANCE_RESULTS: &[Stage3SumcheckInstanceResultPlan] = &[\n{instances}\n];\n\n"
        )
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

    fn emit_point_slice_constants(&self) -> String {
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
        format!("pub const STAGE3_POINT_SLICES: &[Stage3PointSlicePlan] = &[\n{slices}\n];\n\n")
    }

    fn emit_point_concat_constants(&self) -> String {
        let mut source = String::new();
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
        source
    }

    fn emit_opening_claim_constants(&self) -> String {
        let claims = self
            .opening_claims
            .iter()
            .map(|claim| {
                format!(
                    "    Stage3OpeningClaimPlan {{ symbol: {}, oracle: {}, domain: {}, point_arity: {}, claim_kind: {}, point_source: {}, eval_source: {} }},",
                    rust_str(&claim.symbol),
                    rust_str(&claim.oracle),
                    rust_str(&claim.domain),
                    claim.point_arity,
                    rust_str(&claim.claim_kind),
                    rust_str(&claim.point_source),
                    rust_str(&claim.eval_source)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!("pub const STAGE3_OPENING_CLAIMS: &[Stage3OpeningClaimPlan] = &[\n{claims}\n];\n\n")
    }

    fn emit_opening_claim_equality_constants(&self) -> String {
        let equalities = self
            .opening_equalities
            .iter()
            .map(|equality| {
                format!(
                    "    Stage3OpeningClaimEqualityPlan {{ symbol: {}, mode: {}, lhs: {}, rhs: {} }},",
                    rust_str(&equality.symbol),
                    rust_str(&equality.mode),
                    rust_str(&equality.lhs),
                    rust_str(&equality.rhs)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "pub const STAGE3_OPENING_EQUALITIES: &[Stage3OpeningClaimEqualityPlan] = &[\n{equalities}\n];\n\n"
        )
    }

    fn emit_opening_batch_constants(&self) -> String {
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
        "pub fn execute_stage3_prover<E>(\n\
         \x20   executor: &mut E,\n\
         \x20   transcript: &mut DefaultStage3Transcript,\n\
         ) -> Result<Stage3ExecutionArtifacts<Fr>, Stage3KernelError>\n\
         where\n\
         \x20   E: Stage3KernelExecutor<Fr>,\n\
         {\n\
         \x20   execute_stage3_program(&STAGE3_PROGRAM, Stage3ExecutionMode::Prover, executor, transcript)\n\
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
    if proof.sumchecks.len() != STAGE3_PROGRAM.drivers.len() {
        return Err(VerifyStage3Error::UnexpectedProofCount {
            expected: STAGE3_PROGRAM.drivers.len(),
            got: proof.sumchecks.len(),
        });
    }
    let mut store = Stage3ValueStore::with_opening_inputs(opening_inputs);
    store.seed_constants();
    let mut artifacts = Stage3ExecutionArtifacts::default();
    for step in STAGE3_PROGRAM.steps {
        match step.kind {
            "transcript_squeeze" => {
                let squeeze = find_squeeze(step.symbol).ok_or(VerifyStage3Error::MissingValue {
                    symbol: step.symbol,
                })?;
                verify_stage3_squeeze(squeeze, &mut store, transcript, &mut artifacts)?;
            }
            "sumcheck_driver" => {
                let driver = find_driver(step.symbol).ok_or(VerifyStage3Error::MissingProof {
                    driver: step.symbol,
                })?;
                verify_stage3_driver(driver, proof, &mut store, transcript, &mut artifacts)?;
            }
            _ => {
                return Err(VerifyStage3Error::InvalidProof {
                    driver: step.symbol,
                    reason: "unsupported stage3 program step",
                });
            }
        }
    }
    artifacts
        .opening_batches
        .extend(STAGE3_PROGRAM.opening_batches.iter());
    Ok(artifacts)
}

pub fn stage3_verifier_program() -> &'static Stage3VerifierProgramPlan {
    &STAGE3_PROGRAM
}

fn verify_stage3_squeeze<T>(
    squeeze: &'static Stage3TranscriptSqueezePlan,
    store: &mut Stage3ValueStore<Fr>,
    transcript: &mut T,
    artifacts: &mut Stage3ExecutionArtifacts<Fr>,
) -> Result<(), VerifyStage3Error>
where
    T: Transcript<Challenge = Fr>,
{
    let values = transcript.challenge_vector(squeeze.count);
    store.observe_challenge_vector(squeeze, &values)?;
    artifacts.challenge_vectors.push(Stage3ChallengeVector {
        symbol: squeeze.symbol,
        values,
    });
    Ok(())
}

fn verify_stage3_driver<T>(
    driver: &'static Stage3SumcheckDriverPlan,
    proof: &Stage3Proof<Fr>,
    store: &mut Stage3ValueStore<Fr>,
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
    let output = match driver.relation {
        "jolt.stage3.batched" => verify_batched_stage3(driver, proof, store, transcript)?,
        relation => return Err(VerifyStage3Error::UnsupportedRelation { relation }),
    };
    artifacts.sumchecks.push(output);
    Ok(())
}

fn verify_batched_stage3<T>(
    driver: &'static Stage3SumcheckDriverPlan,
    proof: &Stage3SumcheckOutput<Fr>,
    store: &mut Stage3ValueStore<Fr>,
    transcript: &mut T,
) -> Result<Stage3SumcheckOutput<Fr>, VerifyStage3Error>
where
    T: Transcript<Challenge = Fr>,
{
    if proof.driver != driver.symbol {
        return Err(VerifyStage3Error::InvalidProof {
            driver: driver.symbol,
            reason: "driver symbol mismatch",
        });
    }
    let batch = find_batch(driver.batch)?;
    let claims = batch_claims(batch)?;
    let input_claims = store.batch_claim_values(batch)?;
    for claim in &input_claims {
        append_labeled_scalar(transcript, batch.claim_label, claim);
    }
    let batching_coeffs = transcript.challenge_vector(claims.len());
    let claimed_sum = input_claims
        .iter()
        .zip(claims.iter())
        .zip(&batching_coeffs)
        .map(|((claim, plan), coefficient)| {
            claim.mul_pow_2(driver.num_rounds - plan.num_rounds) * *coefficient
        })
        .sum::<Fr>();
    let claim = SumcheckClaim::new(driver.num_rounds, driver.degree, claimed_sum);
    let round_proofs = proof
        .proof
        .round_polynomials
        .iter()
        .map(|poly| CompressedLabeledRoundPoly::new(poly, driver.round_label.as_bytes()))
        .collect::<Vec<_>>();
    let output = SumcheckVerifier::verify(&claim, &round_proofs, transcript)
        .map_err(|error| VerifyStage3Error::Sumcheck {
            driver: driver.symbol,
            error,
        })?;
    if !proof.point.is_empty() && proof.point != output.point {
        return Err(VerifyStage3Error::InvalidProof {
            driver: driver.symbol,
            reason: "batched point mismatch",
        });
    }
    let expected =
        expected_batched_output_claim(driver, &*store, &proof.evals, &output.point, &batching_coeffs)?;
    if output.value != expected {
        return Err(VerifyStage3Error::InvalidProof {
            driver: driver.symbol,
            reason: "batched output claim mismatch",
        });
    }
    let verified = Stage3SumcheckOutput {
        driver: driver.symbol,
        point: output.point,
        evals: proof.evals.clone(),
        proof: proof.proof.clone(),
    };
    store.observe_sumcheck_output(&verified)?;
    append_opening_claims(store, transcript, &verified.evals)?;
    Ok(verified)
}

impl<F: Field> Stage3ValueStore<F> {
    fn with_opening_inputs(inputs: &[Stage3OpeningInputValue<F>]) -> Self {
        let mut store = Self::default();
        for input in inputs {
            store.insert_scalar(input.symbol, input.eval);
            store.insert_point(input.symbol, input.point.clone());
        }
        store
    }

    fn seed_constants(&mut self) {
        for constant in STAGE3_PROGRAM.field_constants {
            self.insert_scalar(constant.symbol, F::from_u64(constant.value as u64));
        }
    }

    fn observe_challenge_vector(
        &mut self,
        plan: &'static Stage3TranscriptSqueezePlan,
        values: &[F],
    ) -> Result<(), VerifyStage3Error> {
        self.insert_point(plan.symbol, values.to_vec());
        if matches!(plan.kind, "challenge_scalar" | "scalar") {
            if values.len() != 1 {
                return Err(VerifyStage3Error::InvalidInputLength {
                    input: plan.symbol,
                    expected: 1,
                    actual: values.len(),
                });
            }
            self.insert_scalar(plan.symbol, values[0]);
        }
        self.evaluate_available_field_exprs()?;
        Ok(())
    }

    fn observe_sumcheck_output(
        &mut self,
        output: &Stage3SumcheckOutput<F>,
    ) -> Result<(), VerifyStage3Error> {
        self.insert_point(output.driver, output.point.clone());
        for instance in STAGE3_PROGRAM
            .instance_results
            .iter()
            .filter(|instance| instance.source == output.driver)
        {
            let end = instance.round_offset + instance.point_arity;
            let mut point = output
                .point
                .get(instance.round_offset..end)
                .ok_or(VerifyStage3Error::InvalidInputLength {
                    input: instance.symbol,
                    expected: end,
                    actual: output.point.len(),
                })?
                .to_vec();
            match instance.point_order {
                "as_is" => {}
                "reverse" => point.reverse(),
                _ => {
                    return Err(VerifyStage3Error::InvalidProof {
                        driver: output.driver,
                        reason: "unsupported point order",
                    });
                }
            }
            self.insert_point(instance.symbol, point);
        }
        for eval in STAGE3_PROGRAM
            .evals
            .iter()
            .filter(|eval| eval.source == output.driver)
        {
            let value = output
                .evals
                .iter()
                .find(|value| value.name == eval.name)
                .or_else(|| output.evals.get(eval.index))
                .ok_or(VerifyStage3Error::MissingValue {
                    symbol: eval.symbol,
                })?
                .value;
            self.insert_scalar(eval.symbol, value);
            self.insert_scalar(eval.name, value);
        }
        self.evaluate_available_points()?;
        self.evaluate_available_field_exprs()?;
        self.verify_opening_equalities()?;
        Ok(())
    }

    fn claim_value(&mut self, claim: &Stage3SumcheckClaimPlan) -> Result<F, VerifyStage3Error> {
        self.evaluate_available_field_exprs()?;
        self.scalar(claim.claim_value)
    }

    fn batch_claim_values(
        &mut self,
        batch: &Stage3SumcheckBatchPlan,
    ) -> Result<Vec<F>, VerifyStage3Error> {
        batch
            .claim_operands
            .iter()
            .map(|symbol| {
                let claim = find_claim(symbol).ok_or(VerifyStage3Error::MissingClaim {
                    batch: batch.symbol,
                    claim: symbol,
                })?;
                self.claim_value(claim)
            })
            .collect()
    }

    fn evaluate_available_points(&mut self) -> Result<(), VerifyStage3Error> {
        loop {
            let mut progress = 0usize;
            for slice in STAGE3_PROGRAM.point_slices {
                if self.try_point(slice.symbol).is_some() {
                    continue;
                }
                let Some(input) = self.try_point(slice.input) else { continue };
                let end = slice.offset + slice.length;
                let point = input
                    .get(slice.offset..end)
                    .ok_or(VerifyStage3Error::InvalidInputLength {
                        input: slice.symbol,
                        expected: end,
                        actual: input.len(),
                    })?
                    .to_vec();
                self.insert_point(slice.symbol, point);
                progress += 1;
            }
            for concat in STAGE3_PROGRAM.point_concats {
                if self.try_point(concat.symbol).is_some() {
                    continue;
                }
                let Some(point) = self.try_concat_point(concat) else { continue };
                if point.len() != concat.arity {
                    return Err(VerifyStage3Error::InvalidInputLength {
                        input: concat.symbol,
                        expected: concat.arity,
                        actual: point.len(),
                    });
                }
                self.insert_point(concat.symbol, point);
                progress += 1;
            }
            if progress == 0 {
                return Ok(());
            }
        }
    }

    fn evaluate_available_field_exprs(&mut self) -> Result<(), VerifyStage3Error> {
        loop {
            let mut progress = 0usize;
            for expr in STAGE3_PROGRAM.field_exprs {
                if self.try_scalar(expr.symbol).is_some() {
                    continue;
                }
                let Some(operands) = self.try_expr_operands(expr) else { continue };
                self.insert_scalar(expr.symbol, evaluate_stage3_field_expr(expr, &operands)?);
                progress += 1;
            }
            if progress == 0 {
                return Ok(());
            }
        }
    }

    fn verify_opening_equalities(&self) -> Result<(), VerifyStage3Error> {
        for equality in STAGE3_PROGRAM.opening_equalities {
            match equality.mode {
                "point_and_eval" => {
                    if self.point(equality.lhs)? != self.point(equality.rhs)?
                        || self.scalar(equality.lhs)? != self.scalar(equality.rhs)?
                    {
                        return Err(VerifyStage3Error::InvalidProof {
                            driver: equality.symbol,
                            reason: "opening claim equality failed",
                        });
                    }
                }
                _ => {
                    return Err(VerifyStage3Error::InvalidProof {
                        driver: equality.symbol,
                        reason: "unsupported opening equality mode",
                    });
                }
            }
        }
        Ok(())
    }

    fn insert_scalar(&mut self, symbol: &'static str, value: F) {
        if let Some((_, existing)) = self.scalars.iter_mut().find(|(name, _)| *name == symbol) {
            *existing = value;
        } else {
            self.scalars.push((symbol, value));
        }
    }

    fn insert_point(&mut self, symbol: &'static str, point: Vec<F>) {
        if let Some((_, existing)) = self.points.iter_mut().find(|(name, _)| *name == symbol) {
            *existing = point;
        } else {
            self.points.push((symbol, point));
        }
    }

    fn scalar(&self, symbol: &'static str) -> Result<F, VerifyStage3Error> {
        self.try_scalar(symbol)
            .ok_or(VerifyStage3Error::MissingValue { symbol })
    }

    fn try_scalar(&self, symbol: &str) -> Option<F> {
        self.scalars
            .iter()
            .find(|(name, _)| *name == symbol)
            .map(|(_, value)| *value)
    }

    fn point(&self, symbol: &'static str) -> Result<&[F], VerifyStage3Error> {
        self.try_point(symbol)
            .ok_or(VerifyStage3Error::MissingValue { symbol })
    }

    fn try_point(&self, symbol: &str) -> Option<&[F]> {
        self.points
            .iter()
            .find(|(name, _)| *name == symbol)
            .map(|(_, point)| point.as_slice())
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

fn evaluate_stage3_field_expr<F: Field>(
    expr: &Stage3FieldExprPlan,
    operands: &[F],
) -> Result<F, VerifyStage3Error> {
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
                    VerifyStage3Error::UnsupportedFieldExpr {
                        symbol: expr.symbol,
                        formula,
                    }
                })?;
                return Ok(pow_field(operands[0], exponent));
            }
            Err(VerifyStage3Error::UnsupportedFieldExpr {
                symbol: expr.symbol,
                formula,
            })
        }
    }
}

fn expected_batched_output_claim(
    driver: &'static Stage3SumcheckDriverPlan,
    store: &Stage3ValueStore<Fr>,
    evals: &[Stage3NamedEval<Fr>],
    point: &[Fr],
    batching_coeffs: &[Fr],
) -> Result<Fr, VerifyStage3Error> {
    let batch = find_batch(driver.batch)?;
    let claims = batch_claims(batch)?;
    let mut expected = Fr::from_u64(0);
    for (claim, coefficient) in claims.iter().zip(batching_coeffs) {
        let instance = STAGE3_PROGRAM
            .instance_results
            .iter()
            .find(|instance| instance.claim == claim.symbol && instance.source == driver.symbol)
            .ok_or(VerifyStage3Error::MissingClaim {
                batch: batch.symbol,
                claim: claim.symbol,
            })?;
        let local_point = point
            .get(instance.round_offset..instance.round_offset + instance.num_rounds)
            .ok_or(VerifyStage3Error::InvalidInputLength {
                input: instance.symbol,
                expected: instance.round_offset + instance.num_rounds,
                actual: point.len(),
            })?;
        let value = match instance.relation {
            "jolt.stage3.spartan_shift" => expected_spartan_shift(store, evals, local_point)?,
            "jolt.stage3.instruction_input" => expected_instruction_input(store, evals, local_point)?,
            "jolt.stage3.registers_claim_reduction" => expected_registers(store, evals, local_point)?,
            relation => return Err(VerifyStage3Error::UnsupportedRelation { relation }),
        };
        expected += *coefficient * value;
    }
    Ok(expected)
}

fn expected_spartan_shift(
    store: &Stage3ValueStore<Fr>,
    evals: &[Stage3NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage3Error> {
    let opening_point = reverse_slice(local_point);
    let eq_outer =
        EqPlusOnePolynomial::<Fr>::new(store.point("stage3.input.stage1.NextPC")?.to_vec())
            .evaluate(&opening_point);
    let eq_product = EqPlusOnePolynomial::<Fr>::new(
        store
            .point("stage3.input.stage2.product_virtual.NextIsNoop")?
            .to_vec(),
    )
    .evaluate(&opening_point);
    let weighted_outer = eval_by_name(evals, "stage3.spartan_shift.eval.UnexpandedPC")?
        + store.scalar("stage3.spartan_shift.gamma")?
            * eval_by_name(evals, "stage3.spartan_shift.eval.PC")?
        + store.scalar("stage3.spartan_shift.gamma2")?
            * eval_by_name(evals, "stage3.spartan_shift.eval.OpFlagVirtualInstruction")?
        + store.scalar("stage3.spartan_shift.gamma3")?
            * eval_by_name(evals, "stage3.spartan_shift.eval.OpFlagIsFirstInSequence")?;
    Ok(eq_outer * weighted_outer
        + store.scalar("stage3.spartan_shift.gamma4")?
            * eq_product
            * (Fr::from_u64(1)
                - eval_by_name(evals, "stage3.spartan_shift.eval.InstructionFlagIsNoop")?))
}

fn expected_instruction_input(
    store: &Stage3ValueStore<Fr>,
    evals: &[Stage3NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage3Error> {
    let opening_point = reverse_slice(local_point);
    let eq_eval = EqPolynomial::<Fr>::mle(
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

fn expected_registers(
    store: &Stage3ValueStore<Fr>,
    evals: &[Stage3NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage3Error> {
    let opening_point = reverse_slice(local_point);
    let eq_eval = EqPolynomial::<Fr>::mle(
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

fn append_opening_claims<T>(
    store: &mut Stage3ValueStore<Fr>,
    transcript: &mut T,
    evals: &[Stage3NamedEval<Fr>],
) -> Result<(), VerifyStage3Error>
where
    T: Transcript<Challenge = Fr>,
{
    if STAGE3_PROGRAM.opening_batches.is_empty() {
        for eval in evals {
            append_labeled_scalar(transcript, "opening_claim", &eval.value);
        }
        return Ok(());
    }
    store.evaluate_available_points()?;
    let mut seen = STAGE3_PROGRAM
        .opening_inputs
        .iter()
        .filter_map(|input| {
            store
                .try_point(input.symbol)
                .map(|point| (input.claim_kind, input.oracle, point.to_vec()))
        })
        .collect::<Vec<_>>();
    for batch in STAGE3_PROGRAM.opening_batches {
        for symbol in batch.claim_operands {
            let claim = find_opening_claim(symbol).ok_or(VerifyStage3Error::MissingClaim {
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

fn find_squeeze(symbol: &str) -> Option<&'static Stage3TranscriptSqueezePlan> {
    STAGE3_PROGRAM
        .transcript_squeezes
        .iter()
        .find(|squeeze| squeeze.symbol == symbol)
}

fn find_driver(symbol: &str) -> Option<&'static Stage3SumcheckDriverPlan> {
    STAGE3_PROGRAM
        .drivers
        .iter()
        .find(|driver| driver.symbol == symbol)
}

fn find_batch(symbol: &'static str) -> Result<&'static Stage3SumcheckBatchPlan, VerifyStage3Error> {
    STAGE3_PROGRAM
        .batches
        .iter()
        .find(|batch| batch.symbol == symbol)
        .ok_or(VerifyStage3Error::MissingBatch {
            driver: symbol,
            batch: symbol,
        })
}

fn find_claim(symbol: &str) -> Option<&'static Stage3SumcheckClaimPlan> {
    STAGE3_PROGRAM
        .claims
        .iter()
        .find(|claim| claim.symbol == symbol)
}

fn find_opening_claim(symbol: &str) -> Option<&'static Stage3OpeningClaimPlan> {
    STAGE3_PROGRAM
        .opening_claims
        .iter()
        .find(|claim| claim.symbol == symbol)
}

fn batch_claims(
    batch: &Stage3SumcheckBatchPlan,
) -> Result<Vec<&'static Stage3SumcheckClaimPlan>, VerifyStage3Error> {
    batch
        .claim_operands
        .iter()
        .map(|symbol| {
            find_claim(symbol).ok_or(VerifyStage3Error::MissingClaim {
                batch: batch.symbol,
                claim: symbol,
            })
        })
        .collect()
}

fn eval_by_name(evals: &[Stage3NamedEval<Fr>], name: &'static str) -> Result<Fr, VerifyStage3Error> {
    evals
        .iter()
        .find(|eval| eval.name == name)
        .map(|eval| eval.value)
        .ok_or(VerifyStage3Error::MissingValue { symbol: name })
}

fn append_labeled_scalar<T>(transcript: &mut T, label: &'static str, scalar: &Fr)
where
    T: Transcript<Challenge = Fr>,
{
    transcript.append(&Label(label.as_bytes()));
    transcript.append(scalar);
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

fn single_operand<F: Field>(symbol: &'static str, operands: &[F]) -> Result<F, VerifyStage3Error> {
    require_operand_count(symbol, 1, operands.len())?;
    Ok(operands[0])
}

fn require_operand_count(
    input: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), VerifyStage3Error> {
    if expected == actual {
        Ok(())
    } else {
        Err(VerifyStage3Error::InvalidInputLength {
            input,
            expected,
            actual,
        })
    }
}

fn reverse_slice(values: &[Fr]) -> Vec<Fr> {
    values.iter().rev().copied().collect()
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
        .map(|value| format!("    {value},"))
        .collect::<Vec<_>>()
        .join("\n");
    format!("pub const {name}: &[usize] = &[\n{entries}\n];\n\n")
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

fn string_attr(operation: OperationRef<'_, '_>, attr: &str) -> Result<String, EmitError> {
    operation
        .attribute(attr)
        .ok()
        .and_then(string_attribute_value)
        .ok_or_else(|| attr_error(operation, attr, "string"))
}

fn symbol_attr(operation: OperationRef<'_, '_>, attr: &str) -> Result<String, EmitError> {
    operation
        .attribute(attr)
        .ok()
        .and_then(symbol_attribute_value)
        .ok_or_else(|| attr_error(operation, attr, "symbol"))
}

fn symbol_array_attr(
    operation: OperationRef<'_, '_>,
    attr: &str,
) -> Result<Vec<String>, EmitError> {
    let attribute = operation
        .attribute(attr)
        .map(|attribute| attribute.to_string())
        .ok()
        .ok_or_else(|| attr_error(operation, attr, "symbol array"))?;
    parse_symbol_array(&attribute).ok_or_else(|| attr_error(operation, attr, "symbol array"))
}

fn parse_symbol_array(attribute: &str) -> Option<Vec<String>> {
    let inner = attribute.strip_prefix('[')?.strip_suffix(']')?.trim();
    if inner.is_empty() {
        return Some(Vec::new());
    }
    inner
        .split(',')
        .map(|item| item.trim().strip_prefix('@').map(ToOwned::to_owned))
        .collect()
}

fn int_attr(operation: OperationRef<'_, '_>, attr: &str) -> Result<usize, EmitError> {
    operation
        .attribute(attr)
        .map(parse_integer_attr)
        .ok()
        .flatten()
        .ok_or_else(|| attr_error(operation, attr, "integer"))
}

fn signed_int_attr(operation: OperationRef<'_, '_>, attr: &str) -> Result<isize, EmitError> {
    operation
        .attribute(attr)
        .map(parse_signed_integer_attr)
        .ok()
        .flatten()
        .ok_or_else(|| attr_error(operation, attr, "signed integer"))
}

fn parse_integer_attr(attribute: Attribute<'_>) -> Option<usize> {
    attribute
        .to_string()
        .split_whitespace()
        .next()
        .and_then(|value| value.parse().ok())
}

fn parse_signed_integer_attr(attribute: Attribute<'_>) -> Option<isize> {
    attribute
        .to_string()
        .split_whitespace()
        .next()
        .and_then(|value| value.parse().ok())
}

fn int_array_attr(operation: OperationRef<'_, '_>, attr: &str) -> Result<Vec<usize>, EmitError> {
    let attribute = operation
        .attribute(attr)
        .map(|attribute| attribute.to_string())
        .ok()
        .ok_or_else(|| attr_error(operation, attr, "integer array"))?;
    parse_int_array(&attribute).ok_or_else(|| attr_error(operation, attr, "integer array"))
}

fn parse_int_array(attribute: &str) -> Option<Vec<usize>> {
    let inner = attribute.strip_prefix('[')?.strip_suffix(']')?.trim();
    if inner.is_empty() {
        return Some(Vec::new());
    }
    inner
        .split(',')
        .map(|item| item.trim().parse().ok())
        .collect()
}

fn operand_symbols(
    operation: OperationRef<'_, '_>,
    start_index: usize,
) -> Result<Vec<String>, EmitError> {
    (start_index..operation.operand_count())
        .map(|index| operand_symbol(operation, index))
        .collect()
}

fn operand_symbol(operation: OperationRef<'_, '_>, index: usize) -> Result<String, EmitError> {
    let operand = operation.operand(index).map_err(|_| {
        EmitError::new(format!(
            "{} requires operand {index}",
            operation_name(operation)
        ))
    })?;
    let owner = OperationResult::try_from(operand).map_err(|_| {
        EmitError::new(format!(
            "{} operand {index} must be an op result",
            operation_name(operation)
        ))
    })?;
    string_attr(owner.owner(), "sym_name")
}

fn attr_error(operation: OperationRef<'_, '_>, attr: &str, expected: &str) -> EmitError {
    EmitError::new(format!(
        "{} attr `{attr}` is not a {expected}",
        operation_name(operation)
    ))
}

fn operation_name<'c: 'a, 'a>(operation: impl OperationLike<'c, 'a>) -> String {
    operation
        .name()
        .as_string_ref()
        .as_str()
        .unwrap_or("<invalid-operation-name>")
        .to_owned()
}
