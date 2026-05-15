use std::collections::{BTreeMap, BTreeSet};

use melior::ir::block::BlockLike;
use melior::ir::operation::{OperationLike, OperationResult};
use melior::ir::{Attribute, OperationRef};

use crate::emit::rust::{push_format, EmitError, RustSourceFile};
use crate::ir::{string_attribute_value, symbol_attribute_value, BoltModule, Cpu, Role};
use crate::protocols::jolt::rust_target_plan::{
    ClaimKind, FieldExprKind, JoltVerifierRelationKind, ProgramStepKind, RustTargetPlanError,
    TranscriptSqueezeKind,
};
use crate::schema::verify_cpu_schema;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2CpuProgram {
    pub role: Role,
    pub params: Stage2Params,
    pub steps: Vec<Stage2ProgramStepPlan>,
    pub transcript_squeezes: Vec<Stage2TranscriptSqueezePlan>,
    pub opening_inputs: Vec<Stage2OpeningInputPlan>,
    pub field_constants: Vec<Stage2FieldConstantPlan>,
    pub field_exprs: Vec<Stage2FieldExprPlan>,
    pub kernels: Vec<Stage2KernelPlan>,
    pub claims: Vec<Stage2SumcheckClaimPlan>,
    pub batches: Vec<Stage2SumcheckBatchPlan>,
    pub drivers: Vec<Stage2SumcheckDriverPlan>,
    pub instance_results: Vec<Stage2SumcheckInstanceResultPlan>,
    pub evals: Vec<Stage2SumcheckEvalPlan>,
    pub point_slices: Vec<Stage2PointSlicePlan>,
    pub point_concats: Vec<Stage2PointConcatPlan>,
    pub opening_claims: Vec<Stage2OpeningClaimPlan>,
    pub opening_batches: Vec<Stage2OpeningBatchPlan>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2Params {
    pub field: String,
    pub pcs: String,
    pub transcript: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2KernelPlan {
    pub symbol: String,
    pub relation: String,
    pub kind: String,
    pub backend: String,
    pub abi: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2TranscriptSqueezePlan {
    pub symbol: String,
    pub label: String,
    pub kind: String,
    pub count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2ProgramStepPlan {
    pub kind: String,
    pub symbol: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2OpeningInputPlan {
    pub symbol: String,
    pub source_stage: String,
    pub source_claim: String,
    pub oracle: String,
    pub domain: String,
    pub point_arity: usize,
    pub claim_kind: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2FieldConstantPlan {
    pub symbol: String,
    pub field: String,
    pub value: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2FieldExprPlan {
    pub symbol: String,
    pub kind: String,
    pub formula: String,
    pub operand_names: Vec<String>,
    pub operands: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2SumcheckClaimPlan {
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
pub struct Stage2SumcheckBatchPlan {
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
pub struct Stage2SumcheckDriverPlan {
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
pub struct Stage2SumcheckInstanceResultPlan {
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
pub struct Stage2SumcheckEvalPlan {
    pub symbol: String,
    pub source: String,
    pub name: String,
    pub index: usize,
    pub oracle: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2PointSlicePlan {
    pub symbol: String,
    pub source: String,
    pub offset: usize,
    pub length: usize,
    pub input: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2PointConcatPlan {
    pub symbol: String,
    pub layout: String,
    pub arity: usize,
    pub inputs: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2OpeningClaimPlan {
    pub symbol: String,
    pub oracle: String,
    pub domain: String,
    pub point_arity: usize,
    pub claim_kind: String,
    pub point_source: String,
    pub eval_source: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2OpeningBatchPlan {
    pub symbol: String,
    pub stage: String,
    pub proof_slot: String,
    pub policy: String,
    pub count: usize,
    pub ordered_claims: Vec<String>,
    pub claim_operands: Vec<String>,
}

pub fn stage2_cpu_program(module: &BoltModule<'_, Cpu>) -> Result<Stage2CpuProgram, EmitError> {
    verify_cpu_schema(module)?;
    let program = Stage2CpuProgram::from_module(module)?;
    program.verify_supported_target()?;
    Ok(program)
}

pub fn emit_stage2_rust(module: &BoltModule<'_, Cpu>) -> Result<RustSourceFile, EmitError> {
    let program = stage2_cpu_program(module)?;

    Ok(RustSourceFile {
        filename: program.filename().to_owned(),
        source: program.emit_source()?,
    })
}

impl Stage2CpuProgram {
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
        let mut opening_batches = Vec::new();

        let mut operation = module.as_mlir_module().body().first_operation();
        while let Some(op) = operation {
            operation = op.next_in_block();
            match operation_name(op).as_str() {
                "cpu.params" => {
                    params = Some(Stage2Params {
                        field: symbol_attr(op, "field")?,
                        pcs: symbol_attr(op, "pcs")?,
                        transcript: symbol_attr(op, "transcript")?,
                    });
                }
                "cpu.kernel" => {
                    kernels.push(Stage2KernelPlan {
                        symbol: string_attr(op, "sym_name")?,
                        relation: symbol_attr(op, "relation")?,
                        kind: string_attr(op, "kind")?,
                        backend: string_attr(op, "backend")?,
                        abi: string_attr(op, "abi")?,
                    });
                }
                "cpu.transcript_squeeze" => {
                    let symbol = string_attr(op, "sym_name")?;
                    steps.push(Stage2ProgramStepPlan {
                        kind: "transcript_squeeze".to_owned(),
                        symbol: symbol.clone(),
                    });
                    transcript_squeezes.push(Stage2TranscriptSqueezePlan {
                        symbol,
                        label: string_attr(op, "label")?,
                        kind: string_attr(op, "kind")?,
                        count: int_attr(op, "count")?,
                    });
                }
                "cpu.opening_input" => {
                    opening_inputs.push(Stage2OpeningInputPlan {
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
                    field_constants.push(Stage2FieldConstantPlan {
                        symbol: string_attr(op, "sym_name")?,
                        field: symbol_attr(op, "field")?,
                        value: int_attr(op, "value")?,
                    });
                }
                "cpu.field_zero" => {
                    field_constants.push(Stage2FieldConstantPlan {
                        symbol: string_attr(op, "sym_name")?,
                        field: symbol_attr(op, "field")?,
                        value: 0,
                    });
                }
                "cpu.field_one" => {
                    field_constants.push(Stage2FieldConstantPlan {
                        symbol: string_attr(op, "sym_name")?,
                        field: symbol_attr(op, "field")?,
                        value: 1,
                    });
                }
                "cpu.field_add" | "cpu.field_sub" | "cpu.field_mul" | "cpu.field_neg" => {
                    let operands = operand_symbols(op, 0)?;
                    field_exprs.push(Stage2FieldExprPlan {
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
                    field_exprs.push(Stage2FieldExprPlan {
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
                    field_exprs.push(Stage2FieldExprPlan {
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
                    claims.push(Stage2SumcheckClaimPlan {
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
                    claims.push(Stage2SumcheckClaimPlan {
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
                    batches.push(Stage2SumcheckBatchPlan {
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
                    steps.push(Stage2ProgramStepPlan {
                        kind: "sumcheck_driver".to_owned(),
                        symbol: symbol.clone(),
                    });
                    drivers.push(Stage2SumcheckDriverPlan {
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
                    steps.push(Stage2ProgramStepPlan {
                        kind: "sumcheck_driver".to_owned(),
                        symbol: symbol.clone(),
                    });
                    drivers.push(Stage2SumcheckDriverPlan {
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
                    instance_results.push(Stage2SumcheckInstanceResultPlan {
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
                    evals.push(Stage2SumcheckEvalPlan {
                        symbol: string_attr(op, "sym_name")?,
                        source: symbol_attr(op, "source")?,
                        name: symbol_attr(op, "name")?,
                        index: int_attr(op, "index")?,
                        oracle: symbol_attr(op, "oracle")?,
                    });
                }
                "cpu.point_slice" => {
                    point_slices.push(Stage2PointSlicePlan {
                        symbol: string_attr(op, "sym_name")?,
                        source: symbol_attr(op, "source")?,
                        offset: int_attr(op, "offset")?,
                        length: int_attr(op, "length")?,
                        input: operand_symbol(op, 0)?,
                    });
                }
                "cpu.point_concat" => {
                    point_concats.push(Stage2PointConcatPlan {
                        symbol: string_attr(op, "sym_name")?,
                        layout: string_attr(op, "layout")?,
                        arity: int_attr(op, "arity")?,
                        inputs: operand_symbols(op, 0)?,
                    });
                }
                "cpu.opening_claim" => {
                    opening_claims.push(Stage2OpeningClaimPlan {
                        symbol: string_attr(op, "sym_name")?,
                        oracle: symbol_attr(op, "oracle")?,
                        domain: symbol_attr(op, "domain")?,
                        point_arity: int_attr(op, "point_arity")?,
                        claim_kind: string_attr(op, "claim_kind")?,
                        point_source: operand_symbol(op, 0)?,
                        eval_source: operand_symbol(op, 1)?,
                    });
                }
                "cpu.opening_batch" => {
                    opening_batches.push(Stage2OpeningBatchPlan {
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
            Role::Verifier => {
                self.verify_verifier_driver_bindings()?;
                self.verify_verifier_rust_target()?;
            }
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
                    "stage2 transcript squeeze @{} has unsupported kind `{}`",
                    squeeze.symbol, squeeze.kind
                )));
            }
            if squeeze.count == 0 {
                return Err(EmitError::new(format!(
                    "stage2 transcript squeeze @{} has zero count",
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
                    "stage2 kernel @{} targets unsupported backend `{}`",
                    kernel.symbol, kernel.backend
                )));
            }
            if kernel.kind != "sumcheck" {
                return Err(EmitError::new(format!(
                    "stage2 kernel @{} has unsupported kind `{}`",
                    kernel.symbol, kernel.kind
                )));
            }
            let expected_abi = match kernel.relation.as_str() {
                "jolt.stage2.product_virtual.uniskip" => "jolt_stage2_product_virtual_uniskip",
                "jolt.stage2.ram.read_write" => "jolt_stage2_ram_read_write",
                "jolt.stage2.product_virtual.remainder" => "jolt_stage2_product_virtual_remainder",
                "jolt.stage2.instruction_lookup.claim_reduction" => {
                    "jolt_stage2_instruction_lookup_claim_reduction"
                }
                "jolt.stage2.ram.raf_evaluation" => "jolt_stage2_ram_raf_evaluation",
                "jolt.stage2.ram.output_check" => "jolt_stage2_ram_output_check",
                "jolt.stage2.batched" => "jolt_stage2_batched",
                _ => {
                    return Err(EmitError::new(format!(
                        "unsupported stage2 kernel relation @{}",
                        kernel.relation
                    )));
                }
            };
            if kernel.abi != expected_abi {
                return Err(EmitError::new(format!(
                    "stage2 kernel @{} ABI `{}` does not match relation @{}",
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
                "verifier stage2 program must not contain kernels",
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

    fn verify_verifier_rust_target(&self) -> Result<(), EmitError> {
        for step in &self.steps {
            let _ = ProgramStepKind::from_cpu_attr(&step.kind).map_err(rust_target_plan_error)?;
        }
        for squeeze in &self.transcript_squeezes {
            let _ = TranscriptSqueezeKind::from_cpu_attr(&squeeze.kind)
                .map_err(rust_target_plan_error)?;
        }
        for input in &self.opening_inputs {
            let _ = ClaimKind::from_cpu_attr(&input.claim_kind).map_err(rust_target_plan_error)?;
        }
        for expr in &self.field_exprs {
            let _ = FieldExprKind::from_cpu_attr(&expr.formula).map_err(rust_target_plan_error)?;
        }
        for claim in &self.claims {
            let relation = claim
                .relation
                .as_deref()
                .ok_or_else(|| missing_role_binding("verifier claim relation", &claim.symbol))?;
            let _ = JoltVerifierRelationKind::from_cpu_attr(relation)
                .map_err(rust_target_plan_error)?;
        }
        for driver in &self.drivers {
            let relation = driver
                .relation
                .as_deref()
                .ok_or_else(|| missing_role_binding("verifier driver relation", &driver.symbol))?;
            let _ = JoltVerifierRelationKind::from_cpu_attr(relation)
                .map_err(rust_target_plan_error)?;
        }
        for instance in &self.instance_results {
            let _ = JoltVerifierRelationKind::from_cpu_attr(&instance.relation)
                .map_err(rust_target_plan_error)?;
        }
        for claim in &self.opening_claims {
            let _ = ClaimKind::from_cpu_attr(&claim.claim_kind).map_err(rust_target_plan_error)?;
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
            Role::Prover => "prove_stage2.rs",
            Role::Verifier => "verify_stage2.rs",
        }
    }

    fn emit_prover_imports() -> &'static str {
        "use jolt_field::Fr;\n\
         use jolt_kernels::stage2::{execute_stage2_program, Stage2CpuProgramPlan, Stage2ExecutionArtifacts, Stage2ExecutionMode, Stage2FieldConstantPlan, Stage2FieldExprPlan, Stage2KernelError, Stage2KernelExecutor, Stage2KernelPlan, Stage2OpeningBatchPlan, Stage2OpeningClaimPlan, Stage2OpeningInputPlan, Stage2Params, Stage2PointConcatPlan, Stage2PointSlicePlan, Stage2ProgramStepPlan, Stage2SumcheckBatchPlan, Stage2SumcheckClaimPlan, Stage2SumcheckDriverPlan, Stage2SumcheckEvalPlan, Stage2SumcheckInstanceResultPlan, Stage2TranscriptSqueezePlan};\n\
         use jolt_transcript::{Blake2bTranscript, Transcript};"
    }

    fn emit_prover_types() -> &'static str {
        "pub type DefaultStage2Transcript = Blake2bTranscript<Fr>;\n"
    }

    fn emit_verifier_imports() -> &'static str {
        "use bolt_verifier_runtime::{append_labeled_scalar, batch_claims, eval_by_name, find_batch, find_plan, reverse_slice};\n\
         use jolt_field::{Field, Fr, MulPow2, MulPrimitiveInt, RingCore};\n\
         use jolt_poly::lagrange::{lagrange_evals, lagrange_kernel_eval};\n\
         use jolt_poly::{EqPolynomial, UnivariatePoly};\n\
         use jolt_sumcheck::{CompressedLabeledRoundPoly, SumcheckClaim, SumcheckError, SumcheckVerifier};\n\
         use jolt_transcript::{Blake2bTranscript, LabelWithCount, Transcript};"
    }

    fn emit_verifier_types() -> &'static str {
        r"pub type DefaultStage2Transcript = Blake2bTranscript<Fr>;

pub type Stage2NamedEval<F> = bolt_verifier_runtime::StageNamedEval<F>;
pub type Stage2SumcheckOutput<F> = bolt_verifier_runtime::StageSumcheckOutput<F>;
pub type Stage2ChallengeVector<F> = bolt_verifier_runtime::StageChallengeVector<F>;
pub type Stage2ExecutionArtifacts<F> = bolt_verifier_runtime::StageExecutionArtifacts<F>;
pub type Stage2Proof<F> = bolt_verifier_runtime::StageProof<F>;
pub type Stage2OpeningInputValue<F> = bolt_verifier_runtime::StageOpeningInputValue<F>;
pub type Stage2VerifierProgramPlan = bolt_verifier_runtime::StageVerifierProgramPlanNoEqualities<Stage2RelationKind>;
pub type Stage2SumcheckClaimPlan = bolt_verifier_runtime::SumcheckClaimPlan<Stage2RelationKind>;
pub type Stage2SumcheckDriverPlan = bolt_verifier_runtime::SumcheckDriverPlan<Stage2RelationKind>;
pub type Stage2SumcheckInstanceResultPlan = bolt_verifier_runtime::SumcheckInstanceResultPlan<Stage2RelationKind>;

pub use super::jolt_relations::JoltRelationKind as Stage2RelationKind;
pub use bolt_verifier_runtime::{
    ClaimKind as Stage2ClaimKind, FieldConstantPlan as Stage2FieldConstantPlan,
    FieldExprKind as Stage2FieldExprKind,
    FieldExprPlan as Stage2FieldExprPlan,
    OpeningBatchPlan as Stage2OpeningBatchPlan, OpeningClaimPlan as Stage2OpeningClaimPlan,
    OpeningInputPlan as Stage2OpeningInputPlan, PointConcatPlan as Stage2PointConcatPlan,
    PointSlicePlan as Stage2PointSlicePlan, ProgramStepKind as Stage2ProgramStepKind,
    ProgramStepPlan as Stage2ProgramStepPlan, StageParams as Stage2Params,
    SumcheckBatchPlan as Stage2SumcheckBatchPlan,
    SumcheckEvalPlan as Stage2SumcheckEvalPlan,
    TranscriptSqueezeKind as Stage2TranscriptSqueezeKind,
    TranscriptSqueezePlan as Stage2TranscriptSqueezePlan,
};

#[derive(Clone, Copy, Debug)]
pub struct Stage2RamAccess {
    pub remapped_address: Option<usize>,
    pub read_value: u64,
    pub write_value: u64,
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

#[derive(Clone, Copy, Debug)]
pub struct Stage2RamReadWriteOutputPlan {
    pub cycle_point: &'static str,
    pub gamma: &'static str,
    pub val_eval: &'static str,
    pub ra_eval: &'static str,
    pub inc_eval: &'static str,
}

#[derive(Clone, Debug, Default)]
struct Stage2ValueStore<F: Field>(bolt_verifier_runtime::ValueStore<F>);

#[derive(Debug)]
pub enum VerifyStage2Error {
    UnexpectedProofCount { expected: usize, got: usize },
    MissingProof { driver: &'static str },
    MissingBatch { driver: &'static str, batch: &'static str },
    MissingClaim { batch: &'static str, claim: &'static str },
    MissingValue { symbol: &'static str },
    InvalidInputLength { input: &'static str, expected: usize, actual: usize },
    InvalidProof { driver: &'static str, reason: &'static str },
    UnsupportedRelation { relation: Stage2RelationKind },
    MissingRam { context: &'static str },
    Sumcheck { driver: &'static str, error: SumcheckError<Fr> },
}

bolt_verifier_runtime::impl_runtime_plan_error_conversion!(VerifyStage2Error);
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
            "pub const STAGE2_PROGRAM: Stage2CpuProgramPlan = Stage2CpuProgramPlan {\n\
             \x20   params: STAGE2_PARAMS,\n\
             \x20   steps: STAGE2_PROGRAM_STEPS,\n\
             \x20   transcript_squeezes: STAGE2_TRANSCRIPT_SQUEEZES,\n\
             \x20   opening_inputs: STAGE2_OPENING_INPUTS,\n\
             \x20   field_constants: STAGE2_FIELD_CONSTANTS,\n\
             \x20   field_exprs: STAGE2_FIELD_EXPRS,\n\
             \x20   kernels: STAGE2_KERNELS,\n\
             \x20   claims: STAGE2_SUMCHECK_CLAIMS,\n\
             \x20   batches: STAGE2_SUMCHECK_BATCHES,\n\
             \x20   drivers: STAGE2_SUMCHECK_DRIVERS,\n\
             \x20   instance_results: STAGE2_SUMCHECK_INSTANCE_RESULTS,\n\
             \x20   evals: STAGE2_SUMCHECK_EVALS,\n\
             \x20   point_slices: STAGE2_POINT_SLICES,\n\
             \x20   point_concats: STAGE2_POINT_CONCATS,\n\
             \x20   opening_claims: STAGE2_OPENING_CLAIMS,\n\
             \x20   opening_batches: STAGE2_OPENING_BATCHES,\n\
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
        source.push_str(Self::emit_verifier_relation_output_constants());
        source.push_str(
            "pub const STAGE2_PROGRAM: Stage2VerifierProgramPlan = Stage2VerifierProgramPlan {\n\
             \x20   params: STAGE2_PARAMS,\n\
             \x20   steps: STAGE2_PROGRAM_STEPS,\n\
             \x20   transcript_squeezes: STAGE2_TRANSCRIPT_SQUEEZES,\n\
             \x20   opening_inputs: STAGE2_OPENING_INPUTS,\n\
             \x20   field_constants: STAGE2_FIELD_CONSTANTS,\n\
             \x20   field_exprs: STAGE2_FIELD_EXPRS,\n\
             \x20   claims: STAGE2_SUMCHECK_CLAIMS,\n\
             \x20   batches: STAGE2_SUMCHECK_BATCHES,\n\
             \x20   drivers: STAGE2_SUMCHECK_DRIVERS,\n\
             \x20   instance_results: STAGE2_SUMCHECK_INSTANCE_RESULTS,\n\
             \x20   evals: STAGE2_SUMCHECK_EVALS,\n\
             \x20   point_slices: STAGE2_POINT_SLICES,\n\
             \x20   point_concats: STAGE2_POINT_CONCATS,\n\
             \x20   opening_claims: STAGE2_OPENING_CLAIMS,\n\
             \x20   opening_batches: STAGE2_OPENING_BATCHES,\n\
             };\n",
        );
        Ok(source)
    }

    fn emit_verifier_relation_output_constants() -> &'static str {
        "pub const STAGE2_RAM_READ_WRITE_OUTPUT: Stage2RamReadWriteOutputPlan = Stage2RamReadWriteOutputPlan {\n\
         \x20   cycle_point: \"stage2.input.stage1.RamReadValue\",\n\
         \x20   gamma: \"stage2.ram_read_write.gamma\",\n\
         \x20   val_eval: \"stage2.ram_read_write.eval.RamVal\",\n\
         \x20   ra_eval: \"stage2.ram_read_write.eval.RamRa\",\n\
         \x20   inc_eval: \"stage2.ram_read_write.eval.RamInc\",\n\
         };\n\n"
    }

    fn emit_shared_constants(&self) -> Result<String, EmitError> {
        let mut source = String::new();
        push_format(
            &mut source,
            format_args!(
                "pub const STAGE2_PARAMS: Stage2Params = Stage2Params {{ field: {}, pcs: {}, transcript: {} }};\n",
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
        Ok(source)
    }

    fn emit_program_step_constants(&self) -> Result<String, EmitError> {
        let steps = self
            .steps
            .iter()
            .map(|step| {
                Ok(format!(
                    "    Stage2ProgramStepPlan {{ kind: {}, symbol: {} }},",
                    super::plan_tokens::role_program_step_kind_expr(
                        "Stage2", &self.role, &step.kind
                    )?,
                    rust_str(&step.symbol),
                ))
            })
            .collect::<Result<Vec<_>, EmitError>>()?
            .join("\n");
        Ok(format!(
            "pub const STAGE2_PROGRAM_STEPS: &[Stage2ProgramStepPlan] = &[\n{steps}\n];\n\n"
        ))
    }

    fn emit_transcript_squeeze_constants(&self) -> Result<String, EmitError> {
        let squeezes = self
            .transcript_squeezes
            .iter()
            .map(|squeeze| {
                Ok(format!(
                    "    Stage2TranscriptSqueezePlan {{ symbol: {}, label: {}, kind: {}, count: {} }},",
                    rust_str(&squeeze.symbol),
                    rust_str(&squeeze.label),
                    super::plan_tokens::role_transcript_squeeze_kind_expr(
                        "Stage2",
                        &self.role,
                        &squeeze.kind
                    )?,
                    squeeze.count,
                ))
            })
            .collect::<Result<Vec<_>, EmitError>>()?
            .join("\n");
        Ok(format!(
            "pub const STAGE2_TRANSCRIPT_SQUEEZES: &[Stage2TranscriptSqueezePlan] = &[\n{squeezes}\n];\n\n"
        ))
    }

    fn emit_opening_input_constants(&self) -> Result<String, EmitError> {
        let inputs = self
            .opening_inputs
            .iter()
            .map(|input| {
                Ok(format!(
                    "    Stage2OpeningInputPlan {{ symbol: {}, source_stage: {}, source_claim: {}, oracle: {}, domain: {}, point_arity: {}, claim_kind: {} }},",
                    rust_str(&input.symbol),
                    rust_str(&input.source_stage),
                    rust_str(&input.source_claim),
                    rust_str(&input.oracle),
                    rust_str(&input.domain),
                    input.point_arity,
                    super::plan_tokens::role_claim_kind_expr("Stage2", &self.role, &input.claim_kind)?
                ))
            })
            .collect::<Result<Vec<_>, EmitError>>()?
            .join("\n");
        Ok(format!(
            "pub const STAGE2_OPENING_INPUTS: &[Stage2OpeningInputPlan] = &[\n{inputs}\n];\n\n"
        ))
    }

    fn emit_field_constant_constants(&self) -> String {
        let constants = self
            .field_constants
            .iter()
            .map(|constant| {
                format!(
                    "    Stage2FieldConstantPlan {{ symbol: {}, field: {}, value: {} }},",
                    rust_str(&constant.symbol),
                    rust_str(&constant.field),
                    constant.value
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "pub const STAGE2_FIELD_CONSTANTS: &[Stage2FieldConstantPlan] = &[\n{constants}\n];\n\n"
        )
    }

    fn emit_field_expr_constants(&self) -> Result<String, EmitError> {
        if self.role == Role::Verifier {
            let exprs = self
                .field_exprs
                .iter()
                .map(|expr| {
                    Ok(format!(
                        "    Stage2FieldExprPlan {{ symbol: {}, kind: {}, operands: {} }},",
                        rust_str(&expr.symbol),
                        super::plan_tokens::role_field_expr_kind_expr(
                            "Stage2",
                            &self.role,
                            &expr.formula
                        )?,
                        super::plan_tokens::rust_str_slice_expr(&expr.operands)
                    ))
                })
                .collect::<Result<Vec<_>, EmitError>>()?
                .join("\n");
            return Ok(format!(
                "pub const STAGE2_FIELD_EXPRS: &[Stage2FieldExprPlan] = &[\n{exprs}\n];\n"
            ));
        }

        let mut source = String::new();
        let mut arrays = Vec::new();
        let mut array_refs = Vec::new();
        for (index, expr) in self.field_exprs.iter().enumerate() {
            let operands = intern_str_array(
                &mut source,
                &mut arrays,
                "STAGE2_FIELD_EXPR_OPERANDS",
                &expr.operands,
            );
            let operand_names = intern_str_array(
                &mut source,
                &mut arrays,
                "STAGE2_FIELD_EXPR_OPERANDS",
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
                    "    Stage2FieldExprPlan {{ symbol: {}, kind: {}, formula: {}, operand_names: {operand_names}, operands: {operands} }},",
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
                "pub const STAGE2_FIELD_EXPRS: &[Stage2FieldExprPlan] = &[\n{exprs}\n];\n"
            ),
        );
        Ok(source)
    }

    fn emit_kernel_constants(&self) -> String {
        let kernels = self
            .kernels
            .iter()
            .map(|kernel| {
                format!(
                    "    Stage2KernelPlan {{ symbol: {}, relation: {}, kind: {}, backend: {}, abi: {} }},",
                    rust_str(&kernel.symbol),
                    rust_str(&kernel.relation),
                    rust_str(&kernel.kind),
                    rust_str(&kernel.backend),
                    rust_str(&kernel.abi)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!("pub const STAGE2_KERNELS: &[Stage2KernelPlan] = &[\n{kernels}\n];\n\n")
    }

    fn emit_prover_sumcheck_claim_constants(&self) -> Result<String, EmitError> {
        self.emit_sumcheck_claim_constants(true)
    }

    fn emit_verifier_sumcheck_claim_constants(&self) -> Result<String, EmitError> {
        self.emit_sumcheck_claim_constants(false)
    }

    fn emit_sumcheck_claim_constants(&self, prover: bool) -> Result<String, EmitError> {
        let mut source = String::new();
        if prover {
            for (index, claim) in self.claims.iter().enumerate() {
                source.push_str(&emit_str_array(
                    &format!("STAGE2_SUMCHECK_CLAIM_{index}_INPUT_OPENINGS"),
                    &claim.input_openings,
                ));
            }
        }
        let mut claims = Vec::new();
        for (index, claim) in self.claims.iter().enumerate() {
            if prover {
                let kernel = claim
                    .kernel
                    .as_deref()
                    .ok_or_else(|| missing_role_binding("prover claim kernel", &claim.symbol))?;
                claims.push(format!(
                        "    Stage2SumcheckClaimPlan {{ symbol: {}, stage: {}, domain: {}, num_rounds: {}, degree: {}, claim: {}, kernel: Some({}), relation: None, claim_value: {}, input_openings: STAGE2_SUMCHECK_CLAIM_{index}_INPUT_OPENINGS }},",
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
                        "    Stage2SumcheckClaimPlan {{ symbol: {}, stage: {}, domain: {}, num_rounds: {}, degree: {}, claim: {}, kernel: None, relation: Some({}), claim_value: {} }},",
                        rust_str(&claim.symbol),
                        rust_str(&claim.stage),
                        rust_str(&claim.domain),
                        claim.num_rounds,
                        claim.degree,
                        rust_str(&claim.claim),
                        super::plan_tokens::role_relation_kind_expr(
                            "Stage2",
                            &self.role,
                            relation
                        )?,
                        rust_str(&claim.claim_value)
                    ));
            }
        }
        let claims = claims.join("\n");
        push_format(
            &mut source,
            format_args!(
                "pub const STAGE2_SUMCHECK_CLAIMS: &[Stage2SumcheckClaimPlan] = &[\n{claims}\n];\n"
            ),
        );
        Ok(source)
    }

    fn emit_sumcheck_batch_constants(&self) -> String {
        if self.role == Role::Verifier {
            let mut source = String::new();
            for (index, batch) in self.batches.iter().enumerate() {
                source.push_str(&emit_usize_array(
                    &format!("STAGE2_SUMCHECK_BATCH_{index}_ROUND_SCHEDULE"),
                    &batch.round_schedule,
                ));
            }
            let batches = self
                .batches
                .iter()
                .enumerate()
                .map(|(index, batch)| {
                    format!(
                        "    Stage2SumcheckBatchPlan {{ symbol: {}, stage: {}, proof_slot: {}, policy: {}, count: {}, claim_operands: {}, claim_label: {}, round_label: {}, round_schedule: STAGE2_SUMCHECK_BATCH_{index}_ROUND_SCHEDULE }},",
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
                    "pub const STAGE2_SUMCHECK_BATCHES: &[Stage2SumcheckBatchPlan] = &[\n{batches}\n];\n"
                ),
            );
            return source;
        }

        let mut source = String::new();
        for (index, batch) in self.batches.iter().enumerate() {
            source.push_str(&emit_str_array(
                &format!("STAGE2_SUMCHECK_BATCH_{index}_ORDERED_CLAIMS"),
                &batch.ordered_claims,
            ));
            source.push_str(&emit_str_array(
                &format!("STAGE2_SUMCHECK_BATCH_{index}_CLAIM_OPERANDS"),
                &batch.claim_operands,
            ));
            source.push_str(&emit_usize_array(
                &format!("STAGE2_SUMCHECK_BATCH_{index}_ROUND_SCHEDULE"),
                &batch.round_schedule,
            ));
        }
        let batches = self
            .batches
            .iter()
            .enumerate()
            .map(|(index, batch)| {
                format!(
                    "    Stage2SumcheckBatchPlan {{ symbol: {}, stage: {}, proof_slot: {}, policy: {}, count: {}, ordered_claims: STAGE2_SUMCHECK_BATCH_{index}_ORDERED_CLAIMS, claim_operands: STAGE2_SUMCHECK_BATCH_{index}_CLAIM_OPERANDS, claim_label: {}, round_label: {}, round_schedule: STAGE2_SUMCHECK_BATCH_{index}_ROUND_SCHEDULE }},",
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
                "pub const STAGE2_SUMCHECK_BATCHES: &[Stage2SumcheckBatchPlan] = &[\n{batches}\n];\n"
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
                &format!("STAGE2_SUMCHECK_DRIVER_{index}_ROUND_SCHEDULE"),
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
                        "    Stage2SumcheckDriverPlan {{ symbol: {}, stage: {}, proof_slot: {}, kernel: Some({}), relation: None, batch: {}, policy: {}, round_schedule: STAGE2_SUMCHECK_DRIVER_{index}_ROUND_SCHEDULE, claim_label: {}, round_label: {}, num_rounds: {}, degree: {} }},",
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
                        "    Stage2SumcheckDriverPlan {{ symbol: {}, stage: {}, proof_slot: {}, kernel: None, relation: Some({}), batch: {}, policy: {}, round_schedule: STAGE2_SUMCHECK_DRIVER_{index}_ROUND_SCHEDULE, claim_label: {}, round_label: {}, num_rounds: {}, degree: {} }},",
                        rust_str(&driver.symbol),
                        rust_str(&driver.stage),
                        rust_str(&driver.proof_slot),
                        super::plan_tokens::role_relation_kind_expr(
                            "Stage2",
                            &self.role,
                            relation
                        )?,
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
                "pub const STAGE2_SUMCHECK_DRIVERS: &[Stage2SumcheckDriverPlan] = &[\n{drivers}\n];\n"
            ),
        );
        Ok(source)
    }

    fn emit_tail_constants(&self) -> Result<String, EmitError> {
        let mut source = String::new();
        source.push_str(&self.emit_sumcheck_instance_result_constants()?);
        source.push_str(&self.emit_sumcheck_eval_constants());
        source.push_str(&self.emit_point_slice_constants());
        source.push_str(&self.emit_point_concat_constants());
        source.push_str(&self.emit_opening_claim_constants()?);
        source.push_str(&self.emit_opening_batch_constants());
        Ok(source)
    }

    fn emit_sumcheck_instance_result_constants(&self) -> Result<String, EmitError> {
        let instances = self
            .instance_results
            .iter()
            .map(|instance| {
                Ok(format!(
                    "    Stage2SumcheckInstanceResultPlan {{ symbol: {}, source: {}, claim: {}, relation: {}, index: {}, point_arity: {}, num_rounds: {}, round_offset: {}, point_order: {}, degree: {} }},",
                    rust_str(&instance.symbol),
                    rust_str(&instance.source),
                    rust_str(&instance.claim),
                    super::plan_tokens::role_relation_kind_expr(
                        "Stage2",
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
            "pub const STAGE2_SUMCHECK_INSTANCE_RESULTS: &[Stage2SumcheckInstanceResultPlan] = &[\n{instances}\n];\n\n"
        ))
    }

    fn emit_sumcheck_eval_constants(&self) -> String {
        let evals = self
            .evals
            .iter()
            .map(|eval| {
                format!(
                    "    Stage2SumcheckEvalPlan {{ symbol: {}, source: {}, name: {}, index: {}, oracle: {} }},",
                    rust_str(&eval.symbol),
                    rust_str(&eval.source),
                    rust_str(&eval.name),
                    eval.index,
                    rust_str(&eval.oracle)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!("pub const STAGE2_SUMCHECK_EVALS: &[Stage2SumcheckEvalPlan] = &[\n{evals}\n];\n\n")
    }

    fn emit_point_slice_constants(&self) -> String {
        let slices = self
            .point_slices
            .iter()
            .map(|slice| {
                format!(
                    "    Stage2PointSlicePlan {{ symbol: {}, source: {}, offset: {}, length: {}, input: {} }},",
                    rust_str(&slice.symbol),
                    rust_str(&slice.source),
                    slice.offset,
                    slice.length,
                    rust_str(&slice.input)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!("pub const STAGE2_POINT_SLICES: &[Stage2PointSlicePlan] = &[\n{slices}\n];\n\n")
    }

    fn emit_point_concat_constants(&self) -> String {
        if self.role == Role::Verifier {
            let concats = self
                .point_concats
                .iter()
                .map(|concat| {
                    format!(
                        "    Stage2PointConcatPlan {{ symbol: {}, layout: {}, arity: {}, inputs: {} }},",
                        rust_str(&concat.symbol),
                        rust_str(&concat.layout),
                        concat.arity,
                        super::plan_tokens::rust_str_slice_expr(&concat.inputs)
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");
            return format!(
                "pub const STAGE2_POINT_CONCATS: &[Stage2PointConcatPlan] = &[\n{concats}\n];\n"
            );
        }

        let mut source = String::new();
        for (index, concat) in self.point_concats.iter().enumerate() {
            source.push_str(&emit_str_array(
                &format!("STAGE2_POINT_CONCAT_{index}_INPUTS"),
                &concat.inputs,
            ));
        }
        let concats = self
            .point_concats
            .iter()
            .enumerate()
            .map(|(index, concat)| {
                format!(
                    "    Stage2PointConcatPlan {{ symbol: {}, layout: {}, arity: {}, inputs: STAGE2_POINT_CONCAT_{index}_INPUTS }},",
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
                "pub const STAGE2_POINT_CONCATS: &[Stage2PointConcatPlan] = &[\n{concats}\n];\n"
            ),
        );
        source
    }

    fn emit_opening_claim_constants(&self) -> Result<String, EmitError> {
        let claims = self
            .opening_claims
            .iter()
            .map(|claim| {
                Ok(format!(
                    "    Stage2OpeningClaimPlan {{ symbol: {}, oracle: {}, domain: {}, point_arity: {}, claim_kind: {}, point_source: {}, eval_source: {} }},",
                    rust_str(&claim.symbol),
                    rust_str(&claim.oracle),
                    rust_str(&claim.domain),
                    claim.point_arity,
                    super::plan_tokens::role_claim_kind_expr("Stage2", &self.role, &claim.claim_kind)?,
                    rust_str(&claim.point_source),
                    rust_str(&claim.eval_source)
                ))
            })
            .collect::<Result<Vec<_>, EmitError>>()?
            .join("\n");
        Ok(format!(
            "pub const STAGE2_OPENING_CLAIMS: &[Stage2OpeningClaimPlan] = &[\n{claims}\n];\n\n"
        ))
    }

    fn emit_opening_batch_constants(&self) -> String {
        if self.role == Role::Verifier {
            let batches = self
                .opening_batches
                .iter()
                .map(|batch| {
                    format!(
                        "    Stage2OpeningBatchPlan {{ symbol: {}, stage: {}, proof_slot: {}, policy: {}, count: {}, ordered_claims: {}, claim_operands: {} }},",
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
                "pub const STAGE2_OPENING_BATCHES: &[Stage2OpeningBatchPlan] = &[\n{batches}\n];\n"
            );
        }

        let mut source = String::new();
        for (index, batch) in self.opening_batches.iter().enumerate() {
            source.push_str(&emit_str_array(
                &format!("STAGE2_OPENING_BATCH_{index}_ORDERED_CLAIMS"),
                &batch.ordered_claims,
            ));
            source.push_str(&emit_str_array(
                &format!("STAGE2_OPENING_BATCH_{index}_CLAIM_OPERANDS"),
                &batch.claim_operands,
            ));
        }
        let batches = self
            .opening_batches
            .iter()
            .enumerate()
            .map(|(index, batch)| {
                format!(
                    "    Stage2OpeningBatchPlan {{ symbol: {}, stage: {}, proof_slot: {}, policy: {}, count: {}, ordered_claims: STAGE2_OPENING_BATCH_{index}_ORDERED_CLAIMS, claim_operands: STAGE2_OPENING_BATCH_{index}_CLAIM_OPERANDS }},",
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
                "pub const STAGE2_OPENING_BATCHES: &[Stage2OpeningBatchPlan] = &[\n{batches}\n];\n"
            ),
        );
        source
    }

    fn emit_prover_entrypoint() -> &'static str {
        "pub fn execute_stage2_prover<E, T>(\n\
         \x20   executor: &mut E,\n\
         \x20   transcript: &mut T,\n\
         ) -> Result<Stage2ExecutionArtifacts<Fr>, Stage2KernelError>\n\
         where\n\
         \x20   E: Stage2KernelExecutor<Fr>,\n\
         \x20   T: Transcript<Challenge = Fr>,\n\
         {\n\
         \x20   execute_stage2_prover_with_program(&STAGE2_PROGRAM, executor, transcript)\n\
         }\n\
         \n\
         pub fn execute_stage2_prover_with_program<E, T>(\n\
         \x20   program: &'static Stage2CpuProgramPlan,\n\
         \x20   executor: &mut E,\n\
         \x20   transcript: &mut T,\n\
         ) -> Result<Stage2ExecutionArtifacts<Fr>, Stage2KernelError>\n\
         where\n\
         \x20   E: Stage2KernelExecutor<Fr>,\n\
         \x20   T: Transcript<Challenge = Fr>,\n\
         {\n\
         \x20   execute_stage2_program(program, Stage2ExecutionMode::Prover, executor, transcript)\n\
         }\n"
    }

    fn emit_verifier_entrypoint() -> &'static str {
        r#"const PRODUCT_VIRTUAL_UNISKIP_DOMAIN_START: i64 = -1;
const PRODUCT_VIRTUAL_UNISKIP_DOMAIN_SIZE: usize = 3;

pub fn verify_stage2<T>(
    proof: &Stage2Proof<Fr>,
    opening_inputs: &[Stage2OpeningInputValue<Fr>],
    ram: Option<&Stage2RamData<'_>>,
    transcript: &mut T,
) -> Result<Stage2ExecutionArtifacts<Fr>, VerifyStage2Error>
where
    T: Transcript<Challenge = Fr>,
{
    verify_stage2_with_program(&STAGE2_PROGRAM, proof, opening_inputs, ram, transcript)
}

pub fn verify_stage2_with_program<T>(
    program: &'static Stage2VerifierProgramPlan,
    proof: &Stage2Proof<Fr>,
    opening_inputs: &[Stage2OpeningInputValue<Fr>],
    ram: Option<&Stage2RamData<'_>>,
    transcript: &mut T,
) -> Result<Stage2ExecutionArtifacts<Fr>, VerifyStage2Error>
where
    T: Transcript<Challenge = Fr>,
{
    if proof.sumchecks.len() != program.drivers.len() {
        return Err(VerifyStage2Error::UnexpectedProofCount {
            expected: program.drivers.len(),
            got: proof.sumchecks.len(),
        });
    }
    let mut store = Stage2ValueStore::with_opening_inputs(program, opening_inputs)?;
    store.seed_constants(program);
    let mut artifacts = Stage2ExecutionArtifacts::default();
    if program.steps.is_empty() {
        for squeeze in program.transcript_squeezes {
            verify_stage2_squeeze(program, squeeze, &mut store, transcript, &mut artifacts)?;
        }
        for driver in program.drivers {
            verify_stage2_driver(program, driver, proof, ram, &mut store, transcript, &mut artifacts)?;
        }
    } else {
        for step in program.steps {
            match step.kind {
                Stage2ProgramStepKind::TranscriptSqueeze => {
                    let squeeze = find_plan(program.transcript_squeezes, step.symbol).ok_or(VerifyStage2Error::MissingValue {
                        symbol: step.symbol,
                    })?;
                    verify_stage2_squeeze(program, squeeze, &mut store, transcript, &mut artifacts)?;
                }
                Stage2ProgramStepKind::SumcheckDriver => {
                    let driver = find_plan(program.drivers, step.symbol).ok_or(VerifyStage2Error::MissingProof {
                        driver: step.symbol,
                    })?;
                    verify_stage2_driver(program, driver, proof, ram, &mut store, transcript, &mut artifacts)?;
                }
                Stage2ProgramStepKind::TranscriptAbsorbBytes => {
                    return Err(VerifyStage2Error::InvalidProof {
                        driver: step.symbol,
                        reason: "unsupported stage2 program step",
                    });
                }
            }
        }
    }
    artifacts
        .opening_batches
        .extend(program.opening_batches.iter());
    Ok(artifacts)
}

pub fn stage2_verifier_program() -> &'static Stage2VerifierProgramPlan {
    &STAGE2_PROGRAM
}

fn verify_stage2_squeeze<T>(
    program: &'static Stage2VerifierProgramPlan,
    squeeze: &'static Stage2TranscriptSqueezePlan,
    store: &mut Stage2ValueStore<Fr>,
    transcript: &mut T,
    artifacts: &mut Stage2ExecutionArtifacts<Fr>,
) -> Result<(), VerifyStage2Error>
where
    T: Transcript<Challenge = Fr>,
{
    let values = if matches!(
        squeeze.symbol,
        "stage2.product_virtual.tau_high" | "stage2.ram_output.r_address"
    ) {
        transcript.challenge_vector_optimized(squeeze.count)
    } else {
        transcript.challenge_vector(squeeze.count)
    };
    store.observe_challenge_vector(program, squeeze, &values)?;
    artifacts.challenge_vectors.push(Stage2ChallengeVector {
        symbol: squeeze.symbol,
        values,
    });
    Ok(())
}

fn verify_stage2_driver<T>(
    program: &'static Stage2VerifierProgramPlan,
    driver: &'static Stage2SumcheckDriverPlan,
    proof: &Stage2Proof<Fr>,
    ram: Option<&Stage2RamData<'_>>,
    store: &mut Stage2ValueStore<Fr>,
    transcript: &mut T,
    artifacts: &mut Stage2ExecutionArtifacts<Fr>,
) -> Result<(), VerifyStage2Error>
where
    T: Transcript<Challenge = Fr>,
{
    let proof = proof
        .sumchecks
        .get(artifacts.sumchecks.len())
        .ok_or(VerifyStage2Error::MissingProof {
            driver: driver.symbol,
        })?;
    let Some(relation) = driver.relation else {
        return Err(VerifyStage2Error::InvalidProof {
            driver: driver.symbol,
            reason: "missing driver relation",
        });
    };
    let output = match relation {
        Stage2RelationKind::Stage2ProductVirtualUniskip => {
            verify_product_virtual_uniskip(program, driver, proof, store, transcript)?
        }
        Stage2RelationKind::Stage2Batched => {
            verify_batched_stage2(program, driver, proof, ram, store, transcript)?
        }
        relation => return Err(VerifyStage2Error::UnsupportedRelation { relation }),
    };
    artifacts.sumchecks.push(output);
    Ok(())
}

fn verify_product_virtual_uniskip<T>(
    program: &'static Stage2VerifierProgramPlan,
    driver: &'static Stage2SumcheckDriverPlan,
    proof: &Stage2SumcheckOutput<Fr>,
    store: &mut Stage2ValueStore<Fr>,
    transcript: &mut T,
) -> Result<Stage2SumcheckOutput<Fr>, VerifyStage2Error>
where
    T: Transcript<Challenge = Fr>,
{
    validate_driver_symbol(driver, proof)?;
    let [poly] = proof.proof.round_polynomials.as_slice() else {
        return Err(VerifyStage2Error::InvalidProof {
            driver: driver.symbol,
            reason: "unexpected product uniskip round count",
        });
    };
    if polynomial_degree(poly) > driver.degree {
        return Err(VerifyStage2Error::InvalidProof {
            driver: driver.symbol,
            reason: "product uniskip polynomial exceeds degree bound",
        });
    }
    let batch = find_batch(program.batches, driver.symbol, driver.batch)?;
    let claim = batch_claims(program.claims, batch)?
        .into_iter()
        .next()
        .ok_or(VerifyStage2Error::MissingClaim {
            batch: batch.symbol,
            claim: "stage2.product_virtual.uniskip.input",
        })?;
    let input_claim = store.claim_value(program, claim)?;
    if !product_uniskip_sum_matches(poly, input_claim) {
        return Err(VerifyStage2Error::InvalidProof {
            driver: driver.symbol,
            reason: "product uniskip input claim mismatch",
        });
    }
    append_univariate_poly(transcript, driver.round_label, poly);
    let r0 = transcript.challenge_optimized();
    if !proof.point.is_empty() && proof.point != [r0] {
        return Err(VerifyStage2Error::InvalidProof {
            driver: driver.symbol,
            reason: "product uniskip point mismatch",
        });
    }
    let eval = poly.evaluate(r0);
    append_labeled_scalar(transcript, "opening_claim", &eval);
    let output = Stage2SumcheckOutput {
        driver: driver.symbol,
        point: vec![r0],
        evals: driver_evals(program, driver.symbol, eval),
        proof: proof.proof.clone(),
    };
    verify_named_evals(driver.symbol, &output.evals, &proof.evals)?;
    store.observe_sumcheck_output(program, &output)?;
    Ok(output)
}

fn verify_batched_stage2<T>(
    program: &'static Stage2VerifierProgramPlan,
    driver: &'static Stage2SumcheckDriverPlan,
    proof: &Stage2SumcheckOutput<Fr>,
    ram: Option<&Stage2RamData<'_>>,
    store: &mut Stage2ValueStore<Fr>,
    transcript: &mut T,
) -> Result<Stage2SumcheckOutput<Fr>, VerifyStage2Error>
where
    T: Transcript<Challenge = Fr>,
{
    validate_driver_symbol(driver, proof)?;
    let batch = find_batch(program.batches, driver.symbol, driver.batch)?;
    let claims = batch_claims(program.claims, batch)?;
    let input_claims = store.batch_claim_values(program, batch)?;
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
    let output = SumcheckVerifier::verify_optimized(&claim, &round_proofs, transcript)
        .map_err(|error| VerifyStage2Error::Sumcheck {
            driver: driver.symbol,
            error,
        })?;
    if !proof.point.is_empty() && proof.point != output.point {
        return Err(VerifyStage2Error::InvalidProof {
            driver: driver.symbol,
            reason: "batched point mismatch",
        });
    }
    let expected =
        expected_batched_output_claim(program, driver, &*store, &proof.evals, &output.point, &batching_coeffs, ram)?;
    if output.value != expected {
        return Err(VerifyStage2Error::InvalidProof {
            driver: driver.symbol,
            reason: "batched output claim mismatch",
        });
    }
    let verified = Stage2SumcheckOutput {
        driver: driver.symbol,
        point: output.point,
        evals: proof.evals.clone(),
        proof: proof.proof.clone(),
    };
    store.observe_sumcheck_output(program, &verified)?;
    bolt_verifier_runtime::append_opening_claims(
        program.opening_inputs,
        program.opening_claims,
        program.opening_batches,
        &mut store.0,
        transcript,
        &verified.evals,
        |batch, claim| VerifyStage2Error::MissingClaim { batch, claim },
        |symbol| VerifyStage2Error::MissingValue { symbol },
    )?;
    Ok(verified)
}

impl<F: Field> Stage2ValueStore<F> {
    fn with_opening_inputs(
        program: &'static Stage2VerifierProgramPlan,
        inputs: &[Stage2OpeningInputValue<F>],
    ) -> Result<Self, VerifyStage2Error> {
        Ok(Self(bolt_verifier_runtime::ValueStore::with_opening_inputs(
            inputs,
            program.opening_inputs,
        )?))
    }

    fn seed_constants(&mut self, program: &'static Stage2VerifierProgramPlan) {
        self.0.seed_constants(program.field_constants);
    }

    fn observe_challenge_vector(
        &mut self,
        program: &'static Stage2VerifierProgramPlan,
        plan: &'static Stage2TranscriptSqueezePlan,
        values: &[F],
    ) -> Result<(), VerifyStage2Error> {
        self.0.observe_challenge_vector(plan, values, |input, expected, actual| {
            VerifyStage2Error::InvalidInputLength { input, expected, actual }
        })?;
        self.evaluate_available_points(program)?;
        self.evaluate_available_field_exprs(program)?;
        Ok(())
    }

    fn observe_sumcheck_output(
        &mut self,
        program: &'static Stage2VerifierProgramPlan,
        output: &Stage2SumcheckOutput<F>,
    ) -> Result<(), VerifyStage2Error> {
        self.0.observe_sumcheck_output(
            program.instance_results,
            program.evals,
            output,
            |instance, mut point| {
                match instance.point_order {
                bolt_verifier_runtime::SumcheckPointOrder::AsIs => {}
                bolt_verifier_runtime::SumcheckPointOrder::Reverse => point.reverse(),
                _ => {
                    return Err(VerifyStage2Error::InvalidProof {
                        driver: output.driver,
                        reason: "unsupported point order",
                    });
                }
            }
                Ok(point)
            },
            |input, expected, actual| VerifyStage2Error::InvalidInputLength {
                input,
                expected,
                actual,
            },
            |symbol| VerifyStage2Error::MissingValue { symbol },
        )?;
        self.evaluate_available_points(program)?;
        self.evaluate_available_field_exprs(program)?;
        Ok(())
    }

    fn claim_value(
        &mut self,
        program: &'static Stage2VerifierProgramPlan,
        claim: &Stage2SumcheckClaimPlan,
    ) -> Result<F, VerifyStage2Error> {
        self.evaluate_available_field_exprs(program)?;
        self.scalar(claim.claim_value)
    }

    fn batch_claim_values(
        &mut self,
        program: &'static Stage2VerifierProgramPlan,
        batch: &Stage2SumcheckBatchPlan,
    ) -> Result<Vec<F>, VerifyStage2Error> {
        batch
            .claim_operands
            .iter()
            .copied()
            .map(|symbol| {
                let claim = find_plan(program.claims, symbol).ok_or(VerifyStage2Error::MissingClaim {
                    batch: batch.symbol,
                    claim: symbol,
                })?;
                self.claim_value(program, claim)
            })
            .collect()
    }

    fn evaluate_available_points(
        &mut self,
        program: &'static Stage2VerifierProgramPlan,
    ) -> Result<(), VerifyStage2Error> {
        self.0.evaluate_available_points(
            program.point_slices,
            program.point_concats,
            |input, expected, actual| VerifyStage2Error::InvalidInputLength {
                input,
                expected,
                actual,
            },
        )
    }

    fn evaluate_available_field_exprs(
        &mut self,
        program: &'static Stage2VerifierProgramPlan,
    ) -> Result<(), VerifyStage2Error> {
        self.0
            .evaluate_available_field_exprs(program.field_exprs, bolt_verifier_runtime::evaluate_field_expr)
            .map_err(VerifyStage2Error::from)
    }

    fn scalar(&self, symbol: &'static str) -> Result<F, VerifyStage2Error> {
        self.0
            .scalar_or(symbol, |symbol| VerifyStage2Error::MissingValue { symbol })
    }

    fn point(&self, symbol: &'static str) -> Result<&[F], VerifyStage2Error> {
        self.0
            .point_or(symbol, |symbol| VerifyStage2Error::MissingValue { symbol })
    }

    fn try_point(&self, symbol: &str) -> Option<&[F]> {
        self.0.try_point(symbol)
    }
}

fn expected_batched_output_claim(
    program: &'static Stage2VerifierProgramPlan,
    driver: &'static Stage2SumcheckDriverPlan,
    store: &Stage2ValueStore<Fr>,
    evals: &[Stage2NamedEval<Fr>],
    point: &[Fr],
    batching_coeffs: &[Fr],
    ram: Option<&Stage2RamData<'_>>,
) -> Result<Fr, VerifyStage2Error> {
    let batch = find_batch(program.batches, driver.symbol, driver.batch)?;
    let claims = batch_claims(program.claims, batch)?;
    let mut expected = Fr::from_u64(0);
    for (claim, coefficient) in claims.iter().zip(batching_coeffs) {
        let instance = program
            .instance_results
            .iter()
            .find(|instance| instance.claim == claim.symbol && instance.source == driver.symbol)
            .ok_or(VerifyStage2Error::MissingClaim {
                batch: batch.symbol,
                claim: claim.symbol,
            })?;
        let local_point = point
            .get(instance.round_offset..instance.round_offset + instance.num_rounds)
            .ok_or(VerifyStage2Error::InvalidInputLength {
                input: instance.symbol,
                expected: instance.round_offset + instance.num_rounds,
                actual: point.len(),
            })?;
        let value = match instance.relation {
            Stage2RelationKind::Stage2RamReadWrite => {
                expected_ram_read_write(
                    &STAGE2_RAM_READ_WRITE_OUTPUT,
                    store,
                    evals,
                    local_point,
                )?
            }
            Stage2RelationKind::Stage2ProductVirtualRemainder => {
                expected_product_remainder(store, evals, local_point)?
            }
            Stage2RelationKind::Stage2InstructionLookupClaimReduction => {
                expected_instruction_lookup(store, evals, local_point)?
            }
            Stage2RelationKind::Stage2RamRafEvaluation => expected_ram_raf(evals, local_point, ram)?,
            Stage2RelationKind::Stage2RamOutputCheck => {
                expected_ram_output(store, evals, local_point, ram)?
            }
            relation => return Err(VerifyStage2Error::UnsupportedRelation { relation }),
        };
        expected += *coefficient * value;
    }
    Ok(expected)
}

fn expected_ram_read_write(
    plan: &'static Stage2RamReadWriteOutputPlan,
    store: &Stage2ValueStore<Fr>,
    evals: &[Stage2NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage2Error> {
    let r_cycle_stage1 = store.point(plan.cycle_point)?;
    let log_t = r_cycle_stage1.len();
    let r_cycle = reverse_slice(&local_point[..log_t]);
    let eq_eval = EqPolynomial::<Fr>::mle(r_cycle_stage1, &r_cycle);
    let gamma = store.scalar(plan.gamma)?;
    let val = eval_by_name(evals, plan.val_eval)?;
    let ra = eval_by_name(evals, plan.ra_eval)?;
    let inc = eval_by_name(evals, plan.inc_eval)?;
    Ok(eq_eval * ra * (val + gamma * (val + inc)))
}

fn expected_product_remainder(
    store: &Stage2ValueStore<Fr>,
    evals: &[Stage2NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage2Error> {
    let tau_low = store.point("stage2.input.stage1.Product")?;
    let tau_high = store.scalar("stage2.product_virtual.tau_high")?;
    let r0 = *store
        .point("stage2.product_virtual.uniskip.sumcheck")?
        .first()
        .ok_or(VerifyStage2Error::MissingValue {
            symbol: "stage2.product_virtual.uniskip.sumcheck",
        })?;
    let r_tail = reverse_slice(local_point);
    let low = EqPolynomial::<Fr>::mle(tau_low, &r_tail);
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
        * eval_by_name(evals, "stage2.product_virtual.remainder.eval.LeftInstructionInput")?
        + weights[1] * eval_by_name(evals, "stage2.product_virtual.remainder.eval.LookupOutput")?
        + weights[2] * eval_by_name(evals, "stage2.product_virtual.remainder.eval.OpFlagJump")?;
    let right = weights[0]
        * eval_by_name(evals, "stage2.product_virtual.remainder.eval.RightInstructionInput")?
        + weights[1]
            * eval_by_name(evals, "stage2.product_virtual.remainder.eval.InstructionFlagBranch")?
        + weights[2]
            * (Fr::from_u64(1)
                - eval_by_name(evals, "stage2.product_virtual.remainder.eval.NextIsNoop")?);
    Ok(high * low * left * right)
}

fn expected_instruction_lookup(
    store: &Stage2ValueStore<Fr>,
    evals: &[Stage2NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage2Error> {
    let opening_point = reverse_slice(local_point);
    let r_spartan = store.point("stage2.input.stage1.LookupOutput")?;
    let eq_eval = EqPolynomial::<Fr>::mle(&opening_point, r_spartan);
    let gamma = store.scalar("stage2.instruction_lookup.gamma")?;
    let gamma2 = gamma.square();
    let gamma3 = gamma2 * gamma;
    let gamma4 = gamma2.square();
    let weighted = eval_by_name(
        evals,
        "stage2.instruction_lookup.claim_reduction.eval.LookupOutput",
    )? + gamma
        * eval_by_name(
            evals,
            "stage2.instruction_lookup.claim_reduction.eval.LeftLookupOperand",
        )?
        + gamma2
            * eval_by_name(
                evals,
                "stage2.instruction_lookup.claim_reduction.eval.RightLookupOperand",
            )?
        + gamma3
            * eval_by_name(
                evals,
                "stage2.instruction_lookup.claim_reduction.eval.LeftInstructionInput",
            )?
        + gamma4
            * eval_by_name(
                evals,
                "stage2.instruction_lookup.claim_reduction.eval.RightInstructionInput",
            )?;
    Ok(eq_eval * weighted)
}

fn expected_ram_raf(
    evals: &[Stage2NamedEval<Fr>],
    local_point: &[Fr],
    ram: Option<&Stage2RamData<'_>>,
) -> Result<Fr, VerifyStage2Error> {
    let ram = ram.ok_or(VerifyStage2Error::MissingRam {
        context: "stage2.ram.raf_evaluation",
    })?;
    let address = reverse_slice(local_point);
    let unmap = unmap_eval(ram.log_k, ram.start_address, &address);
    Ok(unmap * eval_by_name(evals, "stage2.ram_raf.eval.RamRa")?)
}

fn expected_ram_output(
    store: &Stage2ValueStore<Fr>,
    evals: &[Stage2NamedEval<Fr>],
    local_point: &[Fr],
    ram: Option<&Stage2RamData<'_>>,
) -> Result<Fr, VerifyStage2Error> {
    let ram = ram.ok_or(VerifyStage2Error::MissingRam {
        context: "stage2.ram.output_check",
    })?;
    let layout = ram.output_layout.ok_or(VerifyStage2Error::MissingRam {
        context: "stage2.ram.output_check.layout",
    })?;
    let r_address = store.point("stage2.ram_output.r_address")?;
    let opening_point = reverse_slice(local_point);
    let eq_eval = EqPolynomial::<Fr>::mle(r_address, &opening_point);
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

fn driver_evals(
    program: &'static Stage2VerifierProgramPlan,
    driver: &'static str,
    value: Fr,
) -> Vec<Stage2NamedEval<Fr>> {
    program
        .evals
        .iter()
        .filter(|eval| eval.source == driver)
        .map(|eval| Stage2NamedEval {
            name: eval.name,
            oracle: eval.oracle,
            value,
        })
        .collect()
}

fn verify_named_evals(
    driver: &'static str,
    expected: &[Stage2NamedEval<Fr>],
    actual: &[Stage2NamedEval<Fr>],
) -> Result<(), VerifyStage2Error> {
    if expected.len() != actual.len() {
        return Err(VerifyStage2Error::InvalidProof {
            driver,
            reason: "eval count mismatch",
        });
    }
    for (expected, actual) in expected.iter().zip(actual) {
        if expected.name != actual.name || expected.oracle != actual.oracle || expected.value != actual.value {
            return Err(VerifyStage2Error::InvalidProof {
                driver,
                reason: "eval mismatch",
            });
        }
    }
    Ok(())
}

fn validate_driver_symbol(
    driver: &'static Stage2SumcheckDriverPlan,
    proof: &Stage2SumcheckOutput<Fr>,
) -> Result<(), VerifyStage2Error> {
    if proof.driver == driver.symbol {
        Ok(())
    } else {
        Err(VerifyStage2Error::InvalidProof {
            driver: driver.symbol,
            reason: "driver symbol mismatch",
        })
    }
}

fn append_univariate_poly<T>(transcript: &mut T, label: &'static str, poly: &UnivariatePoly<Fr>)
where
    T: Transcript<Challenge = Fr>,
{
    transcript.append(&LabelWithCount(
        label.as_bytes(),
        poly.coefficients().len() as u64,
    ));
    for coefficient in poly.coefficients() {
        transcript.append(coefficient);
    }
}

fn product_uniskip_sum_matches(poly: &UnivariatePoly<Fr>, claim: Fr) -> bool {
    (0..PRODUCT_VIRTUAL_UNISKIP_DOMAIN_SIZE)
        .map(|index| {
            poly.evaluate(Fr::from_i64(
                PRODUCT_VIRTUAL_UNISKIP_DOMAIN_START + index as i64,
            ))
        })
        .sum::<Fr>()
        == claim
}

fn polynomial_degree(poly: &UnivariatePoly<Fr>) -> usize {
    poly.coefficients()
        .iter()
        .rposition(|coefficient| *coefficient != Fr::from_u64(0))
        .unwrap_or(0)
}

fn unmap_eval(log_k: usize, start_address: u64, point: &[Fr]) -> Fr {
    point
        .iter()
        .enumerate()
        .fold(Fr::from_u64(start_address), |acc, (index, value)| {
            acc + value.mul_pow_2(log_k - 1 - index).mul_u64(8)
        })
}

fn range_mask_eval(start: usize, end: usize, point: &[Fr]) -> Fr {
    eq_prefix_sum(end, point) - eq_prefix_sum(start, point)
}

fn sparse_final_ram_eval(values: &[u64], start: usize, end: usize, point: &[Fr]) -> Fr {
    values[start..end]
        .iter()
        .enumerate()
        .filter(|(_, value)| **value != 0)
        .map(|(offset, value)| Fr::from_u64(*value) * eq_eval_at_index(start + offset, point))
        .sum()
}

fn eq_prefix_sum(end: usize, point: &[Fr]) -> Fr {
    let domain_len = 1usize << point.len();
    if end >= domain_len {
        return Fr::from_u64(1);
    }
    let mut sum = Fr::from_u64(0);
    let mut prefix = Fr::from_u64(1);
    for (bit, r) in point.iter().enumerate() {
        let mask = 1usize << (point.len() - 1 - bit);
        if end & mask == 0 {
            prefix *= Fr::from_u64(1) - *r;
        } else {
            sum += prefix * (Fr::from_u64(1) - *r);
            prefix *= *r;
        }
    }
    sum
}

fn eq_eval_at_index(index: usize, point: &[Fr]) -> Fr {
    point.iter().enumerate().fold(Fr::from_u64(1), |acc, (bit, r)| {
        let mask = 1usize << (point.len() - 1 - bit);
        if index & mask == 0 {
            acc * (Fr::from_u64(1) - *r)
        } else {
            acc * *r
        }
    })
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

fn rust_target_plan_error(error: RustTargetPlanError) -> EmitError {
    EmitError::new(error.to_string())
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
