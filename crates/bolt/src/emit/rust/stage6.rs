use std::collections::{BTreeMap, BTreeSet};

use melior::ir::block::BlockLike;
use melior::ir::operation::{OperationLike, OperationResult};
use melior::ir::{Attribute, OperationRef};

use crate::emit::rust::{push_format, EmitError, RustSourceFile};
use crate::ir::{string_attribute_value, symbol_attribute_value, BoltModule, Cpu, Role};
use crate::schema::verify_cpu_schema;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6CpuProgram {
    pub role: Role,
    pub params: Stage6Params,
    pub steps: Vec<Stage6ProgramStepPlan>,
    pub transcript_squeezes: Vec<Stage6TranscriptSqueezePlan>,
    pub transcript_absorb_bytes: Vec<Stage6TranscriptAbsorbBytesPlan>,
    pub opening_inputs: Vec<Stage6OpeningInputPlan>,
    pub field_constants: Vec<Stage6FieldConstantPlan>,
    pub field_exprs: Vec<Stage6FieldExprPlan>,
    pub kernels: Vec<Stage6KernelPlan>,
    pub claims: Vec<Stage6SumcheckClaimPlan>,
    pub batches: Vec<Stage6SumcheckBatchPlan>,
    pub drivers: Vec<Stage6SumcheckDriverPlan>,
    pub instance_results: Vec<Stage6SumcheckInstanceResultPlan>,
    pub evals: Vec<Stage6SumcheckEvalPlan>,
    pub point_zeros: Vec<Stage6PointZeroPlan>,
    pub point_slices: Vec<Stage6PointSlicePlan>,
    pub point_concats: Vec<Stage6PointConcatPlan>,
    pub opening_claims: Vec<Stage6OpeningClaimPlan>,
    pub opening_equalities: Vec<Stage6OpeningClaimEqualityPlan>,
    pub opening_batches: Vec<Stage6OpeningBatchPlan>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6Params {
    pub field: String,
    pub pcs: String,
    pub transcript: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6KernelPlan {
    pub symbol: String,
    pub relation: String,
    pub kind: String,
    pub backend: String,
    pub abi: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6TranscriptSqueezePlan {
    pub symbol: String,
    pub label: String,
    pub kind: String,
    pub count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6TranscriptAbsorbBytesPlan {
    pub symbol: String,
    pub label: String,
    pub payload: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6ProgramStepPlan {
    pub kind: String,
    pub symbol: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6OpeningInputPlan {
    pub symbol: String,
    pub source_stage: String,
    pub source_claim: String,
    pub oracle: String,
    pub domain: String,
    pub point_arity: usize,
    pub claim_kind: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6FieldConstantPlan {
    pub symbol: String,
    pub field: String,
    pub value: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6FieldExprPlan {
    pub symbol: String,
    pub kind: String,
    pub formula: String,
    pub operand_names: Vec<String>,
    pub operands: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6SumcheckClaimPlan {
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
pub struct Stage6SumcheckBatchPlan {
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
pub struct Stage6SumcheckDriverPlan {
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
pub struct Stage6SumcheckInstanceResultPlan {
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
pub struct Stage6SumcheckEvalPlan {
    pub symbol: String,
    pub source: String,
    pub name: String,
    pub index: usize,
    pub oracle: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6PointZeroPlan {
    pub symbol: String,
    pub field: String,
    pub arity: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6PointSlicePlan {
    pub symbol: String,
    pub source: String,
    pub offset: usize,
    pub length: usize,
    pub input: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6PointConcatPlan {
    pub symbol: String,
    pub layout: String,
    pub arity: usize,
    pub inputs: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6OpeningClaimPlan {
    pub symbol: String,
    pub oracle: String,
    pub domain: String,
    pub point_arity: usize,
    pub claim_kind: String,
    pub point_source: String,
    pub eval_source: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6OpeningClaimEqualityPlan {
    pub symbol: String,
    pub mode: String,
    pub lhs: String,
    pub rhs: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6OpeningBatchPlan {
    pub symbol: String,
    pub stage: String,
    pub proof_slot: String,
    pub policy: String,
    pub count: usize,
    pub ordered_claims: Vec<String>,
    pub claim_operands: Vec<String>,
}

pub fn stage6_cpu_program(module: &BoltModule<'_, Cpu>) -> Result<Stage6CpuProgram, EmitError> {
    verify_cpu_schema(module)?;
    let program = Stage6CpuProgram::from_module(module)?;
    program.verify_supported_target()?;
    Ok(program)
}

pub fn emit_stage6_rust(module: &BoltModule<'_, Cpu>) -> Result<RustSourceFile, EmitError> {
    let program = stage6_cpu_program(module)?;

    Ok(RustSourceFile {
        filename: program.filename().to_owned(),
        source: program.emit_source(),
    })
}

impl Stage6CpuProgram {
    fn from_module(module: &BoltModule<'_, Cpu>) -> Result<Self, EmitError> {
        let mut params = None;
        let mut steps = Vec::new();
        let mut transcript_squeezes = Vec::new();
        let mut transcript_absorb_bytes = Vec::new();
        let mut opening_inputs = Vec::new();
        let mut field_constants = Vec::new();
        let mut field_exprs = Vec::new();
        let mut kernels = Vec::new();
        let mut claims = Vec::new();
        let mut batches = Vec::new();
        let mut drivers = Vec::new();
        let mut instance_results = Vec::new();
        let mut evals = Vec::new();
        let mut point_zeros = Vec::new();
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
                    params = Some(Stage6Params {
                        field: symbol_attr(op, "field")?,
                        pcs: symbol_attr(op, "pcs")?,
                        transcript: symbol_attr(op, "transcript")?,
                    });
                }
                "cpu.kernel" => {
                    kernels.push(Stage6KernelPlan {
                        symbol: string_attr(op, "sym_name")?,
                        relation: symbol_attr(op, "relation")?,
                        kind: string_attr(op, "kind")?,
                        backend: string_attr(op, "backend")?,
                        abi: string_attr(op, "abi")?,
                    });
                }
                "cpu.transcript_squeeze" => {
                    let symbol = string_attr(op, "sym_name")?;
                    steps.push(Stage6ProgramStepPlan {
                        kind: "transcript_squeeze".to_owned(),
                        symbol: symbol.clone(),
                    });
                    transcript_squeezes.push(Stage6TranscriptSqueezePlan {
                        symbol,
                        label: string_attr(op, "label")?,
                        kind: string_attr(op, "kind")?,
                        count: int_attr(op, "count")?,
                    });
                }
                "cpu.transcript_absorb_bytes" => {
                    let symbol = string_attr(op, "sym_name")?;
                    steps.push(Stage6ProgramStepPlan {
                        kind: "transcript_absorb_bytes".to_owned(),
                        symbol: symbol.clone(),
                    });
                    transcript_absorb_bytes.push(Stage6TranscriptAbsorbBytesPlan {
                        symbol,
                        label: string_attr(op, "label")?,
                        payload: string_attr(op, "payload")?,
                    });
                }
                "cpu.opening_input" => {
                    opening_inputs.push(Stage6OpeningInputPlan {
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
                    field_constants.push(Stage6FieldConstantPlan {
                        symbol: string_attr(op, "sym_name")?,
                        field: symbol_attr(op, "field")?,
                        value: int_attr(op, "value")?,
                    });
                }
                "cpu.field_zero" => {
                    field_constants.push(Stage6FieldConstantPlan {
                        symbol: string_attr(op, "sym_name")?,
                        field: symbol_attr(op, "field")?,
                        value: 0,
                    });
                }
                "cpu.field_one" => {
                    field_constants.push(Stage6FieldConstantPlan {
                        symbol: string_attr(op, "sym_name")?,
                        field: symbol_attr(op, "field")?,
                        value: 1,
                    });
                }
                "cpu.field_add" | "cpu.field_sub" | "cpu.field_mul" | "cpu.field_neg" => {
                    let operands = operand_symbols(op, 0)?;
                    field_exprs.push(Stage6FieldExprPlan {
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
                    field_exprs.push(Stage6FieldExprPlan {
                        symbol: string_attr(op, "sym_name")?,
                        kind: "op".to_owned(),
                        formula: format!("field.pow:{exponent}"),
                        operand_names: operands.clone(),
                        operands,
                    });
                }
                "cpu.sumcheck_claim" => {
                    claims.push(Stage6SumcheckClaimPlan {
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
                    claims.push(Stage6SumcheckClaimPlan {
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
                    batches.push(Stage6SumcheckBatchPlan {
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
                    steps.push(Stage6ProgramStepPlan {
                        kind: "sumcheck_driver".to_owned(),
                        symbol: symbol.clone(),
                    });
                    drivers.push(Stage6SumcheckDriverPlan {
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
                    steps.push(Stage6ProgramStepPlan {
                        kind: "sumcheck_driver".to_owned(),
                        symbol: symbol.clone(),
                    });
                    drivers.push(Stage6SumcheckDriverPlan {
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
                    instance_results.push(Stage6SumcheckInstanceResultPlan {
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
                    evals.push(Stage6SumcheckEvalPlan {
                        symbol: string_attr(op, "sym_name")?,
                        source: symbol_attr(op, "source")?,
                        name: symbol_attr(op, "name")?,
                        index: int_attr(op, "index")?,
                        oracle: symbol_attr(op, "oracle")?,
                    });
                }
                "cpu.point_zero" => {
                    point_zeros.push(Stage6PointZeroPlan {
                        symbol: string_attr(op, "sym_name")?,
                        field: symbol_attr(op, "field")?,
                        arity: int_attr(op, "arity")?,
                    });
                }
                "cpu.point_slice" => {
                    point_slices.push(Stage6PointSlicePlan {
                        symbol: string_attr(op, "sym_name")?,
                        source: symbol_attr(op, "source")?,
                        offset: int_attr(op, "offset")?,
                        length: int_attr(op, "length")?,
                        input: operand_symbol(op, 0)?,
                    });
                }
                "cpu.point_concat" => {
                    point_concats.push(Stage6PointConcatPlan {
                        symbol: string_attr(op, "sym_name")?,
                        layout: string_attr(op, "layout")?,
                        arity: int_attr(op, "arity")?,
                        inputs: operand_symbols(op, 0)?,
                    });
                }
                "cpu.opening_claim" => {
                    opening_claims.push(Stage6OpeningClaimPlan {
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
                    opening_equalities.push(Stage6OpeningClaimEqualityPlan {
                        symbol: string_attr(op, "sym_name")?,
                        mode: string_attr(op, "mode")?,
                        lhs: operand_symbol(op, 0)?,
                        rhs: operand_symbol(op, 1)?,
                    });
                }
                "cpu.opening_batch" => {
                    opening_batches.push(Stage6OpeningBatchPlan {
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
            transcript_absorb_bytes,
            opening_inputs,
            field_constants,
            field_exprs,
            kernels,
            claims,
            batches,
            drivers,
            instance_results,
            evals,
            point_zeros,
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
        self.verify_opening_flow()
    }

    fn verify_transcript_steps(&self) -> Result<(), EmitError> {
        for squeeze in &self.transcript_squeezes {
            if !matches!(
                squeeze.kind.as_str(),
                "challenge_scalar" | "challenge_vector"
            ) {
                return Err(EmitError::new(format!(
                    "stage6 transcript squeeze @{} has unsupported kind `{}`",
                    squeeze.symbol, squeeze.kind
                )));
            }
            if squeeze.count == 0 {
                return Err(EmitError::new(format!(
                    "stage6 transcript squeeze @{} has zero count",
                    squeeze.symbol
                )));
            }
        }
        for absorb in &self.transcript_absorb_bytes {
            if absorb.label.is_empty() {
                return Err(EmitError::new(format!(
                    "stage6 transcript byte absorb @{} has empty label",
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
                    "stage6 kernel @{} targets unsupported backend `{}`",
                    kernel.symbol, kernel.backend
                )));
            }
            if kernel.kind != "sumcheck" {
                return Err(EmitError::new(format!(
                    "stage6 kernel @{} has unsupported kind `{}`",
                    kernel.symbol, kernel.kind
                )));
            }
            let expected_abi = match kernel.relation.as_str() {
                "jolt.stage6.bytecode_read_raf" => "jolt_stage6_bytecode_read_raf",
                "jolt.stage6.booleanity" => "jolt_stage6_booleanity",
                "jolt.stage6.hamming_booleanity" => "jolt_stage6_hamming_booleanity",
                "jolt.stage6.ram_ra_virtual" => "jolt_stage6_ram_ra_virtual",
                "jolt.stage6.instruction_ra_virtual" => "jolt_stage6_instruction_ra_virtual",
                "jolt.stage6.inc_claim_reduction" => "jolt_stage6_inc_claim_reduction",
                "jolt.stage6.batched" => "jolt_stage6_batched",
                _ => {
                    return Err(EmitError::new(format!(
                        "unsupported stage6 kernel relation @{}",
                        kernel.relation
                    )));
                }
            };
            if kernel.abi != expected_abi {
                return Err(EmitError::new(format!(
                    "stage6 kernel @{} ABI `{}` does not match relation @{}",
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
                "verifier stage6 program must not contain kernels",
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
        point_sources.extend(symbols(self.point_zeros.iter().map(|zero| &zero.symbol)));
        point_sources.extend(symbols(self.point_slices.iter().map(|slice| &slice.symbol)));
        point_sources.extend(symbols(
            self.point_concats.iter().map(|concat| &concat.symbol),
        ));
        for zero in &self.point_zeros {
            require_supported_symbol("point zero field", &zero.field, "bn254_fr")?;
        }
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

    fn filename(&self) -> &'static str {
        match self.role {
            Role::Prover => "prove_stage6.rs",
            Role::Verifier => "verify_stage6.rs",
        }
    }

    fn emit_source(&self) -> String {
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
        source.push_str(&self.emit_constants());
        source.push('\n');
        source.push_str(self.emit_entrypoint());
        source
    }

    fn emit_prover_imports() -> &'static str {
        "use jolt_field::Fr;\n\
         use jolt_kernels::stage6::{execute_stage6_program, Stage6CpuProgramPlan, Stage6ExecutionArtifacts, Stage6ExecutionMode, Stage6FieldConstantPlan, Stage6FieldExprPlan, Stage6KernelError, Stage6KernelExecutor, Stage6KernelPlan, Stage6OpeningBatchPlan, Stage6OpeningClaimEqualityPlan, Stage6OpeningClaimPlan, Stage6OpeningInputPlan, Stage6Params, Stage6PointConcatPlan, Stage6PointSlicePlan, Stage6PointZeroPlan, Stage6ProgramStepPlan, Stage6SumcheckBatchPlan, Stage6SumcheckClaimPlan, Stage6SumcheckDriverPlan, Stage6SumcheckEvalPlan, Stage6SumcheckInstanceResultPlan, Stage6TranscriptAbsorbBytesPlan, Stage6TranscriptSqueezePlan};\n\
         use jolt_transcript::{Blake2bTranscript, Transcript};"
    }

    fn emit_prover_types() -> &'static str {
        "pub type DefaultStage6Transcript = Blake2bTranscript<Fr>;\n"
    }

    fn emit_verifier_imports() -> &'static str {
        "use jolt_field::{Field, Fr};\n\
         use jolt_lookup_tables::LookupTableKind;\n\
         use jolt_poly::EqPolynomial;\n\
         use jolt_sumcheck::{CompressedLabeledRoundPoly, SumcheckClaim, SumcheckError, SumcheckProof, SumcheckVerifier};\n\
         use jolt_transcript::{Blake2bTranscript, Label, LabelWithCount, Transcript};"
    }

    fn emit_types() -> &'static str {
        r#"#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage6Params {
    pub field: &'static str,
    pub pcs: &'static str,
    pub transcript: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage6KernelPlan {
    pub symbol: &'static str,
    pub relation: &'static str,
    pub kind: &'static str,
    pub backend: &'static str,
    pub abi: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage6TranscriptSqueezePlan {
    pub symbol: &'static str,
    pub label: &'static str,
    pub kind: &'static str,
    pub count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage6TranscriptAbsorbBytesPlan {
    pub symbol: &'static str,
    pub label: &'static str,
    pub payload: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage6ProgramStepPlan {
    pub kind: &'static str,
    pub symbol: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage6OpeningInputPlan {
    pub symbol: &'static str,
    pub source_stage: &'static str,
    pub source_claim: &'static str,
    pub oracle: &'static str,
    pub domain: &'static str,
    pub point_arity: usize,
    pub claim_kind: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage6FieldConstantPlan {
    pub symbol: &'static str,
    pub field: &'static str,
    pub value: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage6FieldExprPlan {
    pub symbol: &'static str,
    pub kind: &'static str,
    pub formula: &'static str,
    pub operand_names: &'static [&'static str],
    pub operands: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage6SumcheckClaimPlan {
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
pub struct Stage6SumcheckBatchPlan {
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
pub struct Stage6SumcheckDriverPlan {
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
pub struct Stage6SumcheckInstanceResultPlan {
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
pub struct Stage6SumcheckEvalPlan {
    pub symbol: &'static str,
    pub source: &'static str,
    pub name: &'static str,
    pub index: usize,
    pub oracle: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage6PointZeroPlan {
    pub symbol: &'static str,
    pub field: &'static str,
    pub arity: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage6PointSlicePlan {
    pub symbol: &'static str,
    pub source: &'static str,
    pub offset: usize,
    pub length: usize,
    pub input: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage6PointConcatPlan {
    pub symbol: &'static str,
    pub layout: &'static str,
    pub arity: usize,
    pub inputs: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage6OpeningClaimPlan {
    pub symbol: &'static str,
    pub oracle: &'static str,
    pub domain: &'static str,
    pub point_arity: usize,
    pub claim_kind: &'static str,
    pub point_source: &'static str,
    pub eval_source: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage6OpeningClaimEqualityPlan {
    pub symbol: &'static str,
    pub mode: &'static str,
    pub lhs: &'static str,
    pub rhs: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage6OpeningBatchPlan {
    pub symbol: &'static str,
    pub stage: &'static str,
    pub proof_slot: &'static str,
    pub policy: &'static str,
    pub count: usize,
    pub ordered_claims: &'static [&'static str],
    pub claim_operands: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage6CpuProgramPlan {
    pub role: &'static str,
    pub params: Stage6Params,
    pub steps: &'static [Stage6ProgramStepPlan],
    pub transcript_squeezes: &'static [Stage6TranscriptSqueezePlan],
    pub transcript_absorb_bytes: &'static [Stage6TranscriptAbsorbBytesPlan],
    pub opening_inputs: &'static [Stage6OpeningInputPlan],
    pub field_constants: &'static [Stage6FieldConstantPlan],
    pub field_exprs: &'static [Stage6FieldExprPlan],
    pub kernels: &'static [Stage6KernelPlan],
    pub claims: &'static [Stage6SumcheckClaimPlan],
    pub batches: &'static [Stage6SumcheckBatchPlan],
    pub drivers: &'static [Stage6SumcheckDriverPlan],
    pub instance_results: &'static [Stage6SumcheckInstanceResultPlan],
    pub evals: &'static [Stage6SumcheckEvalPlan],
    pub point_zeros: &'static [Stage6PointZeroPlan],
    pub point_slices: &'static [Stage6PointSlicePlan],
    pub point_concats: &'static [Stage6PointConcatPlan],
    pub opening_claims: &'static [Stage6OpeningClaimPlan],
    pub opening_equalities: &'static [Stage6OpeningClaimEqualityPlan],
    pub opening_batches: &'static [Stage6OpeningBatchPlan],
}
"#
    }

    fn emit_verifier_types() -> String {
        let mut source = Self::emit_types().to_owned();
        source.push_str(
            r#"
pub type DefaultStage6Transcript = Blake2bTranscript<Fr>;
pub type Stage6VerifierProgramPlan = Stage6CpuProgramPlan;

#[derive(Clone, Debug)]
pub struct Stage6NamedEval<F: Field> {
    pub name: &'static str,
    pub oracle: &'static str,
    pub value: F,
}

#[derive(Clone, Debug)]
pub struct Stage6SumcheckOutput<F: Field> {
    pub driver: &'static str,
    pub point: Vec<F>,
    pub evals: Vec<Stage6NamedEval<F>>,
    pub proof: SumcheckProof<F>,
}

#[derive(Clone, Debug)]
pub struct Stage6ChallengeVector<F: Field> {
    pub symbol: &'static str,
    pub values: Vec<F>,
}

#[derive(Clone, Debug)]
pub struct Stage6ExecutionArtifacts<F: Field> {
    pub challenge_vectors: Vec<Stage6ChallengeVector<F>>,
    pub sumchecks: Vec<Stage6SumcheckOutput<F>>,
    pub opening_batches: Vec<&'static Stage6OpeningBatchPlan>,
}

impl<F: Field> Default for Stage6ExecutionArtifacts<F> {
    fn default() -> Self {
        Self {
            challenge_vectors: Vec::new(),
            sumchecks: Vec::new(),
            opening_batches: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Stage6Proof<F: Field> {
    pub sumchecks: Vec<Stage6SumcheckOutput<F>>,
}

#[derive(Clone, Debug)]
pub struct Stage6OpeningInputValue<F: Field> {
    pub symbol: &'static str,
    pub point: Vec<F>,
    pub eval: F,
}

#[derive(Clone, Debug)]
pub struct Stage6BytecodeEntry {
    pub address: Fr,
    pub imm: Fr,
    pub circuit_flags: [bool; 14],
    pub rd: Option<usize>,
    pub rs1: Option<usize>,
    pub rs2: Option<usize>,
    pub lookup_table: Option<usize>,
    pub is_interleaved: bool,
    pub is_branch: bool,
    pub left_is_rs1: bool,
    pub left_is_pc: bool,
    pub right_is_rs2: bool,
    pub right_is_imm: bool,
    pub is_noop: bool,
}

#[derive(Clone, Debug)]
pub struct Stage6BytecodeReadRafData {
    pub entries: Vec<Stage6BytecodeEntry>,
    pub entry_bytecode_index: usize,
    pub num_lookup_tables: usize,
}

#[derive(Clone, Debug)]
pub struct Stage6VerifierData {
    pub bytecode_read_raf: Option<Stage6BytecodeReadRafData>,
}

#[derive(Clone, Debug, Default)]
struct Stage6ValueStore<F: Field> {
    scalars: Vec<(&'static str, F)>,
    points: Vec<(&'static str, Vec<F>)>,
}

#[derive(Debug)]
pub enum VerifyStage6Error {
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
"#,
        );
        source
    }

    fn emit_constants(&self) -> String {
        let mut source = self.emit_shared_constants();
        source.push_str(&self.emit_kernel_constants());
        source.push_str(&self.emit_sumcheck_claim_constants());
        source.push_str(&self.emit_sumcheck_batch_constants());
        source.push_str(&self.emit_sumcheck_driver_constants());
        source.push_str(&self.emit_tail_constants());
        push_format(
            &mut source,
            format_args!(
                "pub const STAGE6_PROGRAM: {} = Stage6CpuProgramPlan {{\n\
                 \x20   role: {},\n\
                 \x20   params: STAGE6_PARAMS,\n\
                 \x20   steps: STAGE6_PROGRAM_STEPS,\n\
                 \x20   transcript_squeezes: STAGE6_TRANSCRIPT_SQUEEZES,\n\
                 \x20   transcript_absorb_bytes: STAGE6_TRANSCRIPT_ABSORB_BYTES,\n\
                 \x20   opening_inputs: STAGE6_OPENING_INPUTS,\n\
                 \x20   field_constants: STAGE6_FIELD_CONSTANTS,\n\
                 \x20   field_exprs: STAGE6_FIELD_EXPRS,\n\
                 \x20   kernels: STAGE6_KERNELS,\n\
                 \x20   claims: STAGE6_SUMCHECK_CLAIMS,\n\
                 \x20   batches: STAGE6_SUMCHECK_BATCHES,\n\
                 \x20   drivers: STAGE6_SUMCHECK_DRIVERS,\n\
                 \x20   instance_results: STAGE6_SUMCHECK_INSTANCE_RESULTS,\n\
                 \x20   evals: STAGE6_SUMCHECK_EVALS,\n\
                 \x20   point_zeros: STAGE6_POINT_ZEROS,\n\
                 \x20   point_slices: STAGE6_POINT_SLICES,\n\
                 \x20   point_concats: STAGE6_POINT_CONCATS,\n\
                 \x20   opening_claims: STAGE6_OPENING_CLAIMS,\n\
                 \x20   opening_equalities: STAGE6_OPENING_EQUALITIES,\n\
                 \x20   opening_batches: STAGE6_OPENING_BATCHES,\n\
                 }};\n",
                self.program_plan_type(),
                rust_str(self.role_label())
            ),
        );
        source
    }

    fn emit_shared_constants(&self) -> String {
        let mut source = String::new();
        push_format(
            &mut source,
            format_args!(
                "pub const STAGE6_PARAMS: Stage6Params = Stage6Params {{\n\
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
        source.push_str(&self.emit_transcript_absorb_bytes_constants());
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
                    "    Stage6ProgramStepPlan {{ kind: {}, symbol: {} }},",
                    rust_str(&step.kind),
                    rust_str(&step.symbol),
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!("pub const STAGE6_PROGRAM_STEPS: &[Stage6ProgramStepPlan] = &[\n{steps}\n];\n\n")
    }

    fn emit_transcript_squeeze_constants(&self) -> String {
        let squeezes = self
            .transcript_squeezes
            .iter()
            .map(|squeeze| {
                format!(
                    "    Stage6TranscriptSqueezePlan {{ symbol: {}, label: {}, kind: {}, count: {} }},",
                    rust_str(&squeeze.symbol),
                    rust_str(&squeeze.label),
                    rust_str(&squeeze.kind),
                    squeeze.count,
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "pub const STAGE6_TRANSCRIPT_SQUEEZES: &[Stage6TranscriptSqueezePlan] = &[\n{squeezes}\n];\n\n"
        )
    }

    fn emit_transcript_absorb_bytes_constants(&self) -> String {
        let absorbs = self
            .transcript_absorb_bytes
            .iter()
            .map(|absorb| {
                format!(
                    "    Stage6TranscriptAbsorbBytesPlan {{ symbol: {}, label: {}, payload: {} }},",
                    rust_str(&absorb.symbol),
                    rust_str(&absorb.label),
                    rust_str(&absorb.payload),
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "pub const STAGE6_TRANSCRIPT_ABSORB_BYTES: &[Stage6TranscriptAbsorbBytesPlan] = &[\n{absorbs}\n];\n\n"
        )
    }

    fn emit_opening_input_constants(&self) -> String {
        let inputs = self
            .opening_inputs
            .iter()
            .map(|input| {
                format!(
                    "    Stage6OpeningInputPlan {{ symbol: {}, source_stage: {}, source_claim: {}, oracle: {}, domain: {}, point_arity: {}, claim_kind: {} }},",
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
        format!("pub const STAGE6_OPENING_INPUTS: &[Stage6OpeningInputPlan] = &[\n{inputs}\n];\n\n")
    }

    fn emit_field_constant_constants(&self) -> String {
        let constants = self
            .field_constants
            .iter()
            .map(|constant| {
                format!(
                    "    Stage6FieldConstantPlan {{ symbol: {}, field: {}, value: {} }},",
                    rust_str(&constant.symbol),
                    rust_str(&constant.field),
                    constant.value
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "pub const STAGE6_FIELD_CONSTANTS: &[Stage6FieldConstantPlan] = &[\n{constants}\n];\n\n"
        )
    }

    fn emit_field_expr_constants(&self) -> String {
        let mut source = String::new();
        for (index, expr) in self.field_exprs.iter().enumerate() {
            source.push_str(&emit_str_array(
                &format!("STAGE6_FIELD_EXPR_{index}_OPERAND_NAMES"),
                &expr.operand_names,
            ));
            source.push_str(&emit_str_array(
                &format!("STAGE6_FIELD_EXPR_{index}_OPERANDS"),
                &expr.operands,
            ));
        }
        let exprs = self
            .field_exprs
            .iter()
            .enumerate()
            .map(|(index, expr)| {
                format!(
                    "    Stage6FieldExprPlan {{ symbol: {}, kind: {}, formula: {}, operand_names: STAGE6_FIELD_EXPR_{index}_OPERAND_NAMES, operands: STAGE6_FIELD_EXPR_{index}_OPERANDS }},",
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
                "pub const STAGE6_FIELD_EXPRS: &[Stage6FieldExprPlan] = &[\n{exprs}\n];\n"
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
                    "    Stage6KernelPlan {{ symbol: {}, relation: {}, kind: {}, backend: {}, abi: {} }},",
                    rust_str(&kernel.symbol),
                    rust_str(&kernel.relation),
                    rust_str(&kernel.kind),
                    rust_str(&kernel.backend),
                    rust_str(&kernel.abi)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!("pub const STAGE6_KERNELS: &[Stage6KernelPlan] = &[\n{kernels}\n];\n\n")
    }

    fn emit_sumcheck_claim_constants(&self) -> String {
        let mut source = String::new();
        for (index, claim) in self.claims.iter().enumerate() {
            source.push_str(&emit_str_array(
                &format!("STAGE6_SUMCHECK_CLAIM_{index}_INPUT_OPENINGS"),
                &claim.input_openings,
            ));
        }
        let claims = self
            .claims
            .iter()
            .enumerate()
            .map(|(index, claim)| {
                format!(
                    "    Stage6SumcheckClaimPlan {{ symbol: {}, stage: {}, domain: {}, num_rounds: {}, degree: {}, claim: {}, kernel: {}, relation: {}, claim_value: {}, input_openings: STAGE6_SUMCHECK_CLAIM_{index}_INPUT_OPENINGS }},",
                    rust_str(&claim.symbol),
                    rust_str(&claim.stage),
                    rust_str(&claim.domain),
                    claim.num_rounds,
                    claim.degree,
                    rust_str(&claim.claim),
                    rust_option_str(claim.kernel.as_deref()),
                    rust_option_str(claim.relation.as_deref()),
                    rust_str(&claim.claim_value)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        push_format(
            &mut source,
            format_args!(
                "pub const STAGE6_SUMCHECK_CLAIMS: &[Stage6SumcheckClaimPlan] = &[\n{claims}\n];\n"
            ),
        );
        source
    }

    fn emit_sumcheck_batch_constants(&self) -> String {
        let mut source = String::new();
        for (index, batch) in self.batches.iter().enumerate() {
            source.push_str(&emit_str_array(
                &format!("STAGE6_SUMCHECK_BATCH_{index}_ORDERED_CLAIMS"),
                &batch.ordered_claims,
            ));
            source.push_str(&emit_str_array(
                &format!("STAGE6_SUMCHECK_BATCH_{index}_CLAIM_OPERANDS"),
                &batch.claim_operands,
            ));
            source.push_str(&emit_usize_array(
                &format!("STAGE6_SUMCHECK_BATCH_{index}_ROUND_SCHEDULE"),
                &batch.round_schedule,
            ));
        }
        let batches = self
            .batches
            .iter()
            .enumerate()
            .map(|(index, batch)| {
                format!(
                    "    Stage6SumcheckBatchPlan {{ symbol: {}, stage: {}, proof_slot: {}, policy: {}, count: {}, ordered_claims: STAGE6_SUMCHECK_BATCH_{index}_ORDERED_CLAIMS, claim_operands: STAGE6_SUMCHECK_BATCH_{index}_CLAIM_OPERANDS, claim_label: {}, round_label: {}, round_schedule: STAGE6_SUMCHECK_BATCH_{index}_ROUND_SCHEDULE }},",
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
                "pub const STAGE6_SUMCHECK_BATCHES: &[Stage6SumcheckBatchPlan] = &[\n{batches}\n];\n"
            ),
        );
        source
    }

    fn emit_sumcheck_driver_constants(&self) -> String {
        let mut source = String::new();
        for (index, driver) in self.drivers.iter().enumerate() {
            source.push_str(&emit_usize_array(
                &format!("STAGE6_SUMCHECK_DRIVER_{index}_ROUND_SCHEDULE"),
                &driver.round_schedule,
            ));
        }
        let drivers = self
            .drivers
            .iter()
            .enumerate()
            .map(|(index, driver)| {
                format!(
                    "    Stage6SumcheckDriverPlan {{ symbol: {}, stage: {}, proof_slot: {}, kernel: {}, relation: {}, batch: {}, policy: {}, round_schedule: STAGE6_SUMCHECK_DRIVER_{index}_ROUND_SCHEDULE, claim_label: {}, round_label: {}, num_rounds: {}, degree: {} }},",
                    rust_str(&driver.symbol),
                    rust_str(&driver.stage),
                    rust_str(&driver.proof_slot),
                    rust_option_str(driver.kernel.as_deref()),
                    rust_option_str(driver.relation.as_deref()),
                    rust_str(&driver.batch),
                    rust_str(&driver.policy),
                    rust_str(&driver.claim_label),
                    rust_str(&driver.round_label),
                    driver.num_rounds,
                    driver.degree
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        push_format(
            &mut source,
            format_args!(
                "pub const STAGE6_SUMCHECK_DRIVERS: &[Stage6SumcheckDriverPlan] = &[\n{drivers}\n];\n"
            ),
        );
        source
    }

    fn emit_tail_constants(&self) -> String {
        let mut source = String::new();
        source.push_str(&self.emit_sumcheck_instance_result_constants());
        source.push_str(&self.emit_sumcheck_eval_constants());
        source.push_str(&self.emit_point_zero_constants());
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
                    "    Stage6SumcheckInstanceResultPlan {{ symbol: {}, source: {}, claim: {}, relation: {}, index: {}, point_arity: {}, num_rounds: {}, round_offset: {}, point_order: {}, degree: {} }},",
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
            "pub const STAGE6_SUMCHECK_INSTANCE_RESULTS: &[Stage6SumcheckInstanceResultPlan] = &[\n{instances}\n];\n\n"
        )
    }

    fn emit_sumcheck_eval_constants(&self) -> String {
        let evals = self
            .evals
            .iter()
            .map(|eval| {
                format!(
                    "    Stage6SumcheckEvalPlan {{ symbol: {}, source: {}, name: {}, index: {}, oracle: {} }},",
                    rust_str(&eval.symbol),
                    rust_str(&eval.source),
                    rust_str(&eval.name),
                    eval.index,
                    rust_str(&eval.oracle)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!("pub const STAGE6_SUMCHECK_EVALS: &[Stage6SumcheckEvalPlan] = &[\n{evals}\n];\n\n")
    }

    fn emit_point_zero_constants(&self) -> String {
        let zeros = self
            .point_zeros
            .iter()
            .map(|zero| {
                format!(
                    "    Stage6PointZeroPlan {{ symbol: {}, field: {}, arity: {} }},",
                    rust_str(&zero.symbol),
                    rust_str(&zero.field),
                    zero.arity
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!("pub const STAGE6_POINT_ZEROS: &[Stage6PointZeroPlan] = &[\n{zeros}\n];\n\n")
    }

    fn emit_point_slice_constants(&self) -> String {
        let slices = self
            .point_slices
            .iter()
            .map(|slice| {
                format!(
                    "    Stage6PointSlicePlan {{ symbol: {}, source: {}, offset: {}, length: {}, input: {} }},",
                    rust_str(&slice.symbol),
                    rust_str(&slice.source),
                    slice.offset,
                    slice.length,
                    rust_str(&slice.input)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!("pub const STAGE6_POINT_SLICES: &[Stage6PointSlicePlan] = &[\n{slices}\n];\n\n")
    }

    fn emit_point_concat_constants(&self) -> String {
        let mut source = String::new();
        for (index, concat) in self.point_concats.iter().enumerate() {
            source.push_str(&emit_str_array(
                &format!("STAGE6_POINT_CONCAT_{index}_INPUTS"),
                &concat.inputs,
            ));
        }
        let concats = self
            .point_concats
            .iter()
            .enumerate()
            .map(|(index, concat)| {
                format!(
                    "    Stage6PointConcatPlan {{ symbol: {}, layout: {}, arity: {}, inputs: STAGE6_POINT_CONCAT_{index}_INPUTS }},",
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
                "pub const STAGE6_POINT_CONCATS: &[Stage6PointConcatPlan] = &[\n{concats}\n];\n"
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
                    "    Stage6OpeningClaimPlan {{ symbol: {}, oracle: {}, domain: {}, point_arity: {}, claim_kind: {}, point_source: {}, eval_source: {} }},",
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
        format!("pub const STAGE6_OPENING_CLAIMS: &[Stage6OpeningClaimPlan] = &[\n{claims}\n];\n\n")
    }

    fn emit_opening_claim_equality_constants(&self) -> String {
        let equalities = self
            .opening_equalities
            .iter()
            .map(|equality| {
                format!(
                    "    Stage6OpeningClaimEqualityPlan {{ symbol: {}, mode: {}, lhs: {}, rhs: {} }},",
                    rust_str(&equality.symbol),
                    rust_str(&equality.mode),
                    rust_str(&equality.lhs),
                    rust_str(&equality.rhs)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "pub const STAGE6_OPENING_EQUALITIES: &[Stage6OpeningClaimEqualityPlan] = &[\n{equalities}\n];\n\n"
        )
    }

    fn emit_opening_batch_constants(&self) -> String {
        let mut source = String::new();
        for (index, batch) in self.opening_batches.iter().enumerate() {
            source.push_str(&emit_str_array(
                &format!("STAGE6_OPENING_BATCH_{index}_ORDERED_CLAIMS"),
                &batch.ordered_claims,
            ));
            source.push_str(&emit_str_array(
                &format!("STAGE6_OPENING_BATCH_{index}_CLAIM_OPERANDS"),
                &batch.claim_operands,
            ));
        }
        let batches = self
            .opening_batches
            .iter()
            .enumerate()
            .map(|(index, batch)| {
                format!(
                    "    Stage6OpeningBatchPlan {{ symbol: {}, stage: {}, proof_slot: {}, policy: {}, count: {}, ordered_claims: STAGE6_OPENING_BATCH_{index}_ORDERED_CLAIMS, claim_operands: STAGE6_OPENING_BATCH_{index}_CLAIM_OPERANDS }},",
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
                "pub const STAGE6_OPENING_BATCHES: &[Stage6OpeningBatchPlan] = &[\n{batches}\n];\n"
            ),
        );
        source
    }

    fn emit_entrypoint(&self) -> &'static str {
        match self.role {
            Role::Prover => {
                "pub fn execute_stage6_prover<E, T>(\n\
                 \x20   executor: &mut E,\n\
                 \x20   transcript: &mut T,\n\
                 ) -> Result<Stage6ExecutionArtifacts<Fr>, Stage6KernelError>\n\
                 where\n\
                 \x20   E: Stage6KernelExecutor<Fr>,\n\
                 \x20   T: Transcript<Challenge = Fr>,\n\
                 {\n\
                 \x20   execute_stage6_prover_with_program(&STAGE6_PROGRAM, executor, transcript)\n\
                 }\n\
                 \n\
                 pub fn execute_stage6_prover_with_program<E, T>(\n\
                 \x20   program: &'static Stage6CpuProgramPlan,\n\
                 \x20   executor: &mut E,\n\
                 \x20   transcript: &mut T,\n\
                 ) -> Result<Stage6ExecutionArtifacts<Fr>, Stage6KernelError>\n\
                 where\n\
                 \x20   E: Stage6KernelExecutor<Fr>,\n\
                 \x20   T: Transcript<Challenge = Fr>,\n\
                 {\n\
                 \x20   execute_stage6_program(program, Stage6ExecutionMode::Prover, executor, transcript)\n\
                 }\n"
            }
            Role::Verifier => {
                r#"pub fn verify_stage6<T>(
    proof: &Stage6Proof<Fr>,
    opening_inputs: &[Stage6OpeningInputValue<Fr>],
    verifier_data: Option<&Stage6VerifierData>,
    transcript: &mut T,
) -> Result<Stage6ExecutionArtifacts<Fr>, VerifyStage6Error>
where
    T: Transcript<Challenge = Fr>,
{
    verify_stage6_with_program(&STAGE6_PROGRAM, proof, opening_inputs, verifier_data, transcript)
}

pub fn verify_stage6_with_program<T>(
    program: &'static Stage6VerifierProgramPlan,
    proof: &Stage6Proof<Fr>,
    opening_inputs: &[Stage6OpeningInputValue<Fr>],
    verifier_data: Option<&Stage6VerifierData>,
    transcript: &mut T,
) -> Result<Stage6ExecutionArtifacts<Fr>, VerifyStage6Error>
where
    T: Transcript<Challenge = Fr>,
{
    if proof.sumchecks.len() != program.drivers.len() {
        return Err(VerifyStage6Error::UnexpectedProofCount {
            expected: program.drivers.len(),
            got: proof.sumchecks.len(),
        });
    }
    let mut store = Stage6ValueStore::with_opening_inputs(opening_inputs);
    store.seed_constants(program);
    let mut artifacts = Stage6ExecutionArtifacts::default();
    for step in program.steps {
        match step.kind {
            "transcript_squeeze" => {
                let squeeze =
                    find_squeeze(program, step.symbol).ok_or(VerifyStage6Error::MissingValue {
                        symbol: step.symbol,
                    })?;
                verify_stage6_squeeze(program, squeeze, &mut store, transcript, &mut artifacts)?;
            }
            "transcript_absorb_bytes" => {
                let absorb = find_absorb_bytes(program, step.symbol).ok_or(
                    VerifyStage6Error::MissingValue {
                        symbol: step.symbol,
                    },
                )?;
                absorb_stage6_bytes(absorb, transcript);
            }
            "sumcheck_driver" => {
                let driver =
                    find_driver(program, step.symbol).ok_or(VerifyStage6Error::MissingProof {
                        driver: step.symbol,
                    })?;
                verify_stage6_driver(
                    program,
                    driver,
                    proof,
                    verifier_data,
                    &mut store,
                    transcript,
                    &mut artifacts,
                )?;
            }
            _ => {
                return Err(VerifyStage6Error::InvalidProof {
                    driver: step.symbol,
                    reason: "unsupported stage6 program step",
                });
            }
        }
    }
    artifacts
        .opening_batches
        .extend(program.opening_batches.iter());
    Ok(artifacts)
}

pub fn stage6_verifier_program() -> &'static Stage6VerifierProgramPlan {
    &STAGE6_PROGRAM
}

fn verify_stage6_squeeze<T>(
    program: &'static Stage6VerifierProgramPlan,
    squeeze: &'static Stage6TranscriptSqueezePlan,
    store: &mut Stage6ValueStore<Fr>,
    transcript: &mut T,
    artifacts: &mut Stage6ExecutionArtifacts<Fr>,
) -> Result<(), VerifyStage6Error>
where
    T: Transcript<Challenge = Fr>,
{
    let values = transcript.challenge_vector(squeeze.count);
    store.observe_challenge_vector(program, squeeze, &values)?;
    artifacts.challenge_vectors.push(Stage6ChallengeVector {
        symbol: squeeze.symbol,
        values,
    });
    Ok(())
}

fn absorb_stage6_bytes<T>(absorb: &'static Stage6TranscriptAbsorbBytesPlan, transcript: &mut T)
where
    T: Transcript<Challenge = Fr>,
{
    transcript.append(&LabelWithCount(
        absorb.label.as_bytes(),
        absorb.payload.len() as u64,
    ));
    transcript.append_bytes(absorb.payload.as_bytes());
}

fn verify_stage6_driver<T>(
    program: &'static Stage6VerifierProgramPlan,
    driver: &'static Stage6SumcheckDriverPlan,
    proof: &Stage6Proof<Fr>,
    verifier_data: Option<&Stage6VerifierData>,
    store: &mut Stage6ValueStore<Fr>,
    transcript: &mut T,
    artifacts: &mut Stage6ExecutionArtifacts<Fr>,
) -> Result<(), VerifyStage6Error>
where
    T: Transcript<Challenge = Fr>,
{
    let proof = proof
        .sumchecks
        .get(artifacts.sumchecks.len())
        .ok_or(VerifyStage6Error::MissingProof {
            driver: driver.symbol,
        })?;
    let output = match driver.relation {
        Some("jolt.stage6.batched") => {
            verify_batched_stage6(program, driver, proof, verifier_data, store, transcript)?
        }
        Some(relation) => return Err(VerifyStage6Error::UnsupportedRelation { relation }),
        None => return Err(VerifyStage6Error::UnsupportedRelation { relation: "<missing>" }),
    };
    artifacts.sumchecks.push(output);
    Ok(())
}

fn verify_batched_stage6<T>(
    program: &'static Stage6VerifierProgramPlan,
    driver: &'static Stage6SumcheckDriverPlan,
    proof: &Stage6SumcheckOutput<Fr>,
    verifier_data: Option<&Stage6VerifierData>,
    store: &mut Stage6ValueStore<Fr>,
    transcript: &mut T,
) -> Result<Stage6SumcheckOutput<Fr>, VerifyStage6Error>
where
    T: Transcript<Challenge = Fr>,
{
    if proof.driver != driver.symbol {
        return Err(VerifyStage6Error::InvalidProof {
            driver: driver.symbol,
            reason: "driver symbol mismatch",
        });
    }
    let batch = find_batch(program, driver.batch)?;
    let claims = batch_claims(program, batch)?;
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
    let output = SumcheckVerifier::verify(&claim, &round_proofs, transcript)
        .map_err(|error| VerifyStage6Error::Sumcheck {
            driver: driver.symbol,
            error,
        })?;
    if !proof.point.is_empty() && proof.point != output.point {
        return Err(VerifyStage6Error::InvalidProof {
            driver: driver.symbol,
            reason: "batched point mismatch",
        });
    }
    let expected = expected_batched_output_claim(
        program,
        driver,
        verifier_data,
        &*store,
        &proof.evals,
        &output.point,
        &batching_coeffs,
    )?;
    if output.value != expected {
        return Err(VerifyStage6Error::InvalidProof {
            driver: driver.symbol,
            reason: "batched output claim mismatch",
        });
    }
    let verified = Stage6SumcheckOutput {
        driver: driver.symbol,
        point: output.point,
        evals: proof.evals.clone(),
        proof: proof.proof.clone(),
    };
    store.observe_sumcheck_output(program, &verified)?;
    append_opening_claims(program, store, transcript, &verified.evals)?;
    Ok(verified)
}

impl<F: Field> Stage6ValueStore<F> {
    fn with_opening_inputs(inputs: &[Stage6OpeningInputValue<F>]) -> Self {
        let mut store = Self::default();
        for input in inputs {
            store.insert_scalar(input.symbol, input.eval);
            store.insert_point(input.symbol, input.point.clone());
        }
        store
    }

    fn seed_constants(&mut self, program: &'static Stage6VerifierProgramPlan) {
        for constant in program.field_constants {
            self.insert_scalar(constant.symbol, F::from_u64(constant.value as u64));
        }
        for zero in program.point_zeros {
            self.insert_point(zero.symbol, vec![F::from_u64(0); zero.arity]);
        }
    }

    fn observe_challenge_vector(
        &mut self,
        program: &'static Stage6VerifierProgramPlan,
        plan: &'static Stage6TranscriptSqueezePlan,
        values: &[F],
    ) -> Result<(), VerifyStage6Error> {
        self.insert_point(plan.symbol, values.to_vec());
        if matches!(plan.kind, "challenge_scalar" | "scalar") {
            if values.len() != 1 {
                return Err(VerifyStage6Error::InvalidInputLength {
                    input: plan.symbol,
                    expected: 1,
                    actual: values.len(),
                });
            }
            self.insert_scalar(plan.symbol, values[0]);
        }
        self.evaluate_available_field_exprs(program)?;
        Ok(())
    }

    fn observe_sumcheck_output(
        &mut self,
        program: &'static Stage6VerifierProgramPlan,
        output: &Stage6SumcheckOutput<F>,
    ) -> Result<(), VerifyStage6Error> {
        self.insert_point(output.driver, output.point.clone());
        for instance in program
            .instance_results
            .iter()
            .filter(|instance| instance.source == output.driver)
        {
            let end = instance.round_offset + instance.point_arity;
            let mut point = output
                .point
                .get(instance.round_offset..end)
                .ok_or(VerifyStage6Error::InvalidInputLength {
                    input: instance.symbol,
                    expected: end,
                    actual: output.point.len(),
                })?
                .to_vec();
            match instance.point_order {
                "as_is" => {}
                "reverse" => point.reverse(),
                "bytecode_read_raf" => point = normalize_bytecode_read_raf_point(program, &point)?,
                "stage6_booleanity" => {}
                "instruction_read_raf" => point = normalize_instruction_read_raf_point(&point)?,
                _ => {
                    return Err(VerifyStage6Error::InvalidProof {
                        driver: output.driver,
                        reason: "unsupported point order",
                    });
                }
            }
            self.insert_point(instance.symbol, point);
        }
        for eval in program
            .evals
            .iter()
            .filter(|eval| eval.source == output.driver)
        {
            let value = output
                .evals
                .iter()
                .find(|value| value.name == eval.name)
                .or_else(|| output.evals.get(eval.index))
                .ok_or(VerifyStage6Error::MissingValue {
                    symbol: eval.symbol,
                })?
                .value;
            self.insert_scalar(eval.symbol, value);
            self.insert_scalar(eval.name, value);
        }
        self.evaluate_available_points(program)?;
        self.evaluate_available_field_exprs(program)?;
        self.verify_opening_equalities(program)?;
        Ok(())
    }

    fn claim_value(
        &mut self,
        program: &'static Stage6VerifierProgramPlan,
        claim: &Stage6SumcheckClaimPlan,
    ) -> Result<F, VerifyStage6Error> {
        self.evaluate_available_field_exprs(program)?;
        self.scalar(claim.claim_value)
    }

    fn batch_claim_values(
        &mut self,
        program: &'static Stage6VerifierProgramPlan,
        batch: &Stage6SumcheckBatchPlan,
    ) -> Result<Vec<F>, VerifyStage6Error> {
        batch
            .claim_operands
            .iter()
            .map(|symbol| {
                let claim = find_claim(program, symbol).ok_or(VerifyStage6Error::MissingClaim {
                    batch: batch.symbol,
                    claim: symbol,
                })?;
                self.claim_value(program, claim)
            })
            .collect()
    }

    fn evaluate_available_points(
        &mut self,
        program: &'static Stage6VerifierProgramPlan,
    ) -> Result<(), VerifyStage6Error> {
        loop {
            let mut progress = 0usize;
            for slice in program.point_slices {
                if self.try_point(slice.symbol).is_some() {
                    continue;
                }
                let Some(input) = self.try_point(slice.input) else { continue };
                let end = slice.offset + slice.length;
                let point = input
                    .get(slice.offset..end)
                    .ok_or(VerifyStage6Error::InvalidInputLength {
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
                let Some(point) = self.try_concat_point(concat) else { continue };
                if point.len() != concat.arity {
                    return Err(VerifyStage6Error::InvalidInputLength {
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

    fn evaluate_available_field_exprs(
        &mut self,
        program: &'static Stage6VerifierProgramPlan,
    ) -> Result<(), VerifyStage6Error> {
        loop {
            let mut progress = 0usize;
            for expr in program.field_exprs {
                if self.try_scalar(expr.symbol).is_some() {
                    continue;
                }
                let Some(operands) = self.try_expr_operands(expr) else { continue };
                self.insert_scalar(expr.symbol, evaluate_stage6_field_expr(expr, &operands)?);
                progress += 1;
            }
            if progress == 0 {
                return Ok(());
            }
        }
    }

    fn verify_opening_equalities(
        &self,
        program: &'static Stage6VerifierProgramPlan,
    ) -> Result<(), VerifyStage6Error> {
        for equality in program.opening_equalities {
            match equality.mode {
                "point_and_eval" => {
                    if self.point(equality.lhs)? != self.point(equality.rhs)?
                        || self.scalar(equality.lhs)? != self.scalar(equality.rhs)?
                    {
                        return Err(VerifyStage6Error::InvalidProof {
                            driver: equality.symbol,
                            reason: "opening claim equality failed",
                        });
                    }
                }
                _ => {
                    return Err(VerifyStage6Error::InvalidProof {
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

    fn scalar(&self, symbol: &'static str) -> Result<F, VerifyStage6Error> {
        self.try_scalar(symbol)
            .ok_or(VerifyStage6Error::MissingValue { symbol })
    }

    fn try_scalar(&self, symbol: &str) -> Option<F> {
        self.scalars
            .iter()
            .find(|(name, _)| *name == symbol)
            .map(|(_, value)| *value)
    }

    fn point(&self, symbol: &'static str) -> Result<&[F], VerifyStage6Error> {
        self.try_point(symbol)
            .ok_or(VerifyStage6Error::MissingValue { symbol })
    }

    fn try_point(&self, symbol: &str) -> Option<&[F]> {
        self.points
            .iter()
            .find(|(name, _)| *name == symbol)
            .map(|(_, point)| point.as_slice())
    }

    fn try_expr_operands(&self, expr: &Stage6FieldExprPlan) -> Option<Vec<F>> {
        expr.operands
            .iter()
            .map(|operand| self.try_scalar(operand))
            .collect()
    }

    fn try_concat_point(&self, concat: &Stage6PointConcatPlan) -> Option<Vec<F>> {
        let mut point = Vec::with_capacity(concat.arity);
        for input in concat.inputs {
            point.extend_from_slice(self.try_point(input)?);
        }
        Some(point)
    }
}

fn evaluate_stage6_field_expr<F: Field>(
    expr: &Stage6FieldExprPlan,
    operands: &[F],
) -> Result<F, VerifyStage6Error> {
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
                    VerifyStage6Error::UnsupportedFieldExpr {
                        symbol: expr.symbol,
                        formula,
                    }
                })?;
                return Ok(pow_field(operands[0], exponent));
            }
            Err(VerifyStage6Error::UnsupportedFieldExpr {
                symbol: expr.symbol,
                formula,
            })
        }
    }
}

fn expected_batched_output_claim(
    program: &'static Stage6VerifierProgramPlan,
    driver: &'static Stage6SumcheckDriverPlan,
    verifier_data: Option<&Stage6VerifierData>,
    store: &Stage6ValueStore<Fr>,
    evals: &[Stage6NamedEval<Fr>],
    point: &[Fr],
    batching_coeffs: &[Fr],
) -> Result<Fr, VerifyStage6Error> {
    let batch = find_batch(program, driver.batch)?;
    let claims = batch_claims(program, batch)?;
    let mut expected = Fr::from_u64(0);
    for (claim, coefficient) in claims.iter().zip(batching_coeffs) {
        let instance = program
            .instance_results
            .iter()
            .find(|instance| instance.claim == claim.symbol && instance.source == driver.symbol)
            .ok_or(VerifyStage6Error::MissingClaim {
                batch: batch.symbol,
                claim: claim.symbol,
            })?;
        let local_point = point
            .get(instance.round_offset..instance.round_offset + instance.num_rounds)
            .ok_or(VerifyStage6Error::InvalidInputLength {
                input: instance.symbol,
                expected: instance.round_offset + instance.num_rounds,
                actual: point.len(),
            })?;
        let value = match claim.relation {
            Some("jolt.stage6.bytecode_read_raf") => {
                let data = verifier_data
                    .and_then(|data| data.bytecode_read_raf.as_ref())
                    .ok_or(VerifyStage6Error::MissingValue {
                        symbol: "stage6.bytecode_read_raf.data",
                    })?;
                expected_bytecode_read_raf(program, data, store, evals, local_point)?
            }
            Some("jolt.stage6.booleanity") => {
                expected_booleanity(program, store, evals, local_point)?
            }
            Some("jolt.stage6.hamming_booleanity") => {
                expected_hamming_booleanity(store, evals, local_point)?
            }
            Some("jolt.stage6.ram_ra_virtual") => {
                expected_ram_ra_virtual(store, evals, local_point)?
            }
            Some("jolt.stage6.instruction_ra_virtual") => {
                expected_instruction_ra_virtual(program, store, evals, local_point)?
            }
            Some("jolt.stage6.inc_claim_reduction") => {
                expected_inc_claim_reduction(store, evals, local_point)?
            }
            Some(relation) => return Err(VerifyStage6Error::UnsupportedRelation { relation }),
            None => return Err(VerifyStage6Error::UnsupportedRelation { relation: "<missing>" }),
        };
        expected += *coefficient * value;
    }
    Ok(expected)
}

fn expected_bytecode_read_raf(
    program: &'static Stage6VerifierProgramPlan,
    data: &Stage6BytecodeReadRafData,
    store: &Stage6ValueStore<Fr>,
    evals: &[Stage6NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage6Error> {
    let log_t = stage6_trace_rounds(program)?;
    let opening_point = normalize_bytecode_read_raf_point(program, local_point)?;
    let log_k = opening_point.len() - log_t;
    let (r_address_prime, r_cycle_prime) = opening_point.split_at(log_k);

    let gamma = store.scalar("stage6.bytecode_read_raf.gamma")?;
    let gamma_powers = bytecode_gamma_powers(gamma);
    let int_eval = identity_polynomial_eval(r_address_prime);
    let stage_value_evals =
        bytecode_stage_value_evals(data, store, r_address_prime, r_cycle_prime.len())?;
    let stage_cycle_points = bytecode_stage_cycle_points(store, r_cycle_prime.len())?;
    let int_contrib = [
        gamma_powers[5] * int_eval,
        Fr::from_u64(0),
        gamma_powers[4] * int_eval,
        Fr::from_u64(0),
        Fr::from_u64(0),
    ];

    let mut val = Fr::from_u64(0);
    for index in 0..stage_value_evals.len() {
        val += (stage_value_evals[index] + int_contrib[index])
            * EqPolynomial::<Fr>::mle(&stage_cycle_points[index], r_cycle_prime)
            * gamma_powers[index];
    }

    let entry_bits = (0..log_k)
        .map(|index| {
            Fr::from_u64(((data.entry_bytecode_index >> (log_k - 1 - index)) & 1) as u64)
        })
        .collect::<Vec<_>>();
    let zero_cycle = vec![Fr::from_u64(0); r_cycle_prime.len()];
    let entry_contrib = gamma_powers[7]
        * EqPolynomial::<Fr>::mle(&entry_bits, r_address_prime)
        * EqPolynomial::<Fr>::mle(&zero_cycle, r_cycle_prime);
    let bytecode_ra =
        indexed_evals_by_prefix_any(evals, "stage6.bytecode_read_raf.eval.BytecodeRa_")?
            .into_iter()
            .product::<Fr>();
    Ok((val + entry_contrib) * bytecode_ra)
}

fn expected_booleanity(
    program: &'static Stage6VerifierProgramPlan,
    store: &Stage6ValueStore<Fr>,
    evals: &[Stage6NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage6Error> {
    let log_t = stage6_trace_rounds(program)?;
    let log_k_chunk =
        local_point
            .len()
            .checked_sub(log_t)
            .ok_or(VerifyStage6Error::InvalidInputLength {
                input: "stage6.booleanity.point",
                expected: log_t,
                actual: local_point.len(),
            })?;
    let stage5_point = store.point("stage6.input.stage5.instruction_read_raf.InstructionRa_0")?;
    let stage5_address_len =
        stage5_point
            .len()
            .checked_sub(log_t)
            .ok_or(VerifyStage6Error::InvalidInputLength {
                input: "stage6.input.stage5.instruction_read_raf.InstructionRa_0",
                expected: log_t,
                actual: stage5_point.len(),
            })?;
    if stage5_address_len < log_k_chunk {
        return Err(VerifyStage6Error::InvalidInputLength {
            input: "stage6.input.stage5.instruction_read_raf.InstructionRa_0",
            expected: log_k_chunk + log_t,
            actual: stage5_point.len(),
        });
    }

    let mut stage5_addr = stage5_point[..stage5_address_len].to_vec();
    stage5_addr.reverse();
    let mut combined_r = stage5_addr[stage5_address_len - log_k_chunk..].to_vec();
    combined_r.extend(stage5_point[stage5_address_len..].iter().rev().copied());
    if combined_r.len() != local_point.len() {
        return Err(VerifyStage6Error::InvalidInputLength {
            input: "stage6.booleanity.combined_point",
            expected: local_point.len(),
            actual: combined_r.len(),
        });
    }
    let eq_eval = EqPolynomial::<Fr>::mle(local_point, &combined_r);

    let gamma = store.scalar("stage6.booleanity.gamma")?;
    let gamma_sq = gamma.square();
    let mut gamma_power = Fr::from_u64(1);
    let mut booleanity = Fr::from_u64(0);
    for ra in booleanity_evals(evals)? {
        booleanity += gamma_power * (ra.square() - ra);
        gamma_power *= gamma_sq;
    }
    Ok(eq_eval * booleanity)
}

fn expected_hamming_booleanity(
    store: &Stage6ValueStore<Fr>,
    evals: &[Stage6NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage6Error> {
    let hamming = eval_by_name(evals, "stage6.hamming_booleanity.eval.HammingWeight")?;
    let lookup_output_point = reverse_slice(store.point("stage6.input.stage1.LookupOutput")?);
    if lookup_output_point.len() != local_point.len() {
        return Err(VerifyStage6Error::InvalidInputLength {
            input: "stage6.input.stage1.LookupOutput",
            expected: local_point.len(),
            actual: lookup_output_point.len(),
        });
    }
    let eq_eval = EqPolynomial::<Fr>::mle(local_point, &lookup_output_point);
    Ok((hamming.square() - hamming) * eq_eval)
}

fn expected_ram_ra_virtual(
    store: &Stage6ValueStore<Fr>,
    evals: &[Stage6NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage6Error> {
    let r_cycle_reduced = reverse_slice(local_point);
    let r_cycle = suffix_point(
        store.point("stage6.input.stage5.ram_ra_claim_reduction.RamRa")?,
        r_cycle_reduced.len(),
        "stage6.input.stage5.ram_ra_claim_reduction.RamRa",
    )?;
    let eq_eval = EqPolynomial::<Fr>::mle(r_cycle, &r_cycle_reduced);
    let ram_ra = indexed_evals_by_prefix_any(evals, "stage6.ram_ra_virtual.eval.RamRa_")?
        .into_iter()
        .product::<Fr>();
    Ok(eq_eval * ram_ra)
}

fn expected_instruction_ra_virtual(
    program: &'static Stage6VerifierProgramPlan,
    store: &Stage6ValueStore<Fr>,
    evals: &[Stage6NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage6Error> {
    let r_cycle_reduced = reverse_slice(local_point);
    let r_cycle = suffix_point(
        store.point("stage6.input.stage5.instruction_read_raf.InstructionRa_0")?,
        r_cycle_reduced.len(),
        "stage6.input.stage5.instruction_read_raf.InstructionRa_0",
    )?;
    let eq_eval = EqPolynomial::<Fr>::mle(r_cycle, &r_cycle_reduced);
    let committed_ra =
        indexed_evals_by_prefix_any(evals, "stage6.instruction_ra_virtual.eval.InstructionRa_")?;
    let virtual_count = program
        .opening_inputs
        .iter()
        .filter(|input| {
            input
                .symbol
                .starts_with("stage6.input.stage5.instruction_read_raf.InstructionRa_")
        })
        .count();
    if virtual_count == 0 || committed_ra.len() % virtual_count != 0 {
        return Err(VerifyStage6Error::InvalidInputLength {
            input: "stage6.instruction_ra_virtual.eval.InstructionRa_",
            expected: virtual_count,
            actual: committed_ra.len(),
        });
    }
    let committed_per_virtual = committed_ra.len() / virtual_count;
    let gamma = store.scalar("stage6.instruction_ra_virtual.gamma")?;
    let mut gamma_power = Fr::from_u64(1);
    let mut value = Fr::from_u64(0);
    for chunk in committed_ra.chunks(committed_per_virtual) {
        value += gamma_power * chunk.iter().copied().product::<Fr>();
        gamma_power *= gamma;
    }
    Ok(eq_eval * value)
}

fn expected_inc_claim_reduction(
    store: &Stage6ValueStore<Fr>,
    evals: &[Stage6NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage6Error> {
    let r_cycle_reduced = reverse_slice(local_point);
    let ram_inc_stage2 = suffix_point(
        store.point("stage6.input.stage2.ram_read_write.RamInc")?,
        r_cycle_reduced.len(),
        "stage6.input.stage2.ram_read_write.RamInc",
    )?;
    let ram_inc_stage4 = suffix_point(
        store.point("stage6.input.stage4.ram_val_check.RamInc")?,
        r_cycle_reduced.len(),
        "stage6.input.stage4.ram_val_check.RamInc",
    )?;
    let rd_inc_stage4 = suffix_point(
        store.point("stage6.input.stage4.registers_read_write.RdInc")?,
        r_cycle_reduced.len(),
        "stage6.input.stage4.registers_read_write.RdInc",
    )?;
    let rd_inc_stage5 = suffix_point(
        store.point("stage6.input.stage5.registers_val_evaluation.RdInc")?,
        r_cycle_reduced.len(),
        "stage6.input.stage5.registers_val_evaluation.RdInc",
    )?;
    let gamma = store.scalar("stage6.inc_claim_reduction.gamma")?;
    let eq_ram_combined = EqPolynomial::<Fr>::mle(ram_inc_stage2, &r_cycle_reduced)
        + gamma * EqPolynomial::<Fr>::mle(ram_inc_stage4, &r_cycle_reduced);
    let eq_rd_combined = EqPolynomial::<Fr>::mle(rd_inc_stage4, &r_cycle_reduced)
        + gamma * EqPolynomial::<Fr>::mle(rd_inc_stage5, &r_cycle_reduced);
    let ram_inc = eval_by_name(evals, "stage6.inc_claim_reduction.eval.RamInc")?;
    let rd_inc = eval_by_name(evals, "stage6.inc_claim_reduction.eval.RdInc")?;
    Ok(ram_inc * eq_ram_combined + gamma.square() * rd_inc * eq_rd_combined)
}

fn expected_instruction_read_raf(
    store: &Stage6ValueStore<Fr>,
    evals: &[Stage6NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage6Error> {
    const LOG_K: usize = 128;
    const XLEN: usize = 64;

    if local_point.len() < LOG_K {
        return Err(VerifyStage6Error::InvalidInputLength {
            input: "stage6.instruction_read_raf.point",
            expected: LOG_K,
            actual: local_point.len(),
        });
    }

    let (r_address_prime, r_cycle) = local_point.split_at(LOG_K);
    let r_cycle_prime = reverse_slice(r_cycle);
    let r_reduction = store.point("stage6.input.stage2.instruction.LookupOutput")?;
    let eq_eval_r_reduction = EqPolynomial::<Fr>::mle(r_reduction, &r_cycle_prime);

    let left_operand_eval = operand_polynomial_eval(r_address_prime, true);
    let right_operand_eval = operand_polynomial_eval(r_address_prime, false);
    let identity_poly_eval = identity_polynomial_eval(r_address_prime);

    let table_values = LookupTableKind::<XLEN>::all()
        .iter()
        .map(|table| table.evaluate_mle::<Fr, Fr>(r_address_prime))
        .collect::<Vec<_>>();
    let table_flag_claims = indexed_evals_by_prefix(
        evals,
        "stage6.instruction_read_raf.eval.LookupTableFlag_",
        table_values.len(),
    )?;
    let val_claim = table_values
        .into_iter()
        .zip(table_flag_claims)
        .map(|(table_value, flag_claim)| table_value * flag_claim)
        .sum::<Fr>();

    let ra_claim = indexed_evals_by_prefix_any(
        evals,
        "stage6.instruction_read_raf.eval.InstructionRa_",
    )?
    .into_iter()
    .product::<Fr>();
    let raf_flag_claim = eval_by_name(
        evals,
        "stage6.instruction_read_raf.eval.InstructionRafFlag",
    )?;
    let gamma = store.scalar("stage6.instruction_read_raf.gamma")?;

    let raf_claim = (Fr::from_u64(1) - raf_flag_claim)
        * (left_operand_eval + gamma * right_operand_eval)
        + raf_flag_claim * gamma * identity_poly_eval;
    Ok(eq_eval_r_reduction * ra_claim * (val_claim + gamma * raf_claim))
}

fn expected_ram_ra_claim_reduction(
    store: &Stage6ValueStore<Fr>,
    evals: &[Stage6NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage6Error> {
    let r_cycle_reduced = reverse_slice(local_point);
    let r_cycle_raf = suffix_point(
        store.point("stage6.input.stage2.ram_raf.RamRa")?,
        r_cycle_reduced.len(),
        "stage6.input.stage2.ram_raf.RamRa",
    )?;
    let r_cycle_rw = suffix_point(
        store.point("stage6.input.stage2.ram_read_write.RamRa")?,
        r_cycle_reduced.len(),
        "stage6.input.stage2.ram_read_write.RamRa",
    )?;
    let r_cycle_val = suffix_point(
        store.point("stage6.input.stage4.ram_val_check.RamRa")?,
        r_cycle_reduced.len(),
        "stage6.input.stage4.ram_val_check.RamRa",
    )?;
    let gamma = store.scalar("stage6.ram_ra_claim_reduction.gamma")?;
    let eq_combined = EqPolynomial::<Fr>::mle(r_cycle_raf, &r_cycle_reduced)
        + gamma * EqPolynomial::<Fr>::mle(r_cycle_rw, &r_cycle_reduced)
        + gamma.square() * EqPolynomial::<Fr>::mle(r_cycle_val, &r_cycle_reduced);
    let ram_ra = eval_by_name(evals, "stage6.ram_ra_claim_reduction.eval.RamRa")?;
    Ok(eq_combined * ram_ra)
}

fn expected_registers_val_evaluation(
    store: &Stage6ValueStore<Fr>,
    evals: &[Stage6NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage6Error> {
    let registers_val_point = store.point("stage6.input.stage4.registers.RegistersVal")?;
    let r_cycle = suffix_point(
        registers_val_point,
        local_point.len(),
        "stage6.input.stage4.registers.RegistersVal",
    )?;
    let r_reduced = reverse_slice(local_point);
    let lt_eval = lt_polynomial_eval(&r_reduced, r_cycle);
    let rd_inc = eval_by_name(evals, "stage6.registers_val_evaluation.eval.RdInc")?;
    let rd_wa = eval_by_name(evals, "stage6.registers_val_evaluation.eval.RdWa")?;
    Ok(rd_inc * rd_wa * lt_eval)
}

fn append_opening_claims<T>(
    program: &'static Stage6VerifierProgramPlan,
    store: &mut Stage6ValueStore<Fr>,
    transcript: &mut T,
    evals: &[Stage6NamedEval<Fr>],
) -> Result<(), VerifyStage6Error>
where
    T: Transcript<Challenge = Fr>,
{
    if program.opening_batches.is_empty() {
        for eval in evals {
            append_labeled_scalar(transcript, "opening_claim", &eval.value);
        }
        return Ok(());
    }
    store.evaluate_available_points(program)?;
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
            let claim = find_opening_claim(program, symbol).ok_or(VerifyStage6Error::MissingClaim {
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

fn find_squeeze(
    program: &'static Stage6VerifierProgramPlan,
    symbol: &str,
) -> Option<&'static Stage6TranscriptSqueezePlan> {
    program
        .transcript_squeezes
        .iter()
        .find(|squeeze| squeeze.symbol == symbol)
}

fn find_absorb_bytes(
    program: &'static Stage6VerifierProgramPlan,
    symbol: &str,
) -> Option<&'static Stage6TranscriptAbsorbBytesPlan> {
    program
        .transcript_absorb_bytes
        .iter()
        .find(|absorb| absorb.symbol == symbol)
}

fn find_driver(
    program: &'static Stage6VerifierProgramPlan,
    symbol: &str,
) -> Option<&'static Stage6SumcheckDriverPlan> {
    program
        .drivers
        .iter()
        .find(|driver| driver.symbol == symbol)
}

fn find_batch(
    program: &'static Stage6VerifierProgramPlan,
    symbol: &'static str,
) -> Result<&'static Stage6SumcheckBatchPlan, VerifyStage6Error> {
    program
        .batches
        .iter()
        .find(|batch| batch.symbol == symbol)
        .ok_or(VerifyStage6Error::MissingBatch {
            driver: symbol,
            batch: symbol,
        })
}

fn find_claim(
    program: &'static Stage6VerifierProgramPlan,
    symbol: &str,
) -> Option<&'static Stage6SumcheckClaimPlan> {
    program
        .claims
        .iter()
        .find(|claim| claim.symbol == symbol)
}

fn find_opening_claim(
    program: &'static Stage6VerifierProgramPlan,
    symbol: &str,
) -> Option<&'static Stage6OpeningClaimPlan> {
    program
        .opening_claims
        .iter()
        .find(|claim| claim.symbol == symbol)
}

fn batch_claims(
    program: &'static Stage6VerifierProgramPlan,
    batch: &Stage6SumcheckBatchPlan,
) -> Result<Vec<&'static Stage6SumcheckClaimPlan>, VerifyStage6Error> {
    batch
        .claim_operands
        .iter()
        .map(|symbol| {
            find_claim(program, symbol).ok_or(VerifyStage6Error::MissingClaim {
                batch: batch.symbol,
                claim: symbol,
            })
        })
        .collect()
}

fn stage6_trace_rounds(
    program: &'static Stage6VerifierProgramPlan,
) -> Result<usize, VerifyStage6Error> {
    program
        .instance_results
        .iter()
        .find(|instance| instance.relation == "jolt.stage6.hamming_booleanity")
        .map(|instance| instance.num_rounds)
        .ok_or(VerifyStage6Error::MissingValue {
            symbol: "stage6.hamming_booleanity.instance",
        })
}

fn bytecode_gamma_powers(gamma: Fr) -> [Fr; 8] {
    let mut powers = [Fr::from_u64(1); 8];
    for index in 1..powers.len() {
        powers[index] = powers[index - 1] * gamma;
    }
    powers
}

fn bytecode_stage_cycle_points(
    store: &Stage6ValueStore<Fr>,
    log_t: usize,
) -> Result<[Vec<Fr>; 5], VerifyStage6Error> {
    Ok([
        suffix_point(store.point("stage6.input.stage1.Imm")?, log_t, "stage6.input.stage1.Imm")?
            .to_vec(),
        suffix_point(
            store.point("stage6.input.stage2.OpFlagJump")?,
            log_t,
            "stage6.input.stage2.OpFlagJump",
        )?
        .to_vec(),
        suffix_point(
            store.point("stage6.input.stage3.spartan_shift.UnexpandedPC")?,
            log_t,
            "stage6.input.stage3.spartan_shift.UnexpandedPC",
        )?
        .to_vec(),
        suffix_point(
            store.point("stage6.input.stage4.Rs1Ra")?,
            log_t,
            "stage6.input.stage4.Rs1Ra",
        )?
        .to_vec(),
        suffix_point(
            store.point("stage6.input.stage5.registers_val_evaluation.RdWa")?,
            log_t,
            "stage6.input.stage5.registers_val_evaluation.RdWa",
        )?
        .to_vec(),
    ])
}

fn bytecode_stage_value_evals(
    data: &Stage6BytecodeReadRafData,
    store: &Stage6ValueStore<Fr>,
    r_address: &[Fr],
    log_t: usize,
) -> Result<[Fr; 5], VerifyStage6Error> {
    let expected_len =
        1usize
            .checked_shl(r_address.len() as u32)
            .ok_or(VerifyStage6Error::InvalidInputLength {
                input: "stage6.bytecode_read_raf.entries",
                expected: usize::BITS as usize,
                actual: r_address.len(),
            })?;
    if data.entries.len() != expected_len {
        return Err(VerifyStage6Error::InvalidInputLength {
            input: "stage6.bytecode_read_raf.entries",
            expected: expected_len,
            actual: data.entries.len(),
        });
    }
    if data.entry_bytecode_index >= expected_len {
        return Err(VerifyStage6Error::InvalidInputLength {
            input: "stage6.bytecode_read_raf.entry_bytecode_index",
            expected: expected_len,
            actual: data.entry_bytecode_index + 1,
        });
    }

    let stage1_gamma = store.scalar("stage6.bytecode_read_raf.stage1_gamma")?;
    let stage2_gamma = store.scalar("stage6.bytecode_read_raf.stage2_gamma")?;
    let stage3_gamma = store.scalar("stage6.bytecode_read_raf.stage3_gamma")?;
    let stage4_gamma = store.scalar("stage6.bytecode_read_raf.stage4_gamma")?;
    let stage5_gamma = store.scalar("stage6.bytecode_read_raf.stage5_gamma")?;
    let stage1_gamma_powers = field_powers(stage1_gamma, 16);
    let stage2_gamma_powers = field_powers(stage2_gamma, 4);
    let stage3_gamma_powers = field_powers(stage3_gamma, 9);
    let stage4_gamma_powers = field_powers(stage4_gamma, 3);
    let stage5_gamma_powers = field_powers(stage5_gamma, data.num_lookup_tables + 2);

    let stage4_register_point =
        register_prefix_point(store, "stage6.input.stage4.Rs1Ra", log_t)?;
    let stage5_register_point = register_prefix_point(
        store,
        "stage6.input.stage5.registers_val_evaluation.RdWa",
        log_t,
    )?;

    let mut evals = [Fr::from_u64(0); 5];
    for (index, entry) in data.entries.iter().enumerate() {
        let eq = indexed_boolean_eq(index, r_address);
        let values = bytecode_entry_stage_values(
            entry,
            data.num_lookup_tables,
            stage4_register_point,
            stage5_register_point,
            &stage1_gamma_powers,
            &stage2_gamma_powers,
            &stage3_gamma_powers,
            &stage4_gamma_powers,
            &stage5_gamma_powers,
        )?;
        for stage in 0..evals.len() {
            evals[stage] += eq * values[stage];
        }
    }
    Ok(evals)
}

fn bytecode_entry_stage_values(
    entry: &Stage6BytecodeEntry,
    num_lookup_tables: usize,
    stage4_register_point: &[Fr],
    stage5_register_point: &[Fr],
    stage1_gamma_powers: &[Fr],
    stage2_gamma_powers: &[Fr],
    stage3_gamma_powers: &[Fr],
    stage4_gamma_powers: &[Fr],
    stage5_gamma_powers: &[Fr],
) -> Result<[Fr; 5], VerifyStage6Error> {
    let mut stage1 = entry.address + entry.imm * stage1_gamma_powers[1];
    for (flag, gamma) in entry
        .circuit_flags
        .iter()
        .zip(stage1_gamma_powers.iter().skip(2))
    {
        if *flag {
            stage1 += *gamma;
        }
    }

    let mut stage2 = Fr::from_u64(0);
    if entry.circuit_flags[5] {
        stage2 += stage2_gamma_powers[0];
    }
    if entry.is_branch {
        stage2 += stage2_gamma_powers[1];
    }
    if entry.circuit_flags[6] {
        stage2 += stage2_gamma_powers[2];
    }
    if entry.circuit_flags[7] {
        stage2 += stage2_gamma_powers[3];
    }

    let mut stage3 = entry.imm + entry.address * stage3_gamma_powers[1];
    if entry.left_is_rs1 {
        stage3 += stage3_gamma_powers[2];
    }
    if entry.left_is_pc {
        stage3 += stage3_gamma_powers[3];
    }
    if entry.right_is_rs2 {
        stage3 += stage3_gamma_powers[4];
    }
    if entry.right_is_imm {
        stage3 += stage3_gamma_powers[5];
    }
    if entry.is_noop {
        stage3 += stage3_gamma_powers[6];
    }
    if entry.circuit_flags[7] {
        stage3 += stage3_gamma_powers[7];
    }
    if entry.circuit_flags[12] {
        stage3 += stage3_gamma_powers[8];
    }

    let stage4 = register_eq(entry.rd, stage4_register_point, "stage6.bytecode.entry.rd")?
        * stage4_gamma_powers[0]
        + register_eq(entry.rs1, stage4_register_point, "stage6.bytecode.entry.rs1")?
            * stage4_gamma_powers[1]
        + register_eq(entry.rs2, stage4_register_point, "stage6.bytecode.entry.rs2")?
            * stage4_gamma_powers[2];

    let mut stage5 =
        register_eq(entry.rd, stage5_register_point, "stage6.bytecode.entry.rd")?
            * stage5_gamma_powers[0];
    if !entry.is_interleaved {
        stage5 += stage5_gamma_powers[1];
    }
    if let Some(table) = entry.lookup_table {
        if table >= num_lookup_tables {
            return Err(VerifyStage6Error::InvalidInputLength {
                input: "stage6.bytecode.entry.lookup_table",
                expected: num_lookup_tables,
                actual: table + 1,
            });
        }
        stage5 += stage5_gamma_powers[2 + table];
    }

    Ok([stage1, stage2, stage3, stage4, stage5])
}

fn register_eq(
    index: Option<usize>,
    point: &[Fr],
    input: &'static str,
) -> Result<Fr, VerifyStage6Error> {
    let Some(index) = index else {
        return Ok(Fr::from_u64(0));
    };
    let register_count =
        1usize
            .checked_shl(point.len() as u32)
            .ok_or(VerifyStage6Error::InvalidInputLength {
                input,
                expected: usize::BITS as usize,
                actual: point.len(),
            })?;
    if index >= register_count {
        return Err(VerifyStage6Error::InvalidInputLength {
            input,
            expected: register_count,
            actual: index + 1,
        });
    }
    Ok(indexed_boolean_eq(index, point))
}

fn indexed_boolean_eq(index: usize, point: &[Fr]) -> Fr {
    let bits = (0..point.len())
        .map(|bit| Fr::from_u64(((index >> (point.len() - 1 - bit)) & 1) as u64))
        .collect::<Vec<_>>();
    EqPolynomial::<Fr>::mle(&bits, point)
}

fn field_powers(base: Fr, count: usize) -> Vec<Fr> {
    let mut powers = Vec::with_capacity(count);
    let mut power = Fr::from_u64(1);
    for _ in 0..count {
        powers.push(power);
        power *= base;
    }
    powers
}

fn normalize_bytecode_read_raf_point<F: Field>(
    program: &'static Stage6VerifierProgramPlan,
    point: &[F],
) -> Result<Vec<F>, VerifyStage6Error> {
    let log_t = stage6_trace_rounds(program)?;
    let log_k = point
        .len()
        .checked_sub(log_t)
        .ok_or(VerifyStage6Error::InvalidInputLength {
            input: "stage6.bytecode_read_raf.point",
            expected: log_t,
            actual: point.len(),
        })?;
    let mut normalized = point.to_vec();
    normalized[..log_k].reverse();
    normalized[log_k..].reverse();
    Ok(normalized)
}

fn prefix_point<'a>(
    point: &'a [Fr],
    length: usize,
    input: &'static str,
) -> Result<&'a [Fr], VerifyStage6Error> {
    point
        .get(..length)
        .filter(|prefix| prefix.len() == length)
        .ok_or(VerifyStage6Error::InvalidInputLength {
            input,
            expected: length,
            actual: point.len(),
        })
}

fn register_prefix_point<'a>(
    store: &'a Stage6ValueStore<Fr>,
    symbol: &'static str,
    log_t: usize,
) -> Result<&'a [Fr], VerifyStage6Error> {
    let point = store.point(symbol)?;
    let register_len = point
        .len()
        .checked_sub(log_t)
        .ok_or(VerifyStage6Error::InvalidInputLength {
            input: symbol,
            expected: log_t,
            actual: point.len(),
        })?;
    prefix_point(point, register_len, symbol)
}

fn booleanity_evals(evals: &[Stage6NamedEval<Fr>]) -> Result<Vec<Fr>, VerifyStage6Error> {
    let mut values = indexed_evals_by_prefix_any(
        evals,
        "stage6.booleanity.eval.InstructionRa_",
    )?;
    values.extend(indexed_evals_by_prefix_any(
        evals,
        "stage6.booleanity.eval.BytecodeRa_",
    )?);
    values.extend(indexed_evals_by_prefix_any(
        evals,
        "stage6.booleanity.eval.RamRa_",
    )?);
    Ok(values)
}

fn eval_by_name(evals: &[Stage6NamedEval<Fr>], name: &'static str) -> Result<Fr, VerifyStage6Error> {
    evals
        .iter()
        .find(|eval| eval.name == name)
        .map(|eval| eval.value)
        .ok_or(VerifyStage6Error::MissingValue { symbol: name })
}

fn indexed_evals_by_prefix(
    evals: &[Stage6NamedEval<Fr>],
    prefix: &'static str,
    count: usize,
) -> Result<Vec<Fr>, VerifyStage6Error> {
    let mut values = vec![None; count];
    for eval in evals {
        let Some(suffix) = eval.name.strip_prefix(prefix) else {
            continue;
        };
        let index = suffix.parse::<usize>().map_err(|_| {
            VerifyStage6Error::InvalidProof {
                driver: prefix,
                reason: "invalid indexed eval suffix",
            }
        })?;
        if index >= count || values[index].is_some() {
            return Err(VerifyStage6Error::InvalidProof {
                driver: prefix,
                reason: "invalid indexed eval",
            });
        }
        values[index] = Some(eval.value);
    }
    values
        .into_iter()
        .map(|value| value.ok_or(VerifyStage6Error::MissingValue { symbol: prefix }))
        .collect()
}

fn indexed_evals_by_prefix_any(
    evals: &[Stage6NamedEval<Fr>],
    prefix: &'static str,
) -> Result<Vec<Fr>, VerifyStage6Error> {
    let mut indexed_values = Vec::new();
    for eval in evals {
        let Some(suffix) = eval.name.strip_prefix(prefix) else {
            continue;
        };
        let index = suffix.parse::<usize>().map_err(|_| {
            VerifyStage6Error::InvalidProof {
                driver: prefix,
                reason: "invalid indexed eval suffix",
            }
        })?;
        if indexed_values
            .iter()
            .any(|(existing_index, _)| *existing_index == index)
        {
            return Err(VerifyStage6Error::InvalidProof {
                driver: prefix,
                reason: "duplicate indexed eval",
            });
        }
        indexed_values.push((index, eval.value));
    }
    if indexed_values.is_empty() {
        return Err(VerifyStage6Error::MissingValue { symbol: prefix });
    }
    indexed_values.sort_by_key(|(index, _)| *index);
    for (expected, (actual, _)) in indexed_values.iter().enumerate() {
        if *actual != expected {
            return Err(VerifyStage6Error::InvalidProof {
                driver: prefix,
                reason: "non-contiguous indexed eval",
            });
        }
    }
    Ok(indexed_values
        .into_iter()
        .map(|(_, value)| value)
        .collect())
}

fn append_labeled_scalar<T>(transcript: &mut T, label: &'static str, scalar: &Fr)
where
    T: Transcript<Challenge = Fr>,
{
    transcript.append(&Label(label.as_bytes()));
    transcript.append(scalar);
}

fn lt_polynomial_eval(x: &[Fr], y: &[Fr]) -> Fr {
    let mut lt_eval = Fr::from_u64(0);
    let mut eq_term = Fr::from_u64(1);
    for (x_i, y_i) in x.iter().zip(y.iter()) {
        lt_eval += (Fr::from_u64(1) - *x_i) * *y_i * eq_term;
        eq_term *= Fr::from_u64(1) - *x_i - *y_i + *x_i * *y_i + *x_i * *y_i;
    }
    lt_eval
}

fn operand_polynomial_eval(point: &[Fr], left: bool) -> Fr {
    let stride_offset = if left { 0 } else { 1 };
    let operand_bits = point.len() / 2;
    (0..operand_bits)
        .map(|index| point[2 * index + stride_offset].mul_pow_2(operand_bits - 1 - index))
        .sum()
}

fn identity_polynomial_eval(point: &[Fr]) -> Fr {
    point
        .iter()
        .enumerate()
        .map(|(index, value)| value.mul_pow_2(point.len() - 1 - index))
        .sum()
}

fn suffix_point<'a>(
    point: &'a [Fr],
    length: usize,
    input: &'static str,
) -> Result<&'a [Fr], VerifyStage6Error> {
    point
        .get(point.len().saturating_sub(length)..)
        .filter(|suffix| suffix.len() == length)
        .ok_or(VerifyStage6Error::InvalidInputLength {
            input,
            expected: length,
            actual: point.len(),
        })
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

fn single_operand<F: Field>(symbol: &'static str, operands: &[F]) -> Result<F, VerifyStage6Error> {
    require_operand_count(symbol, 1, operands.len())?;
    Ok(operands[0])
}

fn require_operand_count(
    input: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), VerifyStage6Error> {
    if expected == actual {
        Ok(())
    } else {
        Err(VerifyStage6Error::InvalidInputLength {
            input,
            expected,
            actual,
        })
    }
}

fn reverse_slice(values: &[Fr]) -> Vec<Fr> {
    values.iter().rev().copied().collect()
}

fn normalize_instruction_read_raf_point<F: Field>(point: &[F]) -> Result<Vec<F>, VerifyStage6Error> {
    const LOG_K: usize = 128;
    if point.len() < LOG_K {
        return Err(VerifyStage6Error::InvalidInputLength {
            input: "stage6.instruction_read_raf.point",
            expected: LOG_K,
            actual: point.len(),
        });
    }
    let mut normalized = point.to_vec();
    normalized[LOG_K..].reverse();
    Ok(normalized)
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
            Role::Prover => "Stage6CpuProgramPlan",
            Role::Verifier => "Stage6VerifierProgramPlan",
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

fn rust_option_str(value: Option<&str>) -> String {
    value
        .map(|value| format!("Some({})", rust_str(value)))
        .unwrap_or_else(|| "None".to_owned())
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

fn parse_integer_attr(attribute: Attribute<'_>) -> Option<usize> {
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
