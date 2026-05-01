use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use melior::ir::block::BlockLike;
use melior::ir::operation::{OperationLike, OperationResult};
use melior::ir::{Attribute, OperationRef};

use crate::emit::rust::{EmitError, RustSourceFile};
use crate::ir::{string_attribute_value, symbol_attribute_value, BoltModule, Cpu, Role};
use crate::schema::verify_cpu_schema;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2CpuProgram {
    pub role: Role,
    pub params: Stage2Params,
    pub steps: Vec<Stage2ProgramStepPlan>,
    pub transcript_squeezes: Vec<Stage2TranscriptSqueezePlan>,
    pub opening_inputs: Vec<Stage2OpeningInputPlan>,
    pub field_constants: Vec<Stage2FieldConstantPlan>,
    pub challenge_extracts: Vec<Stage2ChallengeExtractPlan>,
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
pub struct Stage2ChallengeExtractPlan {
    pub symbol: String,
    pub source: String,
    pub index: usize,
    pub challenge_source: String,
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
        source: program.emit_source(),
    })
}

impl Stage2CpuProgram {
    fn from_module(module: &BoltModule<'_, Cpu>) -> Result<Self, EmitError> {
        let mut params = None;
        let mut steps = Vec::new();
        let mut transcript_squeezes = Vec::new();
        let mut opening_inputs = Vec::new();
        let mut field_constants = Vec::new();
        let mut challenge_extracts = Vec::new();
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
                "cpu.field_constant" => {
                    field_constants.push(Stage2FieldConstantPlan {
                        symbol: string_attr(op, "sym_name")?,
                        field: symbol_attr(op, "field")?,
                        value: int_attr(op, "value")?,
                    });
                }
                "cpu.challenge_extract" => {
                    challenge_extracts.push(Stage2ChallengeExtractPlan {
                        symbol: string_attr(op, "sym_name")?,
                        source: symbol_attr(op, "source")?,
                        index: int_attr(op, "index")?,
                        challenge_source: operand_symbol(op, 0)?,
                    });
                }
                "cpu.field_expr" => {
                    field_exprs.push(Stage2FieldExprPlan {
                        symbol: string_attr(op, "sym_name")?,
                        kind: string_attr(op, "kind")?,
                        formula: string_attr(op, "formula")?,
                        operand_names: symbol_array_attr(op, "operands")?,
                        operands: operand_symbols(op, 0)?,
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
            challenge_extracts,
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
        let squeezes = symbols(
            self.transcript_squeezes
                .iter()
                .map(|squeeze| &squeeze.symbol),
        );
        for extract in &self.challenge_extracts {
            if !squeezes.contains(&extract.challenge_source) {
                return Err(EmitError::new(format!(
                    "challenge extract @{} uses missing challenge @{}",
                    extract.symbol, extract.challenge_source
                )));
            }
            if extract.source != extract.challenge_source {
                return Err(EmitError::new(format!(
                    "challenge extract @{} source attr @{} differs from operand @{}",
                    extract.symbol, extract.source, extract.challenge_source
                )));
            }
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
            self.challenge_extracts
                .iter()
                .map(|extract| &extract.symbol),
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

    fn emit_source(&self) -> String {
        match self.role {
            Role::Prover => self.emit_prover_source(),
            Role::Verifier => self.emit_verifier_source(),
        }
    }

    fn emit_prover_source(&self) -> String {
        let mut source = String::new();
        source.push_str("#![allow(dead_code)]\n\n");
        source.push_str(Self::emit_prover_imports());
        source.push_str("\n\n");
        source.push_str(Self::emit_prover_types());
        source.push('\n');
        source.push_str(&self.emit_prover_constants());
        source.push('\n');
        source.push_str(Self::emit_prover_entrypoint());
        source
    }

    fn emit_verifier_source(&self) -> String {
        let mut source = String::new();
        source.push_str("#![allow(dead_code)]\n\n");
        source.push_str(Self::emit_verifier_imports());
        source.push_str("\n\n");
        source.push_str(Self::emit_verifier_types());
        source.push('\n');
        source.push_str(&self.emit_verifier_constants());
        source.push('\n');
        source.push_str(Self::emit_verifier_entrypoint());
        source
    }

    fn filename(&self) -> &'static str {
        match self.role {
            Role::Prover => "prove_stage2.rs",
            Role::Verifier => "verify_stage2.rs",
        }
    }

    fn emit_prover_imports() -> &'static str {
        "use jolt_field::Fr;\n\
         use jolt_kernels::stage2::{execute_stage2_program, Stage2ChallengeExtractPlan, Stage2CpuProgramPlan, Stage2ExecutionArtifacts, Stage2ExecutionMode, Stage2FieldConstantPlan, Stage2FieldExprPlan, Stage2KernelError, Stage2KernelExecutor, Stage2KernelPlan, Stage2OpeningBatchPlan, Stage2OpeningClaimPlan, Stage2OpeningInputPlan, Stage2Params, Stage2PointConcatPlan, Stage2PointSlicePlan, Stage2ProgramStepPlan, Stage2SumcheckBatchPlan, Stage2SumcheckClaimPlan, Stage2SumcheckDriverPlan, Stage2SumcheckEvalPlan, Stage2SumcheckInstanceResultPlan, Stage2TranscriptSqueezePlan};\n\
         use jolt_transcript::Blake2bTranscript;"
    }

    fn emit_prover_types() -> &'static str {
        "pub type DefaultStage2Transcript = Blake2bTranscript<Fr>;\n"
    }

    fn emit_verifier_imports() -> &'static str {
        "use jolt_field::Fr;\nuse jolt_transcript::Blake2bTranscript;"
    }

    fn emit_verifier_types() -> &'static str {
        r"pub type DefaultStage2Transcript = Blake2bTranscript<Fr>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage2Params {
    pub field: &'static str,
    pub pcs: &'static str,
    pub transcript: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage2TranscriptSqueezePlan {
    pub symbol: &'static str,
    pub label: &'static str,
    pub kind: &'static str,
    pub count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage2ProgramStepPlan {
    pub kind: &'static str,
    pub symbol: &'static str,
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
    pub relation: &'static str,
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
pub struct Stage2VerifierProgramPlan {
    pub params: Stage2Params,
    pub steps: &'static [Stage2ProgramStepPlan],
    pub transcript_squeezes: &'static [Stage2TranscriptSqueezePlan],
    pub opening_inputs: &'static [Stage2OpeningInputPlan],
    pub field_constants: &'static [Stage2FieldConstantPlan],
    pub challenge_extracts: &'static [Stage2ChallengeExtractPlan],
    pub field_exprs: &'static [Stage2FieldExprPlan],
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
"
    }

    fn emit_prover_constants(&self) -> String {
        let mut source = self.emit_shared_constants();
        source.push_str(&self.emit_kernel_constants());
        source.push_str(&self.emit_prover_sumcheck_claim_constants());
        source.push_str(&self.emit_sumcheck_batch_constants());
        source.push_str(&self.emit_prover_sumcheck_driver_constants());
        source.push_str(&self.emit_tail_constants());
        source.push_str(
            "pub const STAGE2_PROGRAM: Stage2CpuProgramPlan = Stage2CpuProgramPlan {\n\
             \x20   params: STAGE2_PARAMS,\n\
             \x20   steps: STAGE2_PROGRAM_STEPS,\n\
             \x20   transcript_squeezes: STAGE2_TRANSCRIPT_SQUEEZES,\n\
             \x20   opening_inputs: STAGE2_OPENING_INPUTS,\n\
             \x20   field_constants: STAGE2_FIELD_CONSTANTS,\n\
             \x20   challenge_extracts: STAGE2_CHALLENGE_EXTRACTS,\n\
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
        source
    }

    fn emit_verifier_constants(&self) -> String {
        let mut source = self.emit_shared_constants();
        source.push_str(&self.emit_verifier_sumcheck_claim_constants());
        source.push_str(&self.emit_sumcheck_batch_constants());
        source.push_str(&self.emit_verifier_sumcheck_driver_constants());
        source.push_str(&self.emit_tail_constants());
        source.push_str(
            "pub const STAGE2_PROGRAM: Stage2VerifierProgramPlan = Stage2VerifierProgramPlan {\n\
             \x20   params: STAGE2_PARAMS,\n\
             \x20   steps: STAGE2_PROGRAM_STEPS,\n\
             \x20   transcript_squeezes: STAGE2_TRANSCRIPT_SQUEEZES,\n\
             \x20   opening_inputs: STAGE2_OPENING_INPUTS,\n\
             \x20   field_constants: STAGE2_FIELD_CONSTANTS,\n\
             \x20   challenge_extracts: STAGE2_CHALLENGE_EXTRACTS,\n\
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
        source
    }

    fn emit_shared_constants(&self) -> String {
        let mut source = String::new();
        writeln!(
            source,
            "pub const STAGE2_PARAMS: Stage2Params = Stage2Params {{\n\
             \x20   field: {},\n\
             \x20   pcs: {},\n\
             \x20   transcript: {},\n\
             }};\n",
            rust_str(&self.params.field),
            rust_str(&self.params.pcs),
            rust_str(&self.params.transcript)
        )
        .expect("write generated Stage2 params");
        source.push_str(&self.emit_program_step_constants());
        source.push_str(&self.emit_transcript_squeeze_constants());
        source.push_str(&self.emit_opening_input_constants());
        source.push_str(&self.emit_field_constant_constants());
        source.push_str(&self.emit_challenge_extract_constants());
        source.push_str(&self.emit_field_expr_constants());
        source
    }

    fn emit_program_step_constants(&self) -> String {
        let steps = self
            .steps
            .iter()
            .map(|step| {
                format!(
                    "    Stage2ProgramStepPlan {{ kind: {}, symbol: {} }},",
                    rust_str(&step.kind),
                    rust_str(&step.symbol),
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!("pub const STAGE2_PROGRAM_STEPS: &[Stage2ProgramStepPlan] = &[\n{steps}\n];\n\n")
    }

    fn emit_transcript_squeeze_constants(&self) -> String {
        let squeezes = self
            .transcript_squeezes
            .iter()
            .map(|squeeze| {
                format!(
                    "    Stage2TranscriptSqueezePlan {{ symbol: {}, label: {}, kind: {}, count: {} }},",
                    rust_str(&squeeze.symbol),
                    rust_str(&squeeze.label),
                    rust_str(&squeeze.kind),
                    squeeze.count,
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "pub const STAGE2_TRANSCRIPT_SQUEEZES: &[Stage2TranscriptSqueezePlan] = &[\n{squeezes}\n];\n\n"
        )
    }

    fn emit_opening_input_constants(&self) -> String {
        let inputs = self
            .opening_inputs
            .iter()
            .map(|input| {
                format!(
                    "    Stage2OpeningInputPlan {{ symbol: {}, source_stage: {}, source_claim: {}, oracle: {}, domain: {}, point_arity: {}, claim_kind: {} }},",
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
        format!("pub const STAGE2_OPENING_INPUTS: &[Stage2OpeningInputPlan] = &[\n{inputs}\n];\n\n")
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

    fn emit_challenge_extract_constants(&self) -> String {
        let extracts = self
            .challenge_extracts
            .iter()
            .map(|extract| {
                format!(
                    "    Stage2ChallengeExtractPlan {{ symbol: {}, source: {}, index: {}, challenge_source: {} }},",
                    rust_str(&extract.symbol),
                    rust_str(&extract.source),
                    extract.index,
                    rust_str(&extract.challenge_source)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "pub const STAGE2_CHALLENGE_EXTRACTS: &[Stage2ChallengeExtractPlan] = &[\n{extracts}\n];\n\n"
        )
    }

    fn emit_field_expr_constants(&self) -> String {
        let mut source = String::new();
        for (index, expr) in self.field_exprs.iter().enumerate() {
            source.push_str(&emit_str_array(
                &format!("STAGE2_FIELD_EXPR_{index}_OPERAND_NAMES"),
                &expr.operand_names,
            ));
            source.push_str(&emit_str_array(
                &format!("STAGE2_FIELD_EXPR_{index}_OPERANDS"),
                &expr.operands,
            ));
        }
        let exprs = self
            .field_exprs
            .iter()
            .enumerate()
            .map(|(index, expr)| {
                format!(
                    "    Stage2FieldExprPlan {{ symbol: {}, kind: {}, formula: {}, operand_names: STAGE2_FIELD_EXPR_{index}_OPERAND_NAMES, operands: STAGE2_FIELD_EXPR_{index}_OPERANDS }},",
                    rust_str(&expr.symbol),
                    rust_str(&expr.kind),
                    rust_str(&expr.formula)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        writeln!(
            source,
            "pub const STAGE2_FIELD_EXPRS: &[Stage2FieldExprPlan] = &[\n{exprs}\n];\n"
        )
        .expect("write generated Stage2 field expressions");
        source
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

    fn emit_prover_sumcheck_claim_constants(&self) -> String {
        self.emit_sumcheck_claim_constants(true)
    }

    fn emit_verifier_sumcheck_claim_constants(&self) -> String {
        self.emit_sumcheck_claim_constants(false)
    }

    fn emit_sumcheck_claim_constants(&self, prover: bool) -> String {
        let mut source = String::new();
        for (index, claim) in self.claims.iter().enumerate() {
            source.push_str(&emit_str_array(
                &format!("STAGE2_SUMCHECK_CLAIM_{index}_INPUT_OPENINGS"),
                &claim.input_openings,
            ));
        }
        let claims = self
            .claims
            .iter()
            .enumerate()
            .map(|(index, claim)| {
                if prover {
                    let kernel = claim
                        .kernel
                        .as_deref()
                        .expect("prover sumcheck claim kernel verified");
                    format!(
                        "    Stage2SumcheckClaimPlan {{ symbol: {}, stage: {}, domain: {}, num_rounds: {}, degree: {}, claim: {}, kernel: {}, claim_value: {}, input_openings: STAGE2_SUMCHECK_CLAIM_{index}_INPUT_OPENINGS }},",
                        rust_str(&claim.symbol),
                        rust_str(&claim.stage),
                        rust_str(&claim.domain),
                        claim.num_rounds,
                        claim.degree,
                        rust_str(&claim.claim),
                        rust_str(kernel),
                        rust_str(&claim.claim_value)
                    )
                } else {
                    let relation = claim
                        .relation
                        .as_deref()
                        .expect("verifier sumcheck claim relation verified");
                    format!(
                        "    Stage2SumcheckClaimPlan {{ symbol: {}, stage: {}, domain: {}, num_rounds: {}, degree: {}, claim: {}, relation: {}, claim_value: {}, input_openings: STAGE2_SUMCHECK_CLAIM_{index}_INPUT_OPENINGS }},",
                        rust_str(&claim.symbol),
                        rust_str(&claim.stage),
                        rust_str(&claim.domain),
                        claim.num_rounds,
                        claim.degree,
                        rust_str(&claim.claim),
                        rust_str(relation),
                        rust_str(&claim.claim_value)
                    )
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        writeln!(
            source,
            "pub const STAGE2_SUMCHECK_CLAIMS: &[Stage2SumcheckClaimPlan] = &[\n{claims}\n];\n"
        )
        .expect("write generated Stage2 sumcheck claims");
        source
    }

    fn emit_sumcheck_batch_constants(&self) -> String {
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
        writeln!(
            source,
            "pub const STAGE2_SUMCHECK_BATCHES: &[Stage2SumcheckBatchPlan] = &[\n{batches}\n];\n"
        )
        .expect("write generated Stage2 sumcheck batches");
        source
    }

    fn emit_prover_sumcheck_driver_constants(&self) -> String {
        self.emit_sumcheck_driver_constants(true)
    }

    fn emit_verifier_sumcheck_driver_constants(&self) -> String {
        self.emit_sumcheck_driver_constants(false)
    }

    fn emit_sumcheck_driver_constants(&self, prover: bool) -> String {
        let mut source = String::new();
        for (index, driver) in self.drivers.iter().enumerate() {
            source.push_str(&emit_usize_array(
                &format!("STAGE2_SUMCHECK_DRIVER_{index}_ROUND_SCHEDULE"),
                &driver.round_schedule,
            ));
        }
        let drivers = self
            .drivers
            .iter()
            .enumerate()
            .map(|(index, driver)| {
                if prover {
                    let kernel = driver
                        .kernel
                        .as_deref()
                        .expect("prover sumcheck driver kernel verified");
                    format!(
                        "    Stage2SumcheckDriverPlan {{ symbol: {}, stage: {}, proof_slot: {}, kernel: {}, batch: {}, policy: {}, round_schedule: STAGE2_SUMCHECK_DRIVER_{index}_ROUND_SCHEDULE, claim_label: {}, round_label: {}, num_rounds: {}, degree: {} }},",
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
                    )
                } else {
                    let relation = driver
                        .relation
                        .as_deref()
                        .expect("verifier sumcheck driver relation verified");
                    format!(
                        "    Stage2SumcheckDriverPlan {{ symbol: {}, stage: {}, proof_slot: {}, relation: {}, batch: {}, policy: {}, round_schedule: STAGE2_SUMCHECK_DRIVER_{index}_ROUND_SCHEDULE, claim_label: {}, round_label: {}, num_rounds: {}, degree: {} }},",
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
                    )
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        writeln!(
            source,
            "pub const STAGE2_SUMCHECK_DRIVERS: &[Stage2SumcheckDriverPlan] = &[\n{drivers}\n];\n"
        )
        .expect("write generated Stage2 sumcheck drivers");
        source
    }

    fn emit_tail_constants(&self) -> String {
        let mut source = String::new();
        source.push_str(&self.emit_sumcheck_instance_result_constants());
        source.push_str(&self.emit_sumcheck_eval_constants());
        source.push_str(&self.emit_point_slice_constants());
        source.push_str(&self.emit_point_concat_constants());
        source.push_str(&self.emit_opening_claim_constants());
        source.push_str(&self.emit_opening_batch_constants());
        source
    }

    fn emit_sumcheck_instance_result_constants(&self) -> String {
        let instances = self
            .instance_results
            .iter()
            .map(|instance| {
                format!(
                    "    Stage2SumcheckInstanceResultPlan {{ symbol: {}, source: {}, claim: {}, relation: {}, index: {}, point_arity: {}, num_rounds: {}, round_offset: {}, point_order: {}, degree: {} }},",
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
            "pub const STAGE2_SUMCHECK_INSTANCE_RESULTS: &[Stage2SumcheckInstanceResultPlan] = &[\n{instances}\n];\n\n"
        )
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
        writeln!(
            source,
            "pub const STAGE2_POINT_CONCATS: &[Stage2PointConcatPlan] = &[\n{concats}\n];\n"
        )
        .expect("write generated Stage2 point concats");
        source
    }

    fn emit_opening_claim_constants(&self) -> String {
        let claims = self
            .opening_claims
            .iter()
            .map(|claim| {
                format!(
                    "    Stage2OpeningClaimPlan {{ symbol: {}, oracle: {}, domain: {}, point_arity: {}, claim_kind: {}, point_source: {}, eval_source: {} }},",
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
        format!("pub const STAGE2_OPENING_CLAIMS: &[Stage2OpeningClaimPlan] = &[\n{claims}\n];\n\n")
    }

    fn emit_opening_batch_constants(&self) -> String {
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
        writeln!(
            source,
            "pub const STAGE2_OPENING_BATCHES: &[Stage2OpeningBatchPlan] = &[\n{batches}\n];\n"
        )
        .expect("write generated Stage2 opening batches");
        source
    }

    fn emit_prover_entrypoint() -> &'static str {
        "pub fn execute_stage2_prover<E>(\n\
         \x20   executor: &mut E,\n\
         \x20   transcript: &mut DefaultStage2Transcript,\n\
         ) -> Result<Stage2ExecutionArtifacts<Fr>, Stage2KernelError>\n\
         where\n\
         \x20   E: Stage2KernelExecutor<Fr>,\n\
         {\n\
         \x20   execute_stage2_program(&STAGE2_PROGRAM, Stage2ExecutionMode::Prover, executor, transcript)\n\
         }\n"
    }

    fn emit_verifier_entrypoint() -> &'static str {
        "pub fn stage2_verifier_program() -> &'static Stage2VerifierProgramPlan {\n\
         \x20   &STAGE2_PROGRAM\n\
         }\n"
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
