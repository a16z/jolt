use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use melior::ir::block::BlockLike;
use melior::ir::operation::{OperationLike, OperationResult};
use melior::ir::{Attribute, OperationRef};

use crate::emit::rust::{EmitError, RustSourceFile};
use crate::ir::{string_attribute_value, symbol_attribute_value, BoltModule, Cpu, Role};
use crate::schema::verify_cpu_schema;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1CpuProgram {
    pub role: Role,
    pub params: Stage1Params,
    pub transcript_squeezes: Vec<Stage1TranscriptSqueezePlan>,
    pub kernels: Vec<Stage1KernelPlan>,
    pub claims: Vec<Stage1SumcheckClaimPlan>,
    pub batches: Vec<Stage1SumcheckBatchPlan>,
    pub drivers: Vec<Stage1SumcheckDriverPlan>,
    pub instance_results: Vec<Stage1SumcheckInstanceResultPlan>,
    pub evals: Vec<Stage1SumcheckEvalPlan>,
    pub opening_claims: Vec<Stage1OpeningClaimPlan>,
    pub opening_batches: Vec<Stage1OpeningBatchPlan>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1Params {
    pub field: String,
    pub pcs: String,
    pub transcript: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1KernelPlan {
    pub symbol: String,
    pub relation: String,
    pub kind: String,
    pub backend: String,
    pub abi: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1TranscriptSqueezePlan {
    pub symbol: String,
    pub label: String,
    pub kind: String,
    pub count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1SumcheckClaimPlan {
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
pub struct Stage1SumcheckBatchPlan {
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
pub struct Stage1SumcheckDriverPlan {
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
pub struct Stage1SumcheckInstanceResultPlan {
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
pub struct Stage1SumcheckEvalPlan {
    pub symbol: String,
    pub source: String,
    pub name: String,
    pub index: usize,
    pub oracle: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1OpeningClaimPlan {
    pub symbol: String,
    pub oracle: String,
    pub domain: String,
    pub point_arity: usize,
    pub claim_kind: String,
    pub point_source: String,
    pub eval_source: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1OpeningBatchPlan {
    pub symbol: String,
    pub stage: String,
    pub proof_slot: String,
    pub policy: String,
    pub count: usize,
    pub ordered_claims: Vec<String>,
    pub claim_operands: Vec<String>,
}

pub fn stage1_cpu_program(module: &BoltModule<'_, Cpu>) -> Result<Stage1CpuProgram, EmitError> {
    verify_cpu_schema(module)?;
    let program = Stage1CpuProgram::from_module(module)?;
    program.verify_supported_target()?;
    Ok(program)
}

pub fn emit_stage1_rust(module: &BoltModule<'_, Cpu>) -> Result<RustSourceFile, EmitError> {
    let program = stage1_cpu_program(module)?;

    Ok(RustSourceFile {
        filename: program.filename().to_owned(),
        source: program.emit_source(),
    })
}

impl Stage1CpuProgram {
    fn from_module(module: &BoltModule<'_, Cpu>) -> Result<Self, EmitError> {
        let mut params = None;
        let mut transcript_squeezes = Vec::new();
        let mut kernels = Vec::new();
        let mut claims = Vec::new();
        let mut batches = Vec::new();
        let mut drivers = Vec::new();
        let mut instance_results = Vec::new();
        let mut evals = Vec::new();
        let mut opening_claims = Vec::new();
        let mut opening_batches = Vec::new();

        let mut operation = module.as_mlir_module().body().first_operation();
        while let Some(op) = operation {
            operation = op.next_in_block();
            match operation_name(op).as_str() {
                "cpu.params" => {
                    params = Some(Stage1Params {
                        field: symbol_attr(op, "field")?,
                        pcs: symbol_attr(op, "pcs")?,
                        transcript: symbol_attr(op, "transcript")?,
                    });
                }
                "cpu.kernel" => {
                    kernels.push(Stage1KernelPlan {
                        symbol: string_attr(op, "sym_name")?,
                        relation: symbol_attr(op, "relation")?,
                        kind: string_attr(op, "kind")?,
                        backend: string_attr(op, "backend")?,
                        abi: string_attr(op, "abi")?,
                    });
                }
                "cpu.transcript_squeeze" => {
                    transcript_squeezes.push(Stage1TranscriptSqueezePlan {
                        symbol: string_attr(op, "sym_name")?,
                        label: string_attr(op, "label")?,
                        kind: string_attr(op, "kind")?,
                        count: int_attr(op, "count")?,
                    });
                }
                "cpu.sumcheck_claim" => {
                    claims.push(Stage1SumcheckClaimPlan {
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
                    claims.push(Stage1SumcheckClaimPlan {
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
                    batches.push(Stage1SumcheckBatchPlan {
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
                    drivers.push(Stage1SumcheckDriverPlan {
                        symbol: string_attr(op, "sym_name")?,
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
                    drivers.push(Stage1SumcheckDriverPlan {
                        symbol: string_attr(op, "sym_name")?,
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
                "cpu.sumcheck_eval" => {
                    evals.push(Stage1SumcheckEvalPlan {
                        symbol: string_attr(op, "sym_name")?,
                        source: symbol_attr(op, "source")?,
                        name: symbol_attr(op, "name")?,
                        index: int_attr(op, "index")?,
                        oracle: symbol_attr(op, "oracle")?,
                    });
                }
                "cpu.sumcheck_instance_result" => {
                    instance_results.push(Stage1SumcheckInstanceResultPlan {
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
                "cpu.opening_claim" => {
                    opening_claims.push(Stage1OpeningClaimPlan {
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
                    opening_batches.push(Stage1OpeningBatchPlan {
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
            transcript_squeezes,
            kernels,
            claims,
            batches,
            drivers,
            instance_results,
            evals,
            opening_claims,
            opening_batches,
        })
    }

    fn verify_supported_target(&self) -> Result<(), EmitError> {
        require_supported_symbol("field", &self.params.field, "bn254_fr")?;
        require_supported_symbol("pcs", &self.params.pcs, "dory")?;
        require_supported_symbol("transcript", &self.params.transcript, "blake2b_transcript")?;
        self.verify_transcript_squeezes()?;
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
            if squeeze.kind != "challenge_vector" {
                return Err(EmitError::new(format!(
                    "stage1 transcript squeeze @{} has unsupported kind `{}`",
                    squeeze.symbol, squeeze.kind
                )));
            }
            if squeeze.count == 0 {
                return Err(EmitError::new(format!(
                    "stage1 transcript squeeze @{} has zero count",
                    squeeze.symbol
                )));
            }
        }
        Ok(())
    }

    fn verify_kernel_definitions(&self) -> Result<(), EmitError> {
        for kernel in &self.kernels {
            if kernel.backend != "cpu" {
                return Err(EmitError::new(format!(
                    "stage1 kernel @{} targets unsupported backend `{}`",
                    kernel.symbol, kernel.backend
                )));
            }
            if kernel.kind != "sumcheck" {
                return Err(EmitError::new(format!(
                    "stage1 kernel @{} has unsupported kind `{}`",
                    kernel.symbol, kernel.kind
                )));
            }
            let expected_abi = match kernel.relation.as_str() {
                "jolt.stage1.outer.uniskip" => "jolt_stage1_outer_uniskip",
                "jolt.stage1.outer.remaining" => "jolt_stage1_outer_remaining",
                _ => {
                    return Err(EmitError::new(format!(
                        "unsupported stage1 kernel relation @{}",
                        kernel.relation
                    )));
                }
            };
            if kernel.abi != expected_abi {
                return Err(EmitError::new(format!(
                    "stage1 kernel @{} ABI `{}` does not match relation @{}",
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
        let claims: BTreeMap<_, _> = self
            .claims
            .iter()
            .map(|claim| (claim.symbol.as_str(), claim))
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
            for claim in &batch.ordered_claims {
                let claim = claims.get(claim.as_str()).ok_or_else(|| {
                    EmitError::new(format!(
                        "sumcheck driver @{} references missing claim @{claim}",
                        driver.symbol
                    ))
                })?;
                if claim.kernel.as_deref() != Some(kernel) {
                    return Err(EmitError::new(format!(
                        "sumcheck driver @{} kernel @{kernel} differs from claim @{} kernel {:?}",
                        driver.symbol, claim.symbol, claim.kernel
                    )));
                }
            }
        }
        Ok(())
    }

    fn verify_verifier_driver_bindings(&self) -> Result<(), EmitError> {
        if !self.kernels.is_empty() {
            return Err(EmitError::new(
                "verifier stage1 program must not contain kernels",
            ));
        }
        let batches: BTreeMap<_, _> = self
            .batches
            .iter()
            .map(|batch| (batch.symbol.as_str(), batch))
            .collect();
        let claims: BTreeMap<_, _> = self
            .claims
            .iter()
            .map(|claim| (claim.symbol.as_str(), claim))
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
            let Some(relation) = driver.relation.as_deref() else {
                return Err(EmitError::new(format!(
                    "verifier sumcheck driver @{} is missing relation",
                    driver.symbol
                )));
            };
            if driver.kernel.is_some() {
                return Err(EmitError::new(format!(
                    "verifier sumcheck driver @{} must not carry kernel",
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
            for claim in &batch.ordered_claims {
                let claim = claims.get(claim.as_str()).ok_or_else(|| {
                    EmitError::new(format!(
                        "sumcheck driver @{} references missing claim @{claim}",
                        driver.symbol
                    ))
                })?;
                if claim.relation.as_deref() != Some(relation) {
                    return Err(EmitError::new(format!(
                        "sumcheck driver @{} relation @{relation} differs from claim @{} relation {:?}",
                        driver.symbol, claim.symbol, claim.relation
                    )));
                }
            }
        }
        Ok(())
    }

    fn verify_opening_flow(&self) -> Result<(), EmitError> {
        let drivers = symbols(self.drivers.iter().map(|driver| &driver.symbol));
        let instance_results = symbols(
            self.instance_results
                .iter()
                .map(|instance| &instance.symbol),
        );
        for instance in &self.instance_results {
            if !drivers.contains(&instance.source) {
                return Err(EmitError::new(format!(
                    "sumcheck instance result @{} references missing driver @{}",
                    instance.symbol, instance.source
                )));
            }
        }
        let mut point_sources = drivers.clone();
        point_sources.extend(instance_results);
        let evals = symbols(self.evals.iter().map(|eval| &eval.symbol));
        let openings = symbols(self.opening_claims.iter().map(|claim| &claim.symbol));
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
            if !evals.contains(&claim.eval_source) {
                return Err(EmitError::new(format!(
                    "opening claim @{} uses missing eval source @{}",
                    claim.symbol, claim.eval_source
                )));
            }
        }
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
            Role::Prover => "prove_stage1_outer.rs",
            Role::Verifier => "verify_stage1_outer.rs",
        }
    }

    fn emit_prover_imports() -> &'static str {
        "use jolt_field::Fr;\n\
         use jolt_kernels::stage1::{execute_stage1_program, Stage1CpuProgramPlan, Stage1ExecutionArtifacts, Stage1ExecutionMode, Stage1KernelError, Stage1KernelExecutor, Stage1KernelPlan, Stage1OpeningBatchPlan, Stage1OpeningClaimPlan, Stage1Params, Stage1SumcheckBatchPlan, Stage1SumcheckClaimPlan, Stage1SumcheckDriverPlan, Stage1SumcheckEvalPlan, Stage1SumcheckInstanceResultPlan, Stage1TranscriptSqueezePlan};\n\
         use jolt_transcript::{Blake2bTranscript, Transcript};"
    }

    fn emit_prover_types() -> &'static str {
        "pub type DefaultStage1Transcript = Blake2bTranscript<Fr>;\n"
    }

    fn emit_prover_constants(&self) -> String {
        let mut source = String::new();
        writeln!(
            source,
            "pub const STAGE1_PARAMS: Stage1Params = Stage1Params {{\n\
             \x20   field: {},\n\
             \x20   pcs: {},\n\
             \x20   transcript: {},\n\
             }};\n",
            rust_str(&self.params.field),
            rust_str(&self.params.pcs),
            rust_str(&self.params.transcript)
        )
        .expect("write generated Stage1 params");

        source.push_str(&self.emit_transcript_squeeze_constants());
        source.push_str(&self.emit_kernel_constants());
        source.push_str(&self.emit_sumcheck_claim_constants());
        source.push_str(&self.emit_sumcheck_batch_constants());
        source.push_str(&self.emit_sumcheck_driver_constants());
        source.push_str(&self.emit_sumcheck_instance_result_constants());
        source.push_str(&self.emit_sumcheck_eval_constants());
        source.push_str(&self.emit_opening_claim_constants());
        source.push_str(&self.emit_opening_batch_constants());
        source.push_str(
            "pub const STAGE1_PROGRAM: Stage1CpuProgramPlan = Stage1CpuProgramPlan {\n\
             \x20   params: STAGE1_PARAMS,\n\
             \x20   transcript_squeezes: STAGE1_TRANSCRIPT_SQUEEZES,\n\
             \x20   kernels: STAGE1_KERNELS,\n\
             \x20   claims: STAGE1_SUMCHECK_CLAIMS,\n\
             \x20   batches: STAGE1_SUMCHECK_BATCHES,\n\
             \x20   drivers: STAGE1_SUMCHECK_DRIVERS,\n\
             \x20   instance_results: STAGE1_SUMCHECK_INSTANCE_RESULTS,\n\
             \x20   evals: STAGE1_SUMCHECK_EVALS,\n\
             \x20   opening_claims: STAGE1_OPENING_CLAIMS,\n\
             \x20   opening_batches: STAGE1_OPENING_BATCHES,\n\
             };\n",
        );
        source
    }

    fn emit_sumcheck_instance_result_constants(&self) -> String {
        let instances = self
            .instance_results
            .iter()
            .map(|instance| {
                format!(
                    "    Stage1SumcheckInstanceResultPlan {{ symbol: {}, source: {}, claim: {}, relation: {}, index: {}, point_arity: {}, num_rounds: {}, round_offset: {}, point_order: {}, degree: {} }},",
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
            "pub const STAGE1_SUMCHECK_INSTANCE_RESULTS: &[Stage1SumcheckInstanceResultPlan] = &[\n{instances}\n];\n\n"
        )
    }

    fn emit_transcript_squeeze_constants(&self) -> String {
        let squeezes = self
            .transcript_squeezes
            .iter()
            .map(|squeeze| {
                format!(
                    "    Stage1TranscriptSqueezePlan {{ symbol: {}, label: {}, kind: {}, count: {} }},",
                    rust_str(&squeeze.symbol),
                    rust_str(&squeeze.label),
                    rust_str(&squeeze.kind),
                    squeeze.count,
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "pub const STAGE1_TRANSCRIPT_SQUEEZES: &[Stage1TranscriptSqueezePlan] = &[\n{squeezes}\n];\n\n"
        )
    }

    fn emit_kernel_constants(&self) -> String {
        let kernels = self
            .kernels
            .iter()
            .map(|kernel| {
                format!(
                    "    Stage1KernelPlan {{ symbol: {}, relation: {}, kind: {}, backend: {}, abi: {} }},",
                    rust_str(&kernel.symbol),
                    rust_str(&kernel.relation),
                    rust_str(&kernel.kind),
                    rust_str(&kernel.backend),
                    rust_str(&kernel.abi)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!("pub const STAGE1_KERNELS: &[Stage1KernelPlan] = &[\n{kernels}\n];\n\n")
    }

    fn emit_sumcheck_claim_constants(&self) -> String {
        let mut source = String::new();
        for (index, claim) in self.claims.iter().enumerate() {
            source.push_str(&emit_str_array(
                &format!("STAGE1_SUMCHECK_CLAIM_{index}_INPUT_OPENINGS"),
                &claim.input_openings,
            ));
        }
        let claims = self
            .claims
            .iter()
            .enumerate()
            .map(|(index, claim)| {
                let kernel = claim
                    .kernel
                    .as_deref()
                    .expect("prover sumcheck claim kernel verified");
                format!(
                    "    Stage1SumcheckClaimPlan {{ symbol: {}, stage: {}, domain: {}, num_rounds: {}, degree: {}, claim: {}, kernel: {}, claim_value: {}, input_openings: STAGE1_SUMCHECK_CLAIM_{index}_INPUT_OPENINGS }},",
                    rust_str(&claim.symbol),
                    rust_str(&claim.stage),
                    rust_str(&claim.domain),
                    claim.num_rounds,
                    claim.degree,
                    rust_str(&claim.claim),
                    rust_str(kernel),
                    rust_str(&claim.claim_value)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        writeln!(
            source,
            "pub const STAGE1_SUMCHECK_CLAIMS: &[Stage1SumcheckClaimPlan] = &[\n{claims}\n];\n"
        )
        .expect("write generated Stage1 sumcheck claims");
        source
    }

    fn emit_sumcheck_batch_constants(&self) -> String {
        let mut source = String::new();
        for (index, batch) in self.batches.iter().enumerate() {
            source.push_str(&emit_str_array(
                &format!("STAGE1_SUMCHECK_BATCH_{index}_ORDERED_CLAIMS"),
                &batch.ordered_claims,
            ));
            source.push_str(&emit_str_array(
                &format!("STAGE1_SUMCHECK_BATCH_{index}_CLAIM_OPERANDS"),
                &batch.claim_operands,
            ));
            source.push_str(&emit_usize_array(
                &format!("STAGE1_SUMCHECK_BATCH_{index}_ROUND_SCHEDULE"),
                &batch.round_schedule,
            ));
        }
        let batches = self
            .batches
            .iter()
            .enumerate()
            .map(|(index, batch)| {
                format!(
                    "    Stage1SumcheckBatchPlan {{ symbol: {}, stage: {}, proof_slot: {}, policy: {}, count: {}, ordered_claims: STAGE1_SUMCHECK_BATCH_{index}_ORDERED_CLAIMS, claim_operands: STAGE1_SUMCHECK_BATCH_{index}_CLAIM_OPERANDS, claim_label: {}, round_label: {}, round_schedule: STAGE1_SUMCHECK_BATCH_{index}_ROUND_SCHEDULE }},",
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
            "pub const STAGE1_SUMCHECK_BATCHES: &[Stage1SumcheckBatchPlan] = &[\n{batches}\n];\n"
        )
        .expect("write generated Stage1 sumcheck batches");
        source
    }

    fn emit_sumcheck_driver_constants(&self) -> String {
        let mut source = String::new();
        for (index, driver) in self.drivers.iter().enumerate() {
            source.push_str(&emit_usize_array(
                &format!("STAGE1_SUMCHECK_DRIVER_{index}_ROUND_SCHEDULE"),
                &driver.round_schedule,
            ));
        }
        let drivers = self
            .drivers
            .iter()
            .enumerate()
            .map(|(index, driver)| {
                let kernel = driver
                    .kernel
                    .as_deref()
                    .expect("prover sumcheck driver kernel verified");
                format!(
                    "    Stage1SumcheckDriverPlan {{ symbol: {}, stage: {}, proof_slot: {}, kernel: {}, batch: {}, policy: {}, round_schedule: STAGE1_SUMCHECK_DRIVER_{index}_ROUND_SCHEDULE, claim_label: {}, round_label: {}, num_rounds: {}, degree: {} }},",
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
            })
            .collect::<Vec<_>>()
            .join("\n");
        writeln!(
            source,
            "pub const STAGE1_SUMCHECK_DRIVERS: &[Stage1SumcheckDriverPlan] = &[\n{drivers}\n];\n"
        )
        .expect("write generated Stage1 sumcheck drivers");
        source
    }

    fn emit_sumcheck_eval_constants(&self) -> String {
        let evals = self
            .evals
            .iter()
            .map(|eval| {
                format!(
                    "    Stage1SumcheckEvalPlan {{ symbol: {}, source: {}, name: {}, index: {}, oracle: {} }},",
                    rust_str(&eval.symbol),
                    rust_str(&eval.source),
                    rust_str(&eval.name),
                    eval.index,
                    rust_str(&eval.oracle)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!("pub const STAGE1_SUMCHECK_EVALS: &[Stage1SumcheckEvalPlan] = &[\n{evals}\n];\n\n")
    }

    fn emit_opening_claim_constants(&self) -> String {
        let claims = self
            .opening_claims
            .iter()
            .map(|claim| {
                format!(
                    "    Stage1OpeningClaimPlan {{ symbol: {}, oracle: {}, domain: {}, point_arity: {}, claim_kind: {}, point_source: {}, eval_source: {} }},",
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
        format!("pub const STAGE1_OPENING_CLAIMS: &[Stage1OpeningClaimPlan] = &[\n{claims}\n];\n\n")
    }

    fn emit_opening_batch_constants(&self) -> String {
        let mut source = String::new();
        for (index, batch) in self.opening_batches.iter().enumerate() {
            source.push_str(&emit_str_array(
                &format!("STAGE1_OPENING_BATCH_{index}_ORDERED_CLAIMS"),
                &batch.ordered_claims,
            ));
            source.push_str(&emit_str_array(
                &format!("STAGE1_OPENING_BATCH_{index}_CLAIM_OPERANDS"),
                &batch.claim_operands,
            ));
        }
        let batches = self
            .opening_batches
            .iter()
            .enumerate()
            .map(|(index, batch)| {
                format!(
                    "    Stage1OpeningBatchPlan {{ symbol: {}, stage: {}, proof_slot: {}, policy: {}, count: {}, ordered_claims: STAGE1_OPENING_BATCH_{index}_ORDERED_CLAIMS, claim_operands: STAGE1_OPENING_BATCH_{index}_CLAIM_OPERANDS }},",
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
            "pub const STAGE1_OPENING_BATCHES: &[Stage1OpeningBatchPlan] = &[\n{batches}\n];\n"
        )
        .expect("write generated Stage1 opening batches");
        source
    }

    fn emit_verifier_imports() -> &'static str {
        "use jolt_field::{Field, Fr};\n\
         use jolt_sumcheck::{ClearRoundEncoding, ClearSumcheckPlan, SumcheckError, SumcheckProof};\n\
         use jolt_transcript::{Blake2bTranscript, Label, Transcript};"
    }

    fn emit_verifier_types() -> &'static str {
        r"pub type DefaultStage1Transcript = Blake2bTranscript<Fr>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage1Params {
    pub field: &'static str,
    pub pcs: &'static str,
    pub transcript: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage1SumcheckClaimPlan {
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
pub struct Stage1SumcheckBatchPlan {
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
pub struct Stage1SumcheckDriverPlan {
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
pub struct Stage1SumcheckInstanceResultPlan {
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
pub struct Stage1SumcheckEvalPlan {
    pub symbol: &'static str,
    pub source: &'static str,
    pub name: &'static str,
    pub index: usize,
    pub oracle: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage1OpeningClaimPlan {
    pub symbol: &'static str,
    pub oracle: &'static str,
    pub domain: &'static str,
    pub point_arity: usize,
    pub claim_kind: &'static str,
    pub point_source: &'static str,
    pub eval_source: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage1OpeningBatchPlan {
    pub symbol: &'static str,
    pub stage: &'static str,
    pub proof_slot: &'static str,
    pub policy: &'static str,
    pub count: usize,
    pub ordered_claims: &'static [&'static str],
    pub claim_operands: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage1TranscriptSqueezePlan {
    pub symbol: &'static str,
    pub label: &'static str,
    pub kind: &'static str,
    pub count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage1VerifierProgramPlan {
    pub params: Stage1Params,
    pub transcript_squeezes: &'static [Stage1TranscriptSqueezePlan],
    pub claims: &'static [Stage1SumcheckClaimPlan],
    pub batches: &'static [Stage1SumcheckBatchPlan],
    pub drivers: &'static [Stage1SumcheckDriverPlan],
    pub instance_results: &'static [Stage1SumcheckInstanceResultPlan],
    pub evals: &'static [Stage1SumcheckEvalPlan],
    pub opening_claims: &'static [Stage1OpeningClaimPlan],
    pub opening_batches: &'static [Stage1OpeningBatchPlan],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1NamedEval<F: Field> {
    pub name: &'static str,
    pub oracle: &'static str,
    pub value: F,
}

#[derive(Clone, Debug)]
pub struct Stage1SumcheckOutput<F: Field> {
    pub driver: &'static str,
    pub point: Vec<F>,
    pub evals: Vec<Stage1NamedEval<F>>,
    pub proof: SumcheckProof<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage1ChallengeVector<F: Field> {
    pub symbol: &'static str,
    pub values: Vec<F>,
}

#[derive(Clone, Debug)]
pub struct Stage1ExecutionArtifacts<F: Field> {
    pub challenge_vectors: Vec<Stage1ChallengeVector<F>>,
    pub sumchecks: Vec<Stage1SumcheckOutput<F>>,
    pub opening_batches: Vec<&'static Stage1OpeningBatchPlan>,
}

impl<F: Field> Default for Stage1ExecutionArtifacts<F> {
    fn default() -> Self {
        Self {
            challenge_vectors: Vec::new(),
            sumchecks: Vec::new(),
            opening_batches: Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Stage1Proof<F: Field> {
    pub sumchecks: Vec<Stage1SumcheckOutput<F>>,
}

#[derive(Debug)]
pub enum VerifyStage1Error {
    UnexpectedProofCount { expected: usize, got: usize },
    MissingProof { driver: &'static str },
    MissingBatch { driver: &'static str, batch: &'static str },
    MissingClaim { driver: &'static str, claim: &'static str },
    MissingDependency { driver: &'static str, dependency: &'static str },
    InvalidProof { driver: &'static str, reason: &'static str },
    UnsupportedRelation { relation: &'static str },
    Sumcheck { driver: &'static str, error: SumcheckError },
}
"
    }

    fn emit_verifier_constants(&self) -> String {
        let mut source = String::new();
        writeln!(
            source,
            "pub const STAGE1_PARAMS: Stage1Params = Stage1Params {{\n\
             \x20   field: {},\n\
             \x20   pcs: {},\n\
             \x20   transcript: {},\n\
             }};\n",
            rust_str(&self.params.field),
            rust_str(&self.params.pcs),
            rust_str(&self.params.transcript)
        )
        .expect("write generated Stage1 verifier params");

        source.push_str(&self.emit_transcript_squeeze_constants());
        source.push_str(&self.emit_verifier_sumcheck_claim_constants());
        source.push_str(&self.emit_sumcheck_batch_constants());
        source.push_str(&self.emit_verifier_sumcheck_driver_constants());
        source.push_str(&self.emit_sumcheck_instance_result_constants());
        source.push_str(&self.emit_sumcheck_eval_constants());
        source.push_str(&self.emit_opening_claim_constants());
        source.push_str(&self.emit_opening_batch_constants());
        source.push_str(
            "pub const STAGE1_PROGRAM: Stage1VerifierProgramPlan = Stage1VerifierProgramPlan {\n\
             \x20   params: STAGE1_PARAMS,\n\
             \x20   transcript_squeezes: STAGE1_TRANSCRIPT_SQUEEZES,\n\
             \x20   claims: STAGE1_SUMCHECK_CLAIMS,\n\
             \x20   batches: STAGE1_SUMCHECK_BATCHES,\n\
             \x20   drivers: STAGE1_SUMCHECK_DRIVERS,\n\
             \x20   instance_results: STAGE1_SUMCHECK_INSTANCE_RESULTS,\n\
             \x20   evals: STAGE1_SUMCHECK_EVALS,\n\
             \x20   opening_claims: STAGE1_OPENING_CLAIMS,\n\
             \x20   opening_batches: STAGE1_OPENING_BATCHES,\n\
             };\n",
        );
        source
    }

    fn emit_verifier_sumcheck_claim_constants(&self) -> String {
        let mut source = String::new();
        for (index, claim) in self.claims.iter().enumerate() {
            source.push_str(&emit_str_array(
                &format!("STAGE1_SUMCHECK_CLAIM_{index}_INPUT_OPENINGS"),
                &claim.input_openings,
            ));
        }
        let claims = self
            .claims
            .iter()
            .enumerate()
            .map(|(index, claim)| {
                let relation = claim
                    .relation
                    .as_deref()
                    .expect("verifier sumcheck claim relation verified");
                format!(
                    "    Stage1SumcheckClaimPlan {{ symbol: {}, stage: {}, domain: {}, num_rounds: {}, degree: {}, claim: {}, relation: {}, claim_value: {}, input_openings: STAGE1_SUMCHECK_CLAIM_{index}_INPUT_OPENINGS }},",
                    rust_str(&claim.symbol),
                    rust_str(&claim.stage),
                    rust_str(&claim.domain),
                    claim.num_rounds,
                    claim.degree,
                    rust_str(&claim.claim),
                    rust_str(relation),
                    rust_str(&claim.claim_value)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        writeln!(
            source,
            "pub const STAGE1_SUMCHECK_CLAIMS: &[Stage1SumcheckClaimPlan] = &[\n{claims}\n];\n"
        )
        .expect("write generated Stage1 verifier sumcheck claims");
        source
    }

    fn emit_verifier_sumcheck_driver_constants(&self) -> String {
        let mut source = String::new();
        for (index, driver) in self.drivers.iter().enumerate() {
            source.push_str(&emit_usize_array(
                &format!("STAGE1_SUMCHECK_DRIVER_{index}_ROUND_SCHEDULE"),
                &driver.round_schedule,
            ));
        }
        let drivers = self
            .drivers
            .iter()
            .enumerate()
            .map(|(index, driver)| {
                let relation = driver
                    .relation
                    .as_deref()
                    .expect("verifier sumcheck driver relation verified");
                format!(
                    "    Stage1SumcheckDriverPlan {{ symbol: {}, stage: {}, proof_slot: {}, relation: {}, batch: {}, policy: {}, round_schedule: STAGE1_SUMCHECK_DRIVER_{index}_ROUND_SCHEDULE, claim_label: {}, round_label: {}, num_rounds: {}, degree: {} }},",
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
            })
            .collect::<Vec<_>>()
            .join("\n");
        writeln!(
            source,
            "pub const STAGE1_SUMCHECK_DRIVERS: &[Stage1SumcheckDriverPlan] = &[\n{drivers}\n];\n"
        )
        .expect("write generated Stage1 verifier sumcheck drivers");
        source
    }

    fn emit_prover_entrypoint() -> &'static str {
        r"pub fn prove_stage1_outer<E, T>(
    executor: &mut E,
    transcript: &mut T,
) -> Result<Stage1ExecutionArtifacts<Fr>, Stage1KernelError>
where
    E: Stage1KernelExecutor<Fr>,
    T: Transcript<Challenge = Fr>,
{
    execute_stage1_program(
        &STAGE1_PROGRAM,
        Stage1ExecutionMode::Prover,
        executor,
        transcript,
    )
}
"
    }

    fn emit_verifier_entrypoint() -> &'static str {
        r#"pub fn verify_stage1_outer<T>(
    proof: &Stage1Proof<Fr>,
    transcript: &mut T,
) -> Result<Stage1ExecutionArtifacts<Fr>, VerifyStage1Error>
where
    T: Transcript<Challenge = Fr>,
{
    if proof.sumchecks.len() != STAGE1_PROGRAM.drivers.len() {
        return Err(VerifyStage1Error::UnexpectedProofCount {
            expected: STAGE1_PROGRAM.drivers.len(),
            got: proof.sumchecks.len(),
        });
    }
    let mut artifacts = Stage1ExecutionArtifacts::default();
    for squeeze in STAGE1_PROGRAM.transcript_squeezes {
        let values = transcript.challenge_vector(squeeze.count);
        artifacts.challenge_vectors.push(Stage1ChallengeVector {
            symbol: squeeze.symbol,
            values,
        });
    }
    for (index, driver) in STAGE1_PROGRAM.drivers.iter().enumerate() {
        let proof = proof.sumchecks.get(index).ok_or(VerifyStage1Error::MissingProof {
            driver: driver.symbol,
        })?;
        let output = verify_stage1_driver(driver, proof, &artifacts.sumchecks, transcript)?;
        artifacts.sumchecks.push(output);
    }
    artifacts
        .opening_batches
        .extend(STAGE1_PROGRAM.opening_batches.iter());
    Ok(artifacts)
}

fn verify_stage1_driver<T>(
    driver: &'static Stage1SumcheckDriverPlan,
    proof: &Stage1SumcheckOutput<Fr>,
    completed: &[Stage1SumcheckOutput<Fr>],
    transcript: &mut T,
) -> Result<Stage1SumcheckOutput<Fr>, VerifyStage1Error>
where
    T: Transcript<Challenge = Fr>,
{
    if proof.driver != driver.symbol {
        return Err(VerifyStage1Error::InvalidProof {
            driver: driver.symbol,
            reason: "driver symbol mismatch",
        });
    }
    match driver.relation {
        "jolt.stage1.outer.uniskip" => verify_outer_uniskip(driver, proof, transcript),
        "jolt.stage1.outer.remaining" => {
            verify_outer_remaining(driver, proof, completed, transcript)
        }
        relation => Err(VerifyStage1Error::UnsupportedRelation { relation }),
    }
}

fn verify_outer_uniskip<T>(
    driver: &'static Stage1SumcheckDriverPlan,
    proof: &Stage1SumcheckOutput<Fr>,
    transcript: &mut T,
) -> Result<Stage1SumcheckOutput<Fr>, VerifyStage1Error>
where
    T: Transcript<Challenge = Fr>,
{
    let plan = ClearSumcheckPlan {
        num_vars: driver.num_rounds,
        degree: driver.degree,
        round_label: driver.round_label.as_bytes(),
        round_encoding: ClearRoundEncoding::Full,
    };
    let output = plan
        .verify(Fr::from_u64(0), &proof.proof, transcript)
        .map_err(|error| VerifyStage1Error::Sumcheck {
            driver: driver.symbol,
            error,
        })?;
    let eval = output.final_eval;
    let point = output.point;
    if !proof.point.is_empty() && proof.point != point {
        return Err(VerifyStage1Error::InvalidProof {
            driver: driver.symbol,
            reason: "uniskip point mismatch",
        });
    }
    validate_eval_shape(driver, &proof.evals, Some(eval))?;
    append_labeled_scalar(transcript, "opening_claim", &eval);
    Ok(Stage1SumcheckOutput {
        driver: driver.symbol,
        point,
        evals: driver_evals(driver.symbol, eval),
        proof: proof.proof.clone(),
    })
}

fn verify_outer_remaining<T>(
    driver: &'static Stage1SumcheckDriverPlan,
    proof: &Stage1SumcheckOutput<Fr>,
    completed: &[Stage1SumcheckOutput<Fr>],
    transcript: &mut T,
) -> Result<Stage1SumcheckOutput<Fr>, VerifyStage1Error>
where
    T: Transcript<Challenge = Fr>,
{
    let input_claim = completed
        .iter()
        .find(|output| output.driver == "stage1.uniskip.sumcheck")
        .and_then(|output| output.evals.first())
        .map(|eval| eval.value)
        .ok_or(VerifyStage1Error::MissingDependency {
            driver: driver.symbol,
            dependency: "stage1.uniskip.eval",
        })?;
    append_labeled_scalar(transcript, driver.claim_label, &input_claim);
    let batching_coeff = transcript.challenge();
    let plan = ClearSumcheckPlan {
        num_vars: driver.num_rounds,
        degree: driver.degree,
        round_label: driver.round_label.as_bytes(),
        round_encoding: ClearRoundEncoding::Compressed,
    };
    let output = plan
        .verify(input_claim * batching_coeff, &proof.proof, transcript)
        .map_err(|error| VerifyStage1Error::Sumcheck {
            driver: driver.symbol,
            error,
        })?;
    let point = output.point;
    if !proof.point.is_empty() && proof.point != point {
        return Err(VerifyStage1Error::InvalidProof {
            driver: driver.symbol,
            reason: "outer remaining point mismatch",
        });
    }
    validate_eval_shape(driver, &proof.evals, None)?;
    append_opening_claims(transcript, &proof.evals);
    Ok(Stage1SumcheckOutput {
        driver: driver.symbol,
        point,
        evals: proof.evals.clone(),
        proof: proof.proof.clone(),
    })
}

fn driver_evals(driver: &'static str, value: Fr) -> Vec<Stage1NamedEval<Fr>> {
    STAGE1_PROGRAM
        .evals
        .iter()
        .filter(|eval| eval.source == driver)
        .map(|eval| Stage1NamedEval {
            name: eval.name,
            oracle: eval.oracle,
            value,
        })
        .collect()
}

fn validate_eval_shape(
    driver: &'static Stage1SumcheckDriverPlan,
    actual: &[Stage1NamedEval<Fr>],
    expected_value: Option<Fr>,
) -> Result<(), VerifyStage1Error> {
    let expected = STAGE1_PROGRAM
        .evals
        .iter()
        .filter(|eval| eval.source == driver.symbol)
        .collect::<Vec<_>>();
    if actual.len() != expected.len() {
        return Err(VerifyStage1Error::InvalidProof {
            driver: driver.symbol,
            reason: "eval count mismatch",
        });
    }
    for (actual, expected) in actual.iter().zip(expected) {
        if actual.name != expected.name {
            return Err(VerifyStage1Error::InvalidProof {
                driver: driver.symbol,
                reason: "eval name mismatch",
            });
        }
        if actual.oracle != expected.oracle {
            return Err(VerifyStage1Error::InvalidProof {
                driver: driver.symbol,
                reason: "eval oracle mismatch",
            });
        }
        if expected_value.is_some_and(|value| actual.value != value) {
            return Err(VerifyStage1Error::InvalidProof {
                driver: driver.symbol,
                reason: "eval value mismatch",
            });
        }
    }
    Ok(())
}

fn append_labeled_scalar<T>(transcript: &mut T, label: &'static str, scalar: &Fr)
where
    T: Transcript<Challenge = Fr>,
{
    transcript.append(&Label(label.as_bytes()));
    transcript.append(scalar);
}

fn append_opening_claims<T>(transcript: &mut T, evals: &[Stage1NamedEval<Fr>])
where
    T: Transcript<Challenge = Fr>,
{
    for eval in evals {
        append_labeled_scalar(transcript, "opening_claim", &eval.value);
    }
}
"#
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

fn operation_name(operation: OperationRef<'_, '_>) -> String {
    operation
        .name()
        .as_string_ref()
        .as_str()
        .unwrap_or("<invalid-operation-name>")
        .to_owned()
}

fn require_supported_symbol(kind: &str, actual: &str, expected: &str) -> Result<(), EmitError> {
    if actual == expected {
        Ok(())
    } else {
        Err(EmitError::new(format!(
            "unsupported {kind} @{actual}; Rust stage1 emitter currently supports @{expected}"
        )))
    }
}
