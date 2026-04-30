use std::error::Error;
use std::fmt::{self, Display, Formatter, Write as _};

use melior::ir::block::BlockLike;
use melior::ir::operation::{OperationLike, OperationResult};
use melior::ir::{Attribute, OperationRef};

use crate::ir::{string_attribute_value, symbol_attribute_value, BoltModule, Cpu, Role};
use crate::schema::{verify_cpu_schema, SchemaError};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RustSourceFile {
    pub filename: String,
    pub source: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommitmentCpuProgram {
    pub role: Role,
    pub params: CommitmentParams,
    pub oracle_plans: Vec<OraclePlan>,
    pub batch_plans: Vec<CommitmentBatchPlan>,
    pub optional_plans: Vec<OptionalCommitmentPlan>,
    pub transcript_steps: Vec<TranscriptStep>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommitmentParams {
    pub field: String,
    pub pcs: String,
    pub transcript: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OraclePlan {
    pub oracle: String,
    pub source: String,
    pub domain: String,
    pub num_vars: usize,
    pub generation: OracleGeneration,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OracleGeneration {
    Reference,
    DenseTrace {
        padding: String,
    },
    OneHotChunk {
        trace_num_vars: usize,
        chunk: usize,
        num_chunks: usize,
        chunk_bits: usize,
        padding: String,
        layout: String,
    },
    OptionalAdvice {
        skip_policy: OptionalSkipPolicy,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommitmentBatchPlan {
    pub artifact: String,
    pub pcs: String,
    pub oracle_family: String,
    pub label: String,
    pub oracles: Vec<String>,
    pub count: usize,
    pub domain: String,
    pub num_vars: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OptionalCommitmentPlan {
    pub artifact: String,
    pub pcs: String,
    pub oracle: String,
    pub label: String,
    pub domain: String,
    pub num_vars: usize,
    pub skip_policy: OptionalSkipPolicy,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OptionalSkipPolicy {
    MissingOrZero,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TranscriptStep {
    pub label: String,
    pub source: String,
    pub optional: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmitError {
    message: String,
}

impl EmitError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Display for EmitError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for EmitError {}

impl From<SchemaError> for EmitError {
    fn from(error: SchemaError) -> Self {
        Self::new(error.to_string())
    }
}

pub fn emit_commitment_rust(module: &BoltModule<'_, Cpu>) -> Result<RustSourceFile, EmitError> {
    let program = commitment_cpu_program(module)?;

    Ok(RustSourceFile {
        filename: program.filename().to_owned(),
        source: program.emit_source(),
    })
}

pub fn commitment_cpu_program(
    module: &BoltModule<'_, Cpu>,
) -> Result<CommitmentCpuProgram, EmitError> {
    verify_cpu_schema(module)?;
    let program = CommitmentCpuProgram::from_module(module)?;
    program.verify_supported_target()?;
    Ok(program)
}

impl CommitmentCpuProgram {
    fn from_module(module: &BoltModule<'_, Cpu>) -> Result<Self, EmitError> {
        let mut params = None;
        let mut oracle_plans = Vec::new();
        let mut batch_plans = Vec::new();
        let mut optional_plans = Vec::new();
        let mut transcript_steps = Vec::new();

        let mut operation = module.as_mlir_module().body().first_operation();
        while let Some(op) = operation {
            operation = op.next_in_block();
            match operation_name(op).as_str() {
                "cpu.params" => {
                    params = Some(CommitmentParams {
                        field: symbol_attr(op, "field")?,
                        pcs: symbol_attr(op, "pcs")?,
                        transcript: symbol_attr(op, "transcript")?,
                    });
                }
                "cpu.oracle_dense_trace" => {
                    oracle_plans.push(OraclePlan {
                        oracle: symbol_attr(op, "oracle")?,
                        source: symbol_attr(op, "source")?,
                        domain: symbol_attr(op, "domain")?,
                        num_vars: int_attr(op, "num_vars")?,
                        generation: OracleGeneration::DenseTrace {
                            padding: string_attr(op, "padding")?,
                        },
                    });
                }
                "cpu.oracle_one_hot_chunk" => {
                    oracle_plans.push(OraclePlan {
                        oracle: symbol_attr(op, "oracle")?,
                        source: symbol_attr(op, "source")?,
                        domain: symbol_attr(op, "domain")?,
                        num_vars: int_attr(op, "num_vars")?,
                        generation: OracleGeneration::OneHotChunk {
                            trace_num_vars: int_attr(op, "trace_num_vars")?,
                            chunk: int_attr(op, "chunk")?,
                            num_chunks: int_attr(op, "num_chunks")?,
                            chunk_bits: int_attr(op, "chunk_bits")?,
                            padding: string_attr(op, "padding")?,
                            layout: string_attr(op, "layout")?,
                        },
                    });
                }
                "cpu.oracle_optional_advice" => {
                    oracle_plans.push(OraclePlan {
                        oracle: symbol_attr(op, "oracle")?,
                        source: symbol_attr(op, "source")?,
                        domain: symbol_attr(op, "domain")?,
                        num_vars: int_attr(op, "num_vars")?,
                        generation: OracleGeneration::OptionalAdvice {
                            skip_policy: skip_policy_attr(op, "skip_policy")?,
                        },
                    });
                }
                "cpu.oracle_ref" => {
                    oracle_plans.push(OraclePlan {
                        oracle: symbol_attr(op, "oracle")?,
                        source: String::new(),
                        domain: symbol_attr(op, "domain")?,
                        num_vars: int_attr(op, "num_vars")?,
                        generation: OracleGeneration::Reference,
                    });
                }
                "cpu.pcs_commit_batch" | "cpu.pcs_receive_batch" => {
                    batch_plans.push(CommitmentBatchPlan {
                        artifact: symbol_attr(op, "artifact")?,
                        pcs: symbol_attr(op, "pcs")?,
                        oracle_family: symbol_attr(op, "oracle_family")?,
                        label: string_attr(op, "label")?,
                        oracles: symbol_array_attr(op, "ordered_oracles")?,
                        count: int_attr(op, "count")?,
                        domain: symbol_attr(op, "domain")?,
                        num_vars: int_attr(op, "num_vars")?,
                    });
                }
                "cpu.pcs_commit_optional" | "cpu.pcs_receive_optional" => {
                    optional_plans.push(OptionalCommitmentPlan {
                        artifact: symbol_attr(op, "artifact")?,
                        pcs: symbol_attr(op, "pcs")?,
                        oracle: symbol_attr(op, "oracle")?,
                        label: string_attr(op, "label")?,
                        domain: symbol_attr(op, "domain")?,
                        num_vars: int_attr(op, "num_vars")?,
                        skip_policy: skip_policy_attr(op, "skip_policy")?,
                    });
                }
                "cpu.transcript_absorb" => {
                    transcript_steps.push(TranscriptStep {
                        label: string_attr(op, "label")?,
                        source: transcript_artifact_source(op)?,
                        optional: bool_attr(op, "optional")?,
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
            oracle_plans,
            batch_plans,
            optional_plans,
            transcript_steps,
        })
    }

    fn verify_supported_target(&self) -> Result<(), EmitError> {
        require_supported_symbol("field", &self.params.field, "bn254_fr")?;
        require_supported_symbol("pcs", &self.params.pcs, "dory")?;
        require_supported_symbol("transcript", &self.params.transcript, "blake2b_transcript")?;
        for plan in &self.batch_plans {
            require_supported_symbol("batch pcs", &plan.pcs, "dory")?;
        }
        for plan in &self.optional_plans {
            require_supported_symbol("optional pcs", &plan.pcs, "dory")?;
        }
        Ok(())
    }

    fn emit_source(&self) -> String {
        let mut source = String::new();
        source.push_str("#![allow(dead_code)]\n\n");
        source.push_str(self.emit_imports());
        source.push_str("\n\n");
        source.push_str(&self.emit_types());
        source.push('\n');
        source.push_str(&self.emit_constants());
        source.push('\n');
        source.push_str(self.emit_entrypoint());
        source
    }

    fn filename(&self) -> &'static str {
        match self.role {
            Role::Prover => "prove_commitment_phase.rs",
            Role::Verifier => "verify_commitment_phase.rs",
        }
    }

    fn emit_imports(&self) -> &'static str {
        match self.role {
            Role::Prover => {
                "use std::borrow::Cow;\n\
                 \n\
                 use jolt_dory::{DoryCommitment, DoryHint, DoryProverSetup, DoryScheme};\n\
                 use jolt_field::{Field, Fr};\n\
                 use jolt_openings::StreamingCommitment;\n\
                 use jolt_transcript::{AppendToTranscript, Blake2bTranscript, LabelWithCount, Transcript};\n\
                 use jolt_witness_v2::{dense_i128_column_to_field, one_hot_chunk_address_major, optional_field_oracle};"
            }
            Role::Verifier => {
                "use jolt_dory::DoryCommitment;\n\
                 use jolt_field::Fr;\n\
                 use jolt_transcript::{AppendToTranscript, Blake2bTranscript, LabelWithCount, Transcript};"
            }
        }
    }

    fn emit_types(&self) -> String {
        match self.role {
            Role::Prover => {
                let mut types = Self::emit_prover_types().to_owned();
                types.push('\n');
                types.push_str(&self.emit_oracle_store_types());
                types
            }
            Role::Verifier => Self::emit_verifier_types().to_owned(),
        }
    }

    fn emit_prover_types() -> &'static str {
        r"pub type DefaultCommitmentTranscript = Blake2bTranscript<Fr>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommitmentParams {
    pub field: &'static str,
    pub pcs: &'static str,
    pub transcript: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OraclePlan {
    pub oracle: &'static str,
    pub domain: &'static str,
    pub num_vars: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommitmentBatchPlan {
    pub artifact: &'static str,
    pub pcs: &'static str,
    pub oracle_family: &'static str,
    pub label: &'static str,
    pub oracles: &'static [&'static str],
    pub count: usize,
    pub domain: &'static str,
    pub num_vars: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptionalSkipPolicy {
    MissingOrZero,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OptionalCommitmentPlan {
    pub artifact: &'static str,
    pub pcs: &'static str,
    pub oracle: &'static str,
    pub label: &'static str,
    pub domain: &'static str,
    pub num_vars: usize,
    pub skip_policy: OptionalSkipPolicy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TranscriptStep {
    pub label: &'static str,
    pub source: &'static str,
    pub optional: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommitmentRecord {
    pub artifact: &'static str,
    pub oracle: &'static str,
    pub label: &'static str,
    pub num_vars: usize,
}

#[derive(Clone, Debug)]
pub struct OracleOpeningHint {
    pub oracle: &'static str,
    pub hint: DoryHint,
}

#[derive(Clone, Debug, Default)]
pub struct CommitmentArtifacts {
    pub commitments: Vec<Option<DoryCommitment>>,
    pub records: Vec<CommitmentRecord>,
    pub hints: Vec<OracleOpeningHint>,
}

pub trait CommitmentInputProvider {
    fn materialize(&mut self, oracle: &'static str) -> Option<Cow<'_, [Fr]>>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CommitmentPhaseError {
    MissingOracle { oracle: &'static str },
    MissingTranscriptSource { source: &'static str },
    PlanCountMismatch { artifact: &'static str, expected: usize, actual: usize },
    OracleTooLarge { oracle: &'static str, len: usize, target_len: usize },
    TargetSizeOverflow { num_vars: usize },
}"
    }

    fn emit_oracle_store_types(&self) -> String {
        let input_type = r"
pub struct CommitmentOracleInputs<'a> {
    pub rd_inc: &'a [i128],
    pub ram_inc: &'a [i128],
    pub instruction_keys: &'a [Option<u128>],
    pub ram_addresses: &'a [Option<u128>],
    pub bytecode_indices: &'a [Option<u128>],
    pub untrusted_advice: Option<&'a [Fr]>,
    pub trusted_advice: Option<&'a [Fr]>,
}
";
        let fields = self
            .oracle_plans
            .iter()
            .filter(|plan| !matches!(plan.generation, OracleGeneration::Reference))
            .map(|plan| {
                let field = rust_field_name(&plan.oracle);
                let ty = match &plan.generation {
                    OracleGeneration::Reference => unreachable!("reference oracles are filtered"),
                    OracleGeneration::OptionalAdvice { .. } => "Option<Vec<Fr>>",
                    _ => "Vec<Fr>",
                };
                format!("    pub {field}: {ty},")
            })
            .collect::<Vec<_>>()
            .join("\n");
        let provider_arms = self
            .oracle_plans
            .iter()
            .filter(|plan| !matches!(plan.generation, OracleGeneration::Reference))
            .map(|plan| {
                let field = rust_field_name(&plan.oracle);
                match &plan.generation {
                    OracleGeneration::Reference => unreachable!("reference oracles are filtered"),
                    OracleGeneration::OptionalAdvice { .. } => format!(
                        "            {} => self.{field}.as_deref().map(Cow::Borrowed),",
                        rust_str(&plan.oracle)
                    ),
                    _ => format!(
                        "            {} => Some(Cow::Borrowed(&self.{field})),",
                        rust_str(&plan.oracle)
                    ),
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        let initializers = self
            .oracle_plans
            .iter()
            .filter(|plan| !matches!(plan.generation, OracleGeneration::Reference))
            .map(|plan| {
                let field = rust_field_name(&plan.oracle);
                let expr = Self::oracle_initializer(plan);
                format!("        {field}: {expr},")
            })
            .collect::<Vec<_>>()
            .join("\n");

        format!(
            "{input_type}
#[derive(Clone, Debug, Default)]
pub struct CommitmentOracles {{
{fields}
}}

impl CommitmentInputProvider for CommitmentOracles {{
    fn materialize(&mut self, oracle: &'static str) -> Option<Cow<'_, [Fr]>> {{
        match oracle {{
{provider_arms}
            _ => None,
        }}
    }}
}}

pub fn build_commitment_oracles(
    inputs: &CommitmentOracleInputs<'_>,
) -> Result<CommitmentOracles, CommitmentPhaseError> {{
    Ok(CommitmentOracles {{
{initializers}
    }})
}}
"
        )
    }

    fn oracle_initializer(plan: &OraclePlan) -> String {
        match &plan.generation {
            OracleGeneration::Reference => {
                panic!("reference oracle @{} has no prover initializer", plan.oracle)
            }
            OracleGeneration::DenseTrace { .. } => format!(
                "dense_i128_column_to_field(inputs.{}, target_len({})?)",
                rust_input_field(&plan.source),
                plan.num_vars
            ),
            OracleGeneration::OneHotChunk {
                trace_num_vars,
                chunk,
                num_chunks,
                chunk_bits,
                padding,
                ..
            } => format!(
                "one_hot_chunk_address_major(inputs.{}, {chunk}, {num_chunks}, {chunk_bits}, target_len({trace_num_vars})?, {})",
                rust_input_field(&plan.source),
                rust_padding_value(padding)
            ),
            OracleGeneration::OptionalAdvice { .. } => format!(
                "optional_field_oracle(inputs.{}, target_len({})?)",
                rust_input_field(&plan.source),
                plan.num_vars
            ),
        }
    }

    fn emit_verifier_types() -> &'static str {
        r"pub type DefaultCommitmentTranscript = Blake2bTranscript<Fr>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommitmentParams {
    pub field: &'static str,
    pub pcs: &'static str,
    pub transcript: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OraclePlan {
    pub oracle: &'static str,
    pub domain: &'static str,
    pub num_vars: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommitmentBatchPlan {
    pub artifact: &'static str,
    pub pcs: &'static str,
    pub oracle_family: &'static str,
    pub label: &'static str,
    pub oracles: &'static [&'static str],
    pub count: usize,
    pub domain: &'static str,
    pub num_vars: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptionalSkipPolicy {
    MissingOrZero,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OptionalCommitmentPlan {
    pub artifact: &'static str,
    pub pcs: &'static str,
    pub oracle: &'static str,
    pub label: &'static str,
    pub domain: &'static str,
    pub num_vars: usize,
    pub skip_policy: OptionalSkipPolicy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TranscriptStep {
    pub label: &'static str,
    pub source: &'static str,
    pub optional: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommitmentRecord {
    pub artifact: &'static str,
    pub oracle: &'static str,
    pub label: &'static str,
    pub num_vars: usize,
}

#[derive(Clone, Debug, Default)]
pub struct CommitmentArtifacts {
    pub commitments: Vec<Option<DoryCommitment>>,
    pub records: Vec<CommitmentRecord>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CommitmentPhaseError {
    MissingProofCommitment { oracle: &'static str },
    MissingProofCommitmentSlot { artifact: &'static str, oracle: &'static str },
    MissingTranscriptSource { source: &'static str },
    PlanCountMismatch { artifact: &'static str, expected: usize, actual: usize },
    ProofCommitmentCountMismatch { expected: usize, actual: usize },
}"
    }

    fn emit_constants(&self) -> String {
        let mut source = String::new();
        writeln!(
            source,
            "pub const COMMITMENT_PARAMS: CommitmentParams = CommitmentParams {{\n\
             \x20   field: {},\n\
             \x20   pcs: {},\n\
             \x20   transcript: {},\n\
             }};\n",
            rust_str(&self.params.field),
            rust_str(&self.params.pcs),
            rust_str(&self.params.transcript)
        )
        .expect("write generated Rust params");

        let oracle_plans = self
            .oracle_plans
            .iter()
            .map(|plan| {
                format!(
                    "    OraclePlan {{ oracle: {}, domain: {}, num_vars: {} }},",
                    rust_str(&plan.oracle),
                    rust_str(&plan.domain),
                    plan.num_vars
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        writeln!(
            source,
            "pub const ORACLE_PLANS: &[OraclePlan] = &[\n{oracle_plans}\n];\n"
        )
        .expect("write generated Rust oracle plans");

        for (index, plan) in self.batch_plans.iter().enumerate() {
            let oracles = plan
                .oracles
                .iter()
                .map(|oracle| format!("    {},", rust_str(oracle)))
                .collect::<Vec<_>>()
                .join("\n");
            writeln!(
                source,
                "pub const COMMITMENT_BATCH_{index}_ORACLES: &[&str] = &[\n{oracles}\n];\n"
            )
            .expect("write generated Rust oracle constants");
        }

        let batch_plans = self
            .batch_plans
            .iter()
            .enumerate()
            .map(|(index, plan)| {
                format!(
                    "    CommitmentBatchPlan {{ artifact: {}, pcs: {}, oracle_family: {}, label: {}, oracles: COMMITMENT_BATCH_{index}_ORACLES, count: {}, domain: {}, num_vars: {} }},",
                    rust_str(&plan.artifact),
                    rust_str(&plan.pcs),
                    rust_str(&plan.oracle_family),
                    rust_str(&plan.label),
                    plan.count,
                    rust_str(&plan.domain),
                    plan.num_vars
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        writeln!(
            source,
            "pub const COMMITMENT_BATCH_PLANS: &[CommitmentBatchPlan] = &[\n{batch_plans}\n];\n"
        )
        .expect("write generated Rust batch plans");

        let optional_plans = self
            .optional_plans
            .iter()
            .map(|plan| {
                format!(
                    "    OptionalCommitmentPlan {{ artifact: {}, pcs: {}, oracle: {}, label: {}, domain: {}, num_vars: {}, skip_policy: {} }},",
                    rust_str(&plan.artifact),
                    rust_str(&plan.pcs),
                    rust_str(&plan.oracle),
                    rust_str(&plan.label),
                    rust_str(&plan.domain),
                    plan.num_vars,
                    plan.skip_policy.rust_variant()
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        writeln!(
            source,
            "pub const OPTIONAL_COMMITMENT_PLANS: &[OptionalCommitmentPlan] = &[\n{optional_plans}\n];\n"
        )
        .expect("write generated Rust optional plans");

        let steps = self
            .transcript_steps
            .iter()
            .map(|step| {
                format!(
                    "    TranscriptStep {{ label: {}, source: {}, optional: {} }},",
                    rust_str(&step.label),
                    rust_str(&step.source),
                    step.optional
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        writeln!(
            source,
            "pub const TRANSCRIPT_PLAN: &[TranscriptStep] = &[\n{steps}\n];"
        )
        .expect("write generated Rust transcript plan");

        source
    }

    fn emit_entrypoint(&self) -> &'static str {
        match self.role {
            Role::Prover => Self::emit_prover_entrypoint(),
            Role::Verifier => Self::emit_verifier_entrypoint(),
        }
    }

    fn emit_prover_entrypoint() -> &'static str {
        r"pub fn prove_commitment_phase<I, T>(
    inputs: &mut I,
    prover_setup: &DoryProverSetup,
    transcript: &mut T,
) -> Result<CommitmentArtifacts, CommitmentPhaseError>
where
    I: CommitmentInputProvider,
    T: Transcript<Challenge = Fr>,
{
    let mut artifacts = CommitmentArtifacts::default();
    for plan in COMMITMENT_BATCH_PLANS {
        commit_batch(inputs, prover_setup, &mut artifacts, plan)?;
    }
    for plan in OPTIONAL_COMMITMENT_PLANS {
        commit_optional(inputs, prover_setup, &mut artifacts, plan)?;
    }
    absorb_transcript(&artifacts, transcript)?;
    Ok(artifacts)
}

fn commit_batch<I>(
    inputs: &mut I,
    prover_setup: &DoryProverSetup,
    artifacts: &mut CommitmentArtifacts,
    plan: &CommitmentBatchPlan,
) -> Result<(), CommitmentPhaseError>
where
    I: CommitmentInputProvider,
{
    if plan.count != plan.oracles.len() {
        return Err(CommitmentPhaseError::PlanCountMismatch {
            artifact: plan.artifact,
            expected: plan.count,
            actual: plan.oracles.len(),
        });
    }
    for &oracle in plan.oracles {
        let data = inputs
            .materialize(oracle)
            .ok_or(CommitmentPhaseError::MissingOracle { oracle })?;
        let oracle_num_vars = oracle_num_vars(oracle, plan.num_vars);
        let data = into_padded_oracle(oracle, oracle_num_vars, data)?;
        let (commitment, hint) = commit_with_layout(&data, plan.num_vars, prover_setup)?;
        artifacts.records.push(CommitmentRecord {
            artifact: plan.artifact,
            oracle,
            label: plan.label,
            num_vars: oracle_num_vars,
        });
        artifacts.commitments.push(Some(commitment));
        artifacts.hints.push(OracleOpeningHint { oracle, hint });
    }
    Ok(())
}

fn commit_optional<I>(
    inputs: &mut I,
    prover_setup: &DoryProverSetup,
    artifacts: &mut CommitmentArtifacts,
    plan: &OptionalCommitmentPlan,
) -> Result<(), CommitmentPhaseError>
where
    I: CommitmentInputProvider,
{
    let Some(data) = inputs.materialize(plan.oracle) else {
        return push_skipped_optional(artifacts, plan);
    };
    if should_skip_optional(plan.skip_policy, data.as_ref()) {
        return push_skipped_optional(artifacts, plan);
    }
    let data = into_padded_oracle(plan.oracle, plan.num_vars, data)?;
    let (commitment, hint) = commit_with_layout(&data, plan.num_vars, prover_setup)?;
    artifacts.records.push(CommitmentRecord {
        artifact: plan.artifact,
        oracle: plan.oracle,
        label: plan.label,
        num_vars: oracle_num_vars(plan.oracle, plan.num_vars),
    });
    artifacts.commitments.push(Some(commitment));
    artifacts.hints.push(OracleOpeningHint {
        oracle: plan.oracle,
        hint,
    });
    Ok(())
}

fn push_skipped_optional(
    artifacts: &mut CommitmentArtifacts,
    plan: &OptionalCommitmentPlan,
) -> Result<(), CommitmentPhaseError> {
    artifacts.records.push(CommitmentRecord {
        artifact: plan.artifact,
        oracle: plan.oracle,
        label: plan.label,
        num_vars: oracle_num_vars(plan.oracle, plan.num_vars),
    });
    artifacts.commitments.push(None);
    Ok(())
}

fn should_skip_optional(policy: OptionalSkipPolicy, data: &[Fr]) -> bool {
    match policy {
        OptionalSkipPolicy::MissingOrZero => data.iter().all(|value| *value == Fr::from_u64(0)),
    }
}

fn into_padded_oracle(
    oracle: &'static str,
    num_vars: usize,
    data: Cow<'_, [Fr]>,
) -> Result<Vec<Fr>, CommitmentPhaseError> {
    let target_len = target_len(num_vars)?;
    if data.len() > target_len {
        return Err(CommitmentPhaseError::OracleTooLarge {
            oracle,
            len: data.len(),
            target_len,
        });
    }
    let mut data = data.into_owned();
    data.resize(target_len, Fr::from_u64(0));
    Ok(data)
}

fn oracle_num_vars(oracle: &'static str, fallback: usize) -> usize {
    ORACLE_PLANS
        .iter()
        .find(|plan| plan.oracle == oracle)
        .map_or(fallback, |plan| plan.num_vars)
}

fn commit_with_layout(
    data: &[Fr],
    layout_num_vars: usize,
    prover_setup: &DoryProverSetup,
) -> Result<(DoryCommitment, DoryHint), CommitmentPhaseError> {
    let row_len = target_len(layout_num_vars.div_ceil(2))?;
    let mut partial = DoryScheme::begin(prover_setup);
    for row in data.chunks(row_len) {
        DoryScheme::feed(&mut partial, row, prover_setup);
    }
    let hint = DoryHint(partial.row_commitments.clone());
    let commitment = DoryScheme::finish(partial, prover_setup);
    Ok((commitment, hint))
}

fn target_len(num_vars: usize) -> Result<usize, CommitmentPhaseError> {
    if num_vars >= usize::BITS as usize {
        return Err(CommitmentPhaseError::TargetSizeOverflow { num_vars });
    }
    Ok(1usize << num_vars)
}

fn absorb_transcript<T>(
    artifacts: &CommitmentArtifacts,
    transcript: &mut T,
) -> Result<(), CommitmentPhaseError>
where
    T: Transcript<Challenge = Fr>,
{
    for step in TRANSCRIPT_PLAN {
        let mut appended = false;
        for (record, commitment) in artifacts.records.iter().zip(&artifacts.commitments) {
            if record.artifact != step.source {
                continue;
            }
            if let Some(commitment) = commitment {
                transcript.append(&LabelWithCount(step.label.as_bytes(), commitment.serialized_len()));
                commitment.append_to_transcript(transcript);
                appended = true;
            }
        }
        if !step.optional && !appended {
            return Err(CommitmentPhaseError::MissingTranscriptSource {
                source: step.source,
            });
        }
    }
    Ok(())
}
"
    }

    fn emit_verifier_entrypoint() -> &'static str {
        r"pub fn verify_commitment_phase<T>(
    proof_commitments: &[Option<DoryCommitment>],
    transcript: &mut T,
) -> Result<CommitmentArtifacts, CommitmentPhaseError>
where
    T: Transcript<Challenge = Fr>,
{
    let mut artifacts = CommitmentArtifacts::default();
    let mut cursor = 0usize;
    for plan in COMMITMENT_BATCH_PLANS {
        receive_batch(proof_commitments, &mut cursor, &mut artifacts, plan)?;
    }
    for plan in OPTIONAL_COMMITMENT_PLANS {
        receive_optional(proof_commitments, &mut cursor, &mut artifacts, plan)?;
    }
    if cursor != proof_commitments.len() {
        return Err(CommitmentPhaseError::ProofCommitmentCountMismatch {
            expected: cursor,
            actual: proof_commitments.len(),
        });
    }
    absorb_transcript(&artifacts, transcript)?;
    Ok(artifacts)
}

fn receive_batch(
    proof_commitments: &[Option<DoryCommitment>],
    cursor: &mut usize,
    artifacts: &mut CommitmentArtifacts,
    plan: &CommitmentBatchPlan,
) -> Result<(), CommitmentPhaseError> {
    if plan.count != plan.oracles.len() {
        return Err(CommitmentPhaseError::PlanCountMismatch {
            artifact: plan.artifact,
            expected: plan.count,
            actual: plan.oracles.len(),
        });
    }
    for &oracle in plan.oracles {
        let commitment = proof_commitments
            .get(*cursor)
            .ok_or(CommitmentPhaseError::MissingProofCommitmentSlot {
                artifact: plan.artifact,
                oracle,
            })?
            .as_ref()
            .ok_or(CommitmentPhaseError::MissingProofCommitment { oracle })?
            .clone();
        *cursor += 1;
        let oracle_num_vars = oracle_num_vars(oracle, plan.num_vars);
        artifacts.records.push(CommitmentRecord {
            artifact: plan.artifact,
            oracle,
            label: plan.label,
            num_vars: oracle_num_vars,
        });
        artifacts.commitments.push(Some(commitment));
    }
    Ok(())
}

fn receive_optional(
    proof_commitments: &[Option<DoryCommitment>],
    cursor: &mut usize,
    artifacts: &mut CommitmentArtifacts,
    plan: &OptionalCommitmentPlan,
) -> Result<(), CommitmentPhaseError> {
    let commitment = proof_commitments
        .get(*cursor)
        .ok_or(CommitmentPhaseError::MissingProofCommitmentSlot {
            artifact: plan.artifact,
            oracle: plan.oracle,
        })?
        .clone();
    *cursor += 1;
    artifacts.records.push(CommitmentRecord {
        artifact: plan.artifact,
        oracle: plan.oracle,
        label: plan.label,
        num_vars: oracle_num_vars(plan.oracle, plan.num_vars),
    });
    artifacts.commitments.push(commitment);
    Ok(())
}

fn oracle_num_vars(oracle: &'static str, fallback: usize) -> usize {
    ORACLE_PLANS
        .iter()
        .find(|plan| plan.oracle == oracle)
        .map_or(fallback, |plan| plan.num_vars)
}

fn absorb_transcript<T>(
    artifacts: &CommitmentArtifacts,
    transcript: &mut T,
) -> Result<(), CommitmentPhaseError>
where
    T: Transcript<Challenge = Fr>,
{
    for step in TRANSCRIPT_PLAN {
        let mut appended = false;
        for (record, commitment) in artifacts.records.iter().zip(&artifacts.commitments) {
            if record.artifact != step.source {
                continue;
            }
            if let Some(commitment) = commitment {
                transcript.append(&LabelWithCount(step.label.as_bytes(), commitment.serialized_len()));
                commitment.append_to_transcript(transcript);
                appended = true;
            }
        }
        if !step.optional && !appended {
            return Err(CommitmentPhaseError::MissingTranscriptSource {
                source: step.source,
            });
        }
    }
    Ok(())
}
"
    }
}

impl OptionalSkipPolicy {
    fn parse(value: &str) -> Result<Self, EmitError> {
        match value {
            "missing_or_zero" => Ok(Self::MissingOrZero),
            _ => Err(EmitError::new(format!(
                "unsupported optional commitment skip policy `{value}`"
            ))),
        }
    }

    fn rust_variant(&self) -> &'static str {
        match self {
            Self::MissingOrZero => "OptionalSkipPolicy::MissingOrZero",
        }
    }
}

fn string_attr(operation: OperationRef<'_, '_>, attr: &str) -> Result<String, EmitError> {
    operation
        .attribute(attr)
        .ok()
        .and_then(string_attribute_value)
        .ok_or_else(|| attr_error(operation, attr, "string"))
}

fn skip_policy_attr(
    operation: OperationRef<'_, '_>,
    attr: &str,
) -> Result<OptionalSkipPolicy, EmitError> {
    OptionalSkipPolicy::parse(&string_attr(operation, attr)?)
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

fn bool_attr(operation: OperationRef<'_, '_>, attr: &str) -> Result<bool, EmitError> {
    operation
        .attribute(attr)
        .map(|attribute| match attribute.to_string().as_str() {
            "true" => Some(true),
            "false" => Some(false),
            _ => None,
        })
        .ok()
        .flatten()
        .ok_or_else(|| attr_error(operation, attr, "bool"))
}

fn transcript_artifact_source(operation: OperationRef<'_, '_>) -> Result<String, EmitError> {
    let artifact = operation
        .operand(1)
        .map_err(|_| attr_error(operation, "artifact operand", "value"))?;
    let owner = OperationResult::try_from(artifact)
        .map_err(|_| EmitError::new("cpu.transcript_absorb artifact operand must be op result"))?
        .owner();
    symbol_attr(owner, "artifact")
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

fn rust_str(value: &str) -> String {
    format!("{value:?}")
}

fn rust_field_name(value: &str) -> String {
    let mut output = String::new();
    let mut previous_was_separator = false;
    for (index, character) in value.chars().enumerate() {
        if character == '_' {
            output.push('_');
            previous_was_separator = true;
            continue;
        }
        if character.is_ascii_uppercase() {
            if index != 0 && !previous_was_separator {
                output.push('_');
            }
            output.push(character.to_ascii_lowercase());
        } else {
            output.push(character);
        }
        previous_was_separator = false;
    }
    output
}

fn rust_input_field(source: &str) -> &'static str {
    match source {
        "trace.rd_inc" => "rd_inc",
        "trace.ram_inc" => "ram_inc",
        "trace.instruction_keys" => "instruction_keys",
        "trace.ram_addresses" => "ram_addresses",
        "trace.bytecode_indices" => "bytecode_indices",
        "advice.untrusted" => "untrusted_advice",
        "advice.trusted" => "trusted_advice",
        _ => panic!("unsupported oracle source `{source}`"),
    }
}

fn rust_padding_value(padding: &str) -> &'static str {
    match padding {
        "zero" => "Some(0)",
        "none" => "None",
        _ => panic!("unsupported oracle padding `{padding}`"),
    }
}

fn require_supported_symbol(kind: &str, actual: &str, expected: &str) -> Result<(), EmitError> {
    if actual == expected {
        Ok(())
    } else {
        Err(EmitError::new(format!(
            "unsupported {kind} @{actual}; Rust commitment emitter currently supports @{expected}"
        )))
    }
}
