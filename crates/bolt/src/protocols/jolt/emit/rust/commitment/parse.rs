use melior::ir::block::BlockLike;
use melior::ir::operation::{OperationLike, OperationResult};
use melior::ir::OperationRef;

use super::{
    CommitmentBatchPlan, CommitmentCpuProgram, CommitmentParams, OptionalCommitmentPlan,
    OptionalSkipPolicy, OracleGeneration, OraclePlan, TranscriptStep,
};
use crate::emit::rust::EmitError;
use crate::ir::{BoltModule, Cpu};
use crate::protocols::jolt::emit::rust::mlir::{
    attr_error, bool_attr, int_attr, operation_name, string_attr, symbol_array_attr, symbol_attr,
};

impl CommitmentCpuProgram {
    pub(super) fn from_module(module: &BoltModule<'_, Cpu>) -> Result<Self, EmitError> {
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
}

fn skip_policy_attr(
    operation: OperationRef<'_, '_>,
    attr: &str,
) -> Result<OptionalSkipPolicy, EmitError> {
    OptionalSkipPolicy::parse(&string_attr(operation, attr)?)
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
