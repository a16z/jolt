use melior::ir::OperationRef;

use crate::emit::rust::EmitError;
use crate::protocols::jolt::cpu_attrs::{int_attr, string_attr, symbol_attr};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuProgramStepPlan {
    pub kind: String,
    pub symbol: String,
}

impl CpuProgramStepPlan {
    pub(crate) fn new(kind: &str, symbol: String) -> Self {
        Self {
            kind: kind.to_owned(),
            symbol,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuTranscriptSqueezePlan {
    pub symbol: String,
    pub label: String,
    pub kind: String,
    pub count: usize,
}

impl CpuTranscriptSqueezePlan {
    pub(crate) fn from_cpu(operation: OperationRef<'_, '_>) -> Result<Self, EmitError> {
        Ok(Self {
            symbol: string_attr(operation, "sym_name")?,
            label: string_attr(operation, "label")?,
            kind: string_attr(operation, "kind")?,
            count: int_attr(operation, "count")?,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuTranscriptAbsorbBytesPlan {
    pub symbol: String,
    pub label: String,
    pub payload: String,
}

impl CpuTranscriptAbsorbBytesPlan {
    pub(crate) fn from_cpu(operation: OperationRef<'_, '_>) -> Result<Self, EmitError> {
        Ok(Self {
            symbol: string_attr(operation, "sym_name")?,
            label: string_attr(operation, "label")?,
            payload: string_attr(operation, "payload")?,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuOpeningInputPlan {
    pub symbol: String,
    pub source_stage: String,
    pub source_claim: String,
    pub oracle: String,
    pub domain: String,
    pub point_arity: usize,
    pub claim_kind: String,
}

impl CpuOpeningInputPlan {
    pub(crate) fn from_cpu(operation: OperationRef<'_, '_>) -> Result<Self, EmitError> {
        Ok(Self {
            symbol: string_attr(operation, "sym_name")?,
            source_stage: symbol_attr(operation, "source_stage")?,
            source_claim: symbol_attr(operation, "source_claim")?,
            oracle: symbol_attr(operation, "oracle")?,
            domain: symbol_attr(operation, "domain")?,
            point_arity: int_attr(operation, "point_arity")?,
            claim_kind: string_attr(operation, "claim_kind")?,
        })
    }
}
