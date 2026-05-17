use melior::ir::OperationRef;

use crate::emit::rust::EmitError;
use crate::protocols::jolt::cpu_attrs::{
    int_attr, operand_symbol, operand_symbols, string_attr, symbol_array_attr, symbol_attr,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuOpeningClaimPlan {
    pub symbol: String,
    pub oracle: String,
    pub domain: String,
    pub point_arity: usize,
    pub claim_kind: String,
    pub point_source: String,
    pub eval_source: String,
}

impl CpuOpeningClaimPlan {
    pub(crate) fn from_cpu(operation: OperationRef<'_, '_>) -> Result<Self, EmitError> {
        Ok(Self {
            symbol: string_attr(operation, "sym_name")?,
            oracle: symbol_attr(operation, "oracle")?,
            domain: symbol_attr(operation, "domain")?,
            point_arity: int_attr(operation, "point_arity")?,
            claim_kind: string_attr(operation, "claim_kind")?,
            point_source: operand_symbol(operation, 0)?,
            eval_source: operand_symbol(operation, 1)?,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuOpeningClaimEqualityPlan {
    pub symbol: String,
    pub mode: String,
    pub lhs: String,
    pub rhs: String,
}

impl CpuOpeningClaimEqualityPlan {
    pub(crate) fn from_cpu(operation: OperationRef<'_, '_>) -> Result<Self, EmitError> {
        Ok(Self {
            symbol: string_attr(operation, "sym_name")?,
            mode: string_attr(operation, "mode")?,
            lhs: operand_symbol(operation, 0)?,
            rhs: operand_symbol(operation, 1)?,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuOpeningBatchPlan {
    pub symbol: String,
    pub stage: String,
    pub proof_slot: String,
    pub policy: String,
    pub count: usize,
    pub ordered_claims: Vec<String>,
    pub claim_operands: Vec<String>,
}

impl CpuOpeningBatchPlan {
    pub(crate) fn from_cpu(operation: OperationRef<'_, '_>) -> Result<Self, EmitError> {
        Ok(Self {
            symbol: string_attr(operation, "sym_name")?,
            stage: symbol_attr(operation, "stage")?,
            proof_slot: symbol_attr(operation, "proof_slot")?,
            policy: string_attr(operation, "policy")?,
            count: int_attr(operation, "count")?,
            ordered_claims: symbol_array_attr(operation, "ordered_claims")?,
            claim_operands: operand_symbols(operation, 0)?,
        })
    }
}
