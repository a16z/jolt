use melior::ir::OperationRef;

use crate::emit::rust::EmitError;
use crate::protocols::jolt::cpu_attrs::{
    int_array_attr, int_attr, operand_symbol, operand_symbols, string_attr, symbol_array_attr,
    symbol_attr,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuSumcheckClaimPlan {
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

impl CpuSumcheckClaimPlan {
    pub(crate) fn from_claim(operation: OperationRef<'_, '_>) -> Result<Self, EmitError> {
        Ok(Self {
            symbol: string_attr(operation, "sym_name")?,
            stage: symbol_attr(operation, "stage")?,
            domain: symbol_attr(operation, "domain")?,
            num_rounds: int_attr(operation, "num_rounds")?,
            degree: int_attr(operation, "degree")?,
            claim: symbol_attr(operation, "claim")?,
            kernel: Some(symbol_attr(operation, "kernel")?),
            relation: None,
            claim_value: operand_symbol(operation, 0)?,
            input_openings: operand_symbols(operation, 1)?,
        })
    }

    pub(crate) fn from_verify_claim(operation: OperationRef<'_, '_>) -> Result<Self, EmitError> {
        Ok(Self {
            symbol: string_attr(operation, "sym_name")?,
            stage: symbol_attr(operation, "stage")?,
            domain: symbol_attr(operation, "domain")?,
            num_rounds: int_attr(operation, "num_rounds")?,
            degree: int_attr(operation, "degree")?,
            claim: symbol_attr(operation, "claim")?,
            kernel: None,
            relation: Some(symbol_attr(operation, "relation")?),
            claim_value: operand_symbol(operation, 0)?,
            input_openings: operand_symbols(operation, 1)?,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuSumcheckBatchPlan {
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

impl CpuSumcheckBatchPlan {
    pub(crate) fn from_cpu(operation: OperationRef<'_, '_>) -> Result<Self, EmitError> {
        Ok(Self {
            symbol: string_attr(operation, "sym_name")?,
            stage: symbol_attr(operation, "stage")?,
            proof_slot: symbol_attr(operation, "proof_slot")?,
            policy: string_attr(operation, "policy")?,
            count: int_attr(operation, "count")?,
            ordered_claims: symbol_array_attr(operation, "ordered_claims")?,
            claim_operands: operand_symbols(operation, 0)?,
            claim_label: string_attr(operation, "claim_label")?,
            round_label: string_attr(operation, "round_label")?,
            round_schedule: int_array_attr(operation, "round_schedule")?,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuSumcheckDriverPlan {
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

impl CpuSumcheckDriverPlan {
    pub(crate) fn from_driver(operation: OperationRef<'_, '_>) -> Result<Self, EmitError> {
        Ok(Self {
            symbol: string_attr(operation, "sym_name")?,
            stage: symbol_attr(operation, "stage")?,
            proof_slot: symbol_attr(operation, "proof_slot")?,
            kernel: Some(symbol_attr(operation, "kernel")?),
            relation: None,
            batch: operand_symbol(operation, 1)?,
            policy: string_attr(operation, "policy")?,
            round_schedule: int_array_attr(operation, "round_schedule")?,
            claim_label: string_attr(operation, "claim_label")?,
            round_label: string_attr(operation, "round_label")?,
            num_rounds: int_attr(operation, "num_rounds")?,
            degree: int_attr(operation, "degree")?,
        })
    }

    pub(crate) fn from_verify(operation: OperationRef<'_, '_>) -> Result<Self, EmitError> {
        Ok(Self {
            symbol: string_attr(operation, "sym_name")?,
            stage: symbol_attr(operation, "stage")?,
            proof_slot: symbol_attr(operation, "proof_slot")?,
            kernel: None,
            relation: Some(symbol_attr(operation, "relation")?),
            batch: operand_symbol(operation, 1)?,
            policy: string_attr(operation, "policy")?,
            round_schedule: int_array_attr(operation, "round_schedule")?,
            claim_label: string_attr(operation, "claim_label")?,
            round_label: string_attr(operation, "round_label")?,
            num_rounds: int_attr(operation, "num_rounds")?,
            degree: int_attr(operation, "degree")?,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuSumcheckInstanceResultPlan {
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

impl CpuSumcheckInstanceResultPlan {
    pub(crate) fn from_cpu(operation: OperationRef<'_, '_>) -> Result<Self, EmitError> {
        Ok(Self {
            symbol: string_attr(operation, "sym_name")?,
            source: symbol_attr(operation, "source")?,
            claim: symbol_attr(operation, "claim")?,
            relation: symbol_attr(operation, "relation")?,
            index: int_attr(operation, "index")?,
            point_arity: int_attr(operation, "point_arity")?,
            num_rounds: int_attr(operation, "num_rounds")?,
            round_offset: int_attr(operation, "round_offset")?,
            point_order: string_attr(operation, "point_order")?,
            degree: int_attr(operation, "degree")?,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuSumcheckEvalPlan {
    pub symbol: String,
    pub source: String,
    pub name: String,
    pub index: usize,
    pub oracle: String,
}

impl CpuSumcheckEvalPlan {
    pub(crate) fn from_cpu(operation: OperationRef<'_, '_>) -> Result<Self, EmitError> {
        Ok(Self {
            symbol: string_attr(operation, "sym_name")?,
            source: symbol_attr(operation, "source")?,
            name: symbol_attr(operation, "name")?,
            index: int_attr(operation, "index")?,
            oracle: symbol_attr(operation, "oracle")?,
        })
    }
}
