use melior::ir::OperationRef;

use crate::emit::rust::EmitError;
use crate::protocols::jolt::cpu_attrs::{
    int_attr, operand_symbol, operand_symbols, string_attr, symbol_attr,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuPointZeroPlan {
    pub symbol: String,
    pub field: String,
    pub arity: usize,
}

impl CpuPointZeroPlan {
    pub(crate) fn from_cpu(operation: OperationRef<'_, '_>) -> Result<Self, EmitError> {
        Ok(Self {
            symbol: string_attr(operation, "sym_name")?,
            field: symbol_attr(operation, "field")?,
            arity: int_attr(operation, "arity")?,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuPointSlicePlan {
    pub symbol: String,
    pub source: String,
    pub offset: usize,
    pub length: usize,
    pub input: String,
}

impl CpuPointSlicePlan {
    pub(crate) fn from_cpu(operation: OperationRef<'_, '_>) -> Result<Self, EmitError> {
        Ok(Self {
            symbol: string_attr(operation, "sym_name")?,
            source: symbol_attr(operation, "source")?,
            offset: int_attr(operation, "offset")?,
            length: int_attr(operation, "length")?,
            input: operand_symbol(operation, 0)?,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuPointConcatPlan {
    pub symbol: String,
    pub layout: String,
    pub arity: usize,
    pub inputs: Vec<String>,
}

impl CpuPointConcatPlan {
    pub(crate) fn from_cpu(operation: OperationRef<'_, '_>) -> Result<Self, EmitError> {
        Ok(Self {
            symbol: string_attr(operation, "sym_name")?,
            layout: string_attr(operation, "layout")?,
            arity: int_attr(operation, "arity")?,
            inputs: operand_symbols(operation, 0)?,
        })
    }
}
