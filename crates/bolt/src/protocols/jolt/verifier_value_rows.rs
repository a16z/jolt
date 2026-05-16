use melior::ir::OperationRef;

use crate::emit::rust::EmitError;
use crate::protocols::jolt::cpu_attrs::{
    int_attr, operand_symbols, operation_name, signed_int_attr, string_attr, symbol_attr,
};
use crate::protocols::jolt::verifier_relation_outputs::FieldExprDependencies;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuFieldConstantPlan {
    pub symbol: String,
    pub field: String,
    pub value: usize,
}

impl CpuFieldConstantPlan {
    pub(crate) fn from_const(operation: OperationRef<'_, '_>) -> Result<Self, EmitError> {
        Self::from_value(operation, int_attr(operation, "value")?)
    }

    pub(crate) fn from_zero(operation: OperationRef<'_, '_>) -> Result<Self, EmitError> {
        Self::from_value(operation, 0)
    }

    pub(crate) fn from_one(operation: OperationRef<'_, '_>) -> Result<Self, EmitError> {
        Self::from_value(operation, 1)
    }

    fn from_value(operation: OperationRef<'_, '_>, value: usize) -> Result<Self, EmitError> {
        Ok(Self {
            symbol: string_attr(operation, "sym_name")?,
            field: symbol_attr(operation, "field")?,
            value,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuFieldExprPlan {
    pub symbol: String,
    pub kind: String,
    pub formula: String,
    pub operand_names: Vec<String>,
    pub operands: Vec<String>,
}

impl CpuFieldExprPlan {
    pub(crate) fn from_field_op(operation: OperationRef<'_, '_>) -> Result<Self, EmitError> {
        Ok(Self::op(
            string_attr(operation, "sym_name")?,
            operation_name(operation).replace("cpu.field_", "field."),
            operand_symbols(operation, 0)?,
        ))
    }

    pub(crate) fn from_field_pow(operation: OperationRef<'_, '_>) -> Result<Self, EmitError> {
        Ok(Self::op(
            string_attr(operation, "sym_name")?,
            format!("field.pow:{}", int_attr(operation, "exponent")?),
            operand_symbols(operation, 0)?,
        ))
    }

    pub(crate) fn from_lagrange_basis_eval(
        operation: OperationRef<'_, '_>,
    ) -> Result<Self, EmitError> {
        let domain_start = signed_int_attr(operation, "domain_start")?;
        let domain_size = int_attr(operation, "domain_size")?;
        let index = int_attr(operation, "index")?;
        Ok(Self::op(
            string_attr(operation, "sym_name")?,
            format!("poly.lagrange_basis_eval:{domain_start}:{domain_size}:{index}"),
            operand_symbols(operation, 0)?,
        ))
    }

    pub(crate) fn op(symbol: String, formula: String, operands: Vec<String>) -> Self {
        Self {
            symbol,
            kind: "op".to_owned(),
            formula,
            operand_names: operands.clone(),
            operands,
        }
    }
}

impl FieldExprDependencies for CpuFieldExprPlan {
    fn symbol(&self) -> &str {
        &self.symbol
    }

    fn operands(&self) -> &[String] {
        &self.operands
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuScalarExprPlan {
    pub symbol: String,
    pub kind: String,
    pub formula: String,
    pub operand_names: Vec<String>,
    pub operands: Vec<String>,
}

impl CpuScalarExprPlan {
    pub(crate) fn op(symbol: String, formula: String, operands: Vec<String>) -> Self {
        Self {
            symbol,
            kind: "op".to_owned(),
            formula,
            operand_names: operands.clone(),
            operands,
        }
    }
}
