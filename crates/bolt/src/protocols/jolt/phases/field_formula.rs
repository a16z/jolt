use std::collections::BTreeMap;

use melior::ir::Value;

use crate::ir::{BoltModule, Protocol};
use crate::mlir::{MeliorContext, MlirError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FieldOp {
    Add,
    Sub,
    Mul,
}

impl FieldOp {
    fn op_name(self) -> &'static str {
        match self {
            Self::Add => "field.add",
            Self::Sub => "field.sub",
            Self::Mul => "field.mul",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FieldFormulaStep {
    pub(crate) symbol: &'static str,
    op: FieldOp,
    lhs: &'static str,
    rhs: &'static str,
}

impl FieldFormulaStep {
    pub(crate) const fn add(symbol: &'static str, lhs: &'static str, rhs: &'static str) -> Self {
        Self {
            symbol,
            op: FieldOp::Add,
            lhs,
            rhs,
        }
    }

    pub(crate) const fn sub(symbol: &'static str, lhs: &'static str, rhs: &'static str) -> Self {
        Self {
            symbol,
            op: FieldOp::Sub,
            lhs,
            rhs,
        }
    }

    pub(crate) const fn mul(symbol: &'static str, lhs: &'static str, rhs: &'static str) -> Self {
        Self {
            symbol,
            op: FieldOp::Mul,
            lhs,
            rhs,
        }
    }
}

pub(crate) struct FieldFormulaBuilder<'c, 'a> {
    context: &'c MeliorContext,
    module: &'a BoltModule<'c, Protocol>,
    values: BTreeMap<&'static str, Value<'c, 'a>>,
}

impl<'c, 'a> FieldFormulaBuilder<'c, 'a> {
    pub(crate) fn new(context: &'c MeliorContext, module: &'a BoltModule<'c, Protocol>) -> Self {
        Self {
            context,
            module,
            values: BTreeMap::new(),
        }
    }

    pub(crate) fn bind(&mut self, symbol: &'static str, value: Value<'c, 'a>) {
        let _ = self.values.insert(symbol, value);
    }

    pub(crate) fn bind_all(&mut self, values: &[(&'static str, Value<'c, 'a>)]) {
        for (symbol, value) in values {
            self.bind(symbol, *value);
        }
    }

    pub(crate) fn append_all(&mut self, formulas: &[FieldFormulaStep]) -> Result<(), MlirError> {
        for formula in formulas {
            let _ = self.append(*formula)?;
        }
        Ok(())
    }

    pub(crate) fn value(&self, symbol: &'static str) -> Result<Value<'c, 'a>, MlirError> {
        self.values
            .get(symbol)
            .copied()
            .ok_or_else(|| MlirError::Schema {
                message: format!("field formula value @{symbol} is missing"),
            })
    }

    fn append(&mut self, formula: FieldFormulaStep) -> Result<Value<'c, 'a>, MlirError> {
        let lhs = self.value(formula.lhs)?;
        let rhs = self.value(formula.rhs)?;
        let op = self.context.append_typed_op(
            self.module,
            formula.op.op_name(),
            Some(formula.symbol),
            &[],
            &[lhs, rhs],
            &["!field.scalar"],
        )?;
        let value: Value<'c, 'a> = op
            .result(0)
            .map_err(|_| MlirError::Schema {
                message: format!(
                    "{} @{} did not produce a field value",
                    formula.op.op_name(),
                    formula.symbol
                ),
            })?
            .into();
        self.bind(formula.symbol, value);
        Ok(value)
    }
}
