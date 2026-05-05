use std::collections::BTreeSet;

use melior::ir::OperationRef;

use super::constraints::{operation_name, string_attr, symbol_attr};
use super::SchemaError;

#[derive(Default)]
pub(super) struct KernelReferenceTracker {
    symbols: BTreeSet<String>,
    refs: Vec<String>,
}

impl KernelReferenceTracker {
    pub(super) fn record(&mut self, operation: OperationRef<'_, '_>) -> Result<(), SchemaError> {
        match operation_name(operation).as_str() {
            "compute.kernel" | "cpu.kernel" => {
                let _ = self.symbols.insert(string_attr(operation, "sym_name")?);
            }
            "compute.sumcheck_kernel_claim"
            | "compute.sumcheck_kernel_driver"
            | "cpu.sumcheck_claim"
            | "cpu.sumcheck_driver" => {
                self.refs.push(symbol_attr(operation, "kernel")?);
            }
            _ => {}
        }
        Ok(())
    }

    pub(super) fn verify(self) -> Result<(), SchemaError> {
        for kernel in self.refs {
            if !self.symbols.contains(&kernel) {
                return Err(SchemaError::new(format!(
                    "kernel reference @{kernel} has no matching kernel definition"
                )));
            }
        }
        Ok(())
    }
}
