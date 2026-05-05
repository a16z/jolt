mod declarations;
mod proofs;
mod registry;
mod transcript;
mod values;

use std::collections::BTreeMap;

use melior::ir::block::BlockLike;
use melior::ir::operation::OperationLike;

use crate::ir::{BoltModule, Compute};
use crate::mlir::{verify_module, MeliorContext, MlirError};
use crate::schema::verify_compute_schema;

use super::support::require_module_role;
use declarations::copy_declaration_op;
use proofs::lower_proof_op;
pub use registry::{ComputeKernelSpec, KernelRegistry};

pub fn resolve_compute_kernels_with<'c, R>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Compute>,
    mut kernel_registry: R,
) -> Result<BoltModule<'c, Compute>, MlirError>
where
    R: KernelRegistry,
{
    verify_compute_schema(module)?;
    let role = require_module_role(module, "kernel resolution requires compute party role")?;
    let kernelized = context.new_module::<Compute>(&module.name(), Some(role));
    let mut value_map = BTreeMap::new();
    let mut kernels = BTreeMap::new();
    let mut operation = module.as_mlir_module().body().first_operation();
    while let Some(op) = operation {
        operation = op.next_in_block();
        if copy_declaration_op(context, &kernelized, op)? {
            continue;
        }
        if transcript::lower_op(context, &kernelized, &mut value_map, op)? {
            continue;
        }
        if values::lower_op(context, &kernelized, &mut value_map, op)? {
            continue;
        }
        lower_proof_op(
            context,
            &kernelized,
            &mut value_map,
            &mut kernels,
            &mut kernel_registry,
            op,
        )?;
    }

    verify_module(&kernelized)?;
    verify_compute_schema(&kernelized)?;
    Ok(kernelized)
}
