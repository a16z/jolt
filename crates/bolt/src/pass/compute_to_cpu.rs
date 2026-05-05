mod commitment;
mod declarations;
mod oracles;
mod proofs;
mod transcript;
mod values;

use std::collections::BTreeMap;

use melior::ir::block::BlockLike;
use melior::ir::operation::OperationLike;

use crate::ir::{BoltModule, Compute, Cpu};
use crate::mlir::{verify_module, MeliorContext, MlirError};
use crate::schema::{verify_compute_schema, verify_cpu_schema};

use super::support::require_module_role;
use declarations::lower_declaration_op;
use proofs::lower_proof_op;

pub fn lower_compute_to_cpu<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Compute>,
) -> Result<BoltModule<'c, Cpu>, MlirError> {
    verify_compute_schema(module)?;
    let role = require_module_role(module, "CPU lowering requires compute party role")?;
    let module_name = module.name();
    let cpu = context.new_module::<Cpu>(&module_name, Some(role));
    let mut value_map = BTreeMap::new();
    let mut operation = module.as_mlir_module().body().first_operation();
    while let Some(op) = operation {
        operation = op.next_in_block();
        if lower_declaration_op(context, &cpu, op)? {
            continue;
        }
        if transcript::lower_op(context, &cpu, &mut value_map, op)? {
            continue;
        }
        if oracles::lower_op(context, &cpu, &mut value_map, op)? {
            continue;
        }
        if commitment::lower_op(context, &cpu, &mut value_map, op)? {
            continue;
        }
        if values::lower_op(context, &cpu, &mut value_map, op)? {
            continue;
        }
        lower_proof_op(context, &cpu, &mut value_map, op)?;
    }
    verify_module(&cpu)?;
    verify_cpu_schema(&cpu)?;
    Ok(cpu)
}
