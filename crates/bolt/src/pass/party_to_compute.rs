mod declarations;
mod proofs;
mod transcript;
mod values;

use std::collections::BTreeMap;

use melior::ir::block::BlockLike;
use melior::ir::operation::OperationLike;

use crate::ir::{BoltModule, Compute, Party};
use crate::mlir::{verify_module, MeliorContext, MlirError};
use crate::schema::{verify_compute_schema, verify_party_schema};

use super::support::require_module_role;
use declarations::{append_module_declarations, copy_relation_op};
use proofs::lower_proof_op;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PartyToComputeLowering<'a> {
    pub params_symbol: &'a str,
    pub function_symbol: &'a str,
    pub source_symbol: &'a str,
    pub diagnostic_label: &'a str,
}

pub fn lower_party_to_compute<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Party>,
    options: PartyToComputeLowering<'_>,
) -> Result<BoltModule<'c, Compute>, MlirError> {
    verify_party_schema(module)?;
    let role = require_module_role(
        module,
        format!("{} lowering requires party role", options.diagnostic_label),
    )?;
    let compute = context.new_module::<Compute>(&module.name(), Some(role.clone()));
    append_module_declarations(context, module, &compute, options)?;

    let mut value_map = BTreeMap::new();
    let mut operation = module.as_mlir_module().body().first_operation();
    while let Some(op) = operation {
        operation = op.next_in_block();
        if copy_relation_op(context, &compute, op)? {
            continue;
        }
        if transcript::lower_op(context, &compute, &mut value_map, op)? {
            continue;
        }
        if values::lower_op(context, &compute, &mut value_map, op)? {
            continue;
        }
        lower_proof_op(context, &compute, &mut value_map, op, &role)?;
    }

    verify_module(&compute)?;
    verify_compute_schema(&compute)?;
    Ok(compute)
}
