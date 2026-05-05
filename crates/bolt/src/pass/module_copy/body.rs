use melior::ir::block::BlockLike;
use melior::ir::operation::OperationLike;

use crate::ir::{BoltModule, Phase};

pub(super) fn push_body_text<P: Phase>(module_source: &mut String, module: &BoltModule<'_, P>) {
    let mut operation = module.as_mlir_module().body().first_operation();
    while let Some(op) = operation {
        operation = op.next_in_block();
        module_source.push_str("  ");
        module_source.push_str(&op.to_string());
        module_source.push('\n');
    }
}
