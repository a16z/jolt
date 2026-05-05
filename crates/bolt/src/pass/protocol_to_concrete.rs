use crate::ir::{BoltModule, Concrete, Phase, Protocol};
use crate::mlir::{verify_module, MeliorContext, MlirError};
use crate::schema::{verify_concrete_schema, verify_protocol_schema};

use super::module_copy::{phase_copy_source, PhaseCopyRole};

pub fn lower_piop_and_fiat_shamir<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Protocol>,
) -> Result<BoltModule<'c, Concrete>, MlirError> {
    verify_protocol_schema(module)?;
    let source = phase_copy_source(module, Concrete::NAME, PhaseCopyRole::absent(), &[]);
    let concrete = context.parse_module::<Concrete>(&source)?;
    verify_module(&concrete)?;
    verify_concrete_schema(&concrete)?;
    Ok(concrete)
}
