mod boundary;
mod derive;
mod party;

use crate::ir::{BoltModule, Concrete, Party, Role};
use crate::mlir::{MeliorContext, MlirError};

use derive::derive_role;
pub use party::project_party;

pub fn derive_prover_role<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Concrete>,
) -> Result<BoltModule<'c, Concrete>, MlirError> {
    derive_role(context, module, Role::Prover)
}

pub fn derive_verifier_role<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Concrete>,
) -> Result<BoltModule<'c, Concrete>, MlirError> {
    derive_role(context, module, Role::Verifier)
}

pub fn project_prover_party<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Concrete>,
) -> Result<BoltModule<'c, Party>, MlirError> {
    project_party(context, module, Role::Prover)
}

pub fn project_verifier_party<'c>(
    context: &'c MeliorContext,
    module: &BoltModule<'c, Concrete>,
) -> Result<BoltModule<'c, Party>, MlirError> {
    project_party(context, module, Role::Verifier)
}
