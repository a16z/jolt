use melior::ir::operation::OperationRef;

use crate::ir::Role;
use crate::mlir::MlirError;
use crate::schema::operation_name;

#[derive(Clone, Copy, Debug)]
pub(in crate::pass) enum PcsLoweringRole<'a> {
    Available(&'a Role),
    Unavailable,
}

impl<'a> PcsLoweringRole<'a> {
    pub(in crate::pass) const fn available(role: &'a Role) -> Self {
        Self::Available(role)
    }

    pub(in crate::pass) const fn unavailable() -> Self {
        Self::Unavailable
    }

    pub(in crate::pass) fn required_for(
        self,
        operation: OperationRef<'_, '_>,
    ) -> Result<&'a Role, MlirError> {
        match self {
            Self::Available(role) => Ok(role),
            Self::Unavailable => Err(MlirError::Schema {
                message: format!(
                    "PCS lowering for `{}` requires an explicit role",
                    operation_name(operation)
                ),
            }),
        }
    }
}
