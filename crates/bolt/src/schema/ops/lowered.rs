mod attrs;
mod declarations;
mod dialect;
mod opening;
mod oracles;
mod pcs;
mod sumcheck;
mod transcript;
mod values;

use melior::ir::OperationRef;

use super::support::MaybeValidation;
use dialect::LoweredDialect;
pub(super) use dialect::{is_verifier_forbidden, ComputeDialect, CpuDialect};

pub(super) fn validate_op<D: LoweredDialect>(
    operation: OperationRef<'_, '_>,
    name: &str,
) -> MaybeValidation {
    let suffix = name.strip_prefix(D::PREFIX)?.strip_prefix('.')?;
    if let Some(result) = values::validate_op(operation, suffix) {
        return Some(result);
    }
    if let Some(result) = transcript::validate_op(operation, suffix) {
        return Some(result);
    }
    if let Some(result) = oracles::validate_op(operation, suffix) {
        return Some(result);
    }
    if let Some(result) = pcs::validate_op(operation, suffix) {
        return Some(result);
    }
    if let Some(result) = opening::validate_op(operation, suffix) {
        return Some(result);
    }
    if let Some(result) = sumcheck::validate_op::<D>(operation, suffix) {
        return Some(result);
    }
    declarations::validate_op::<D>(operation, suffix)
}
