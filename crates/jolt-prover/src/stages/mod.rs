pub mod stage0;

pub mod stage1;
pub mod stage2;
pub mod stage3;
pub mod stage4;
pub mod stage5;
pub mod stage6;
pub mod stage7;
pub mod stage8;

use std::collections::BTreeMap;

use jolt_backends::{BackendValueSlot, SumcheckMaterializationOutput};
use jolt_claims::protocols::jolt::{JoltOpeningId, JoltPolynomialId};
use jolt_field::Field;
use jolt_witness::{
    protocols::jolt_vm::JoltVmNamespace, OracleRef, ViewRequirement, WitnessNamespace,
    WitnessProvider,
};

use crate::ProverError;

#[cfg(feature = "zk")]
pub mod zk {}

#[cfg(feature = "field-inline")]
pub(crate) type FieldInlineExtension<'a, FI> = &'a FI;

#[cfg(not(feature = "field-inline"))]
pub(crate) type FieldInlineExtension<'a, FI> = std::marker::PhantomData<&'a FI>;

#[cfg(not(feature = "field-inline"))]
pub(crate) const fn no_field_inline_extension<'a, FI>() -> FieldInlineExtension<'a, FI> {
    std::marker::PhantomData
}

pub(crate) fn oracle_ref_from_jolt_opening(
    opening: JoltOpeningId,
) -> Result<OracleRef<JoltVmNamespace>, ProverError> {
    match opening {
        JoltOpeningId::Polynomial { polynomial, .. } => match polynomial {
            JoltPolynomialId::Committed(polynomial) => Ok(OracleRef::committed(polynomial)),
            JoltPolynomialId::Virtual(polynomial) => Ok(OracleRef::virtual_polynomial(polynomial)),
        },
        JoltOpeningId::TrustedAdvice { .. } | JoltOpeningId::UntrustedAdvice { .. } => {
            Err(ProverError::InvalidStageRequest {
                reason: format!("expected a Jolt polynomial opening request, got {opening:?}"),
            })
        }
    }
}

pub(crate) fn view_requirement_from_jolt_opening<F, W>(
    witness: &W,
    opening: JoltOpeningId,
) -> Result<ViewRequirement<JoltVmNamespace>, ProverError>
where
    W: WitnessProvider<F, JoltVmNamespace>,
{
    let oracle = oracle_ref_from_jolt_opening(opening)?;
    primary_view_requirement::<F, W, JoltVmNamespace>(witness, oracle)
}

pub(crate) fn primary_view_requirement<F, W, N>(
    witness: &W,
    oracle: OracleRef<N>,
) -> Result<ViewRequirement<N>, ProverError>
where
    N: WitnessNamespace,
    W: WitnessProvider<F, N>,
{
    let Some(requirement) = witness.view_requirements(oracle)?.into_iter().next() else {
        return Err(ProverError::InvalidStageRequest {
            reason: format!("witness returned no view requirement for {:?}", oracle.kind),
        });
    };
    if requirement.oracle.kind != oracle.kind {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "witness returned requirement for {:?}, expected {oracle:?}",
                requirement.oracle.kind,
                oracle = oracle.kind
            ),
        });
    }
    Ok(requirement)
}

pub(crate) fn collect_backend_materializations<F: Field>(
    materializations: Vec<SumcheckMaterializationOutput<F>>,
    duplicate_label: &'static str,
) -> Result<BTreeMap<BackendValueSlot, Vec<F>>, ProverError> {
    let mut values = BTreeMap::new();
    for materialization in materializations {
        if values
            .insert(materialization.slot, materialization.values)
            .is_some()
        {
            return Err(invalid_sumcheck_output(format!(
                "duplicate {duplicate_label} slot {:?}",
                materialization.slot
            )));
        }
    }
    Ok(values)
}

pub(crate) fn take_backend_materialization<F: Field>(
    values: &mut BTreeMap<BackendValueSlot, Vec<F>>,
    slot: BackendValueSlot,
    missing_message: impl std::fmt::Display,
) -> Result<Vec<F>, ProverError> {
    values
        .remove(&slot)
        .ok_or_else(|| invalid_sumcheck_output(missing_message))
}

pub(crate) fn invalid_sumcheck_output(error: impl std::fmt::Display) -> ProverError {
    ProverError::InvalidSumcheckOutput {
        reason: error.to_string(),
    }
}
