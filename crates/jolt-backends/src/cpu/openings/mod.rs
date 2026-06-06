//! CPU opening compute modules.
//!
//! Dory joint-opening construction and hint reuse should live here, behind
//! hardware-agnostic opening requests.

use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_witness::{OracleViewRequest, WitnessNamespace, WitnessProvider};
use rayon::prelude::*;

use crate::{
    Backend, BackendError, OpeningBackend, OpeningRlcMaterializationRequest,
    OpeningRlcMaterializationResult,
};

use super::CpuBackend;

const RLC_MATERIALIZATION_TASK: &str = "opening RLC materialization";

impl<F, N, PCS> OpeningBackend<F, N, PCS> for CpuBackend
where
    F: Field,
    N: WitnessNamespace,
    PCS: CommitmentScheme<Field = F>,
{
    fn materialize_opening_rlc<W>(
        &mut self,
        request: &OpeningRlcMaterializationRequest<F, N>,
        witness: &W,
    ) -> Result<OpeningRlcMaterializationResult<F>, BackendError>
    where
        W: WitnessProvider<F, N>,
    {
        materialize_opening_rlc(self.name(), request, witness)
    }
}

pub(crate) fn materialize_opening_rlc<F, N, W>(
    backend: &'static str,
    request: &OpeningRlcMaterializationRequest<F, N>,
    witness: &W,
) -> Result<OpeningRlcMaterializationResult<F>, BackendError>
where
    F: Field,
    N: WitnessNamespace,
    W: WitnessProvider<F, N>,
{
    if request.components.is_empty() {
        return Err(BackendError::InvalidRequest {
            backend,
            task: RLC_MATERIALIZATION_TASK,
            reason: format!("{} has no components", request.label),
        });
    }

    let mut component_views = Vec::with_capacity(request.components.len());
    let mut expected_rows = None;
    for (component_index, component) in request.components.iter().enumerate() {
        let context = format!("{} component {component_index}", request.label);
        let view = witness
            .oracle_view(OracleViewRequest::new(component.view))
            .map_err(|error| BackendError::InvalidRequest {
                backend,
                task: RLC_MATERIALIZATION_TASK,
                reason: format!(
                    "{context} materialize {:?} failed: {error}",
                    component.view.oracle.kind
                ),
            })?;
        let descriptor = view.descriptor();
        if descriptor.reference.kind != component.view.oracle.kind {
            return Err(BackendError::InvalidRequest {
                backend,
                task: RLC_MATERIALIZATION_TASK,
                reason: format!("{context} materialized the wrong oracle"),
            });
        }
        if descriptor.encoding != component.view.encoding {
            return Err(BackendError::InvalidRequest {
                backend,
                task: RLC_MATERIALIZATION_TASK,
                reason: format!(
                    "{context} materialized encoding {:?}, expected {:?}",
                    descriptor.encoding, component.view.encoding
                ),
            });
        }
        let Some(values) = view.as_slice() else {
            return Err(BackendError::InvalidRequest {
                backend,
                task: RLC_MATERIALIZATION_TASK,
                reason: format!("{context} did not materialize a concrete view"),
            });
        };
        if let Some(expected_rows) = expected_rows {
            if values.len() != expected_rows {
                return Err(BackendError::InvalidRequest {
                    backend,
                    task: RLC_MATERIALIZATION_TASK,
                    reason: format!(
                        "{context} materialized {} rows, expected {expected_rows}",
                        values.len()
                    ),
                });
            }
        } else {
            expected_rows = Some(values.len());
        }

        component_views.push((component.scalar, view));
    }

    let mut unit_slices = Vec::new();
    let mut scaled_slices = Vec::with_capacity(component_views.len());
    for (component_index, (scalar, view)) in component_views.iter().enumerate() {
        let Some(values) = view.as_slice() else {
            return Err(BackendError::InvalidRequest {
                backend,
                task: RLC_MATERIALIZATION_TASK,
                reason: format!(
                    "{} component {component_index} did not retain a concrete view",
                    request.label
                ),
            });
        };
        if scalar.is_zero() {
            continue;
        }
        if scalar.is_one() {
            unit_slices.push(values);
        } else {
            scaled_slices.push((*scalar, values));
        }
    }

    let mut result = vec![F::zero(); expected_rows.unwrap_or(0)];
    result.par_iter_mut().enumerate().for_each(|(row, acc)| {
        let mut value = F::zero();
        for values in &unit_slices {
            value += values[row];
        }
        for (scalar, values) in &scaled_slices {
            value += *scalar * values[row];
        }
        *acc = value;
    });

    Ok(OpeningRlcMaterializationResult::new(result))
}
