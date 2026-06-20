pub mod stage0;

pub mod stage1;
pub mod stage2;
pub mod stage3;
pub mod stage4;
pub mod stage5;
pub mod stage6;
pub mod stage7;
pub mod stage8;

pub(crate) mod advice;
mod recorder;

#[cfg(test)]
use jolt_witness::{OracleRef, ViewRequirement, WitnessNamespace, WitnessProvider};

use crate::ProverError;

#[cfg(feature = "zk")]
pub mod zk {}

#[cfg(test)]
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

pub(crate) fn invalid_sumcheck_output(error: impl std::fmt::Display) -> ProverError {
    ProverError::InvalidSumcheckOutput {
        reason: error.to_string(),
    }
}
