use jolt_field::{Field, Fr};

use super::HammingWeightClaimReductionState;
use crate::cuda::{DeviceFrVec, HammingRoundPolyInputs};

pub(crate) fn hamming_round_poly<F: Field>(
    state: &HammingWeightClaimReductionState<F>,
) -> Option<[F; 2]> {
    let ctx = crate::cuda::shared_ctx()?;
    let scale = crate::cuda::into_fr(state.active_scale)?;
    let gamma_powers: Vec<Fr> = state
        .gamma_powers
        .iter()
        .map(|value| crate::cuda::into_fr(*value))
        .collect::<Option<Vec<Fr>>>()?;

    let g_devs: Vec<DeviceFrVec> = state
        .g
        .iter()
        .map(|poly| ctx.upload(crate::cuda::as_fr_slice(poly)?).ok())
        .collect::<Option<Vec<DeviceFrVec>>>()?;
    let eq_virt_devs: Vec<DeviceFrVec> = state
        .eq_virt
        .iter()
        .map(|poly| ctx.upload(crate::cuda::as_fr_slice(poly)?).ok())
        .collect::<Option<Vec<DeviceFrVec>>>()?;
    let eq_bool_dev = ctx.upload(crate::cuda::as_fr_slice(&state.eq_bool)?).ok()?;

    let g_refs: Vec<&DeviceFrVec> = g_devs.iter().collect();
    let eq_virt_refs: Vec<&DeviceFrVec> = eq_virt_devs.iter().collect();

    let evals = ctx
        .hamming_round_poly(HammingRoundPolyInputs {
            g: &g_refs,
            eq_virt: &eq_virt_refs,
            eq_bool: &eq_bool_dev,
            gamma_powers: &gamma_powers,
            scale,
        })
        .ok()?;

    Some([
        crate::cuda::fr_into::<F>(evals[0])?,
        crate::cuda::fr_into::<F>(evals[1])?,
    ])
}
