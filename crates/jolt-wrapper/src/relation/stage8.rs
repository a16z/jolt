//! Stage 8 (clear mode): the committed-opening RLC and the Dory `Fr` scalar
//! algebra. Every RA polynomial opens at the stage-7 point; the increment
//! polynomials open at its cycle suffix, embedded by `Π (1 − r_address)`.

use jolt_claims::protocols::jolt::geometry::committed_openings::{
    final_opening_id, final_opening_polynomial_order,
};
use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, TracePolynomialOrder};
use jolt_field::{Fr, One};

use super::ctx::{Accum, Ctx};
use super::dory::{self, DoryLinks};
use super::gadgets::{powers, zero_selector};
use super::replay::SqueezeKind;
use super::stage5::Stage5;
use super::stage7::Stage7;
use super::wiring::Wires;
use super::RelationError;
use crate::profile::WrapperProfile;

const RLC_CLAIMS_LABEL: &[u8] = b"rlc_claims";
const OPENING_POINT_LABEL: &[u8] = b"opening_point";
const OPENING_EVAL_LABEL: &[u8] = b"opening_eval";

pub(crate) fn walk(
    ctx: &mut Ctx,
    profile: &WrapperProfile,
    wires: &Wires,
    stage5: &Stage5,
    stage7: &Stage7,
) -> Result<DoryLinks, RelationError> {
    if profile.trace_polynomial_order != TracePolynomialOrder::CycleMajor {
        return Err(RelationError::Unsupported(
            "address-major trace polynomials",
        ));
    }
    let chunk_bits = profile.one_hot_config.committed_chunk_bits();
    // Cycle-major with no precommitted anchors: the unified point is the
    // stage-7 point itself, whose cycle suffix is the increment point.
    let opening_point = &stage7.point;

    ctx.section("stage8/rlc");
    let order = final_opening_polynomial_order(stage5.formula.ra_layout, false, false, None);
    let inc_scale = zero_selector(ctx, &opening_point[..chunk_bits]);
    let mut values = Vec::with_capacity(order.len());
    for polynomial in &order {
        let claim = wires.lc(&final_opening_id(*polynomial))?;
        let value = match polynomial {
            JoltCommittedPolynomial::RamInc | JoltCommittedPolynomial::RdInc => {
                ctx.mul(&claim, &inc_scale)
            }
            _ => claim,
        };
        values.push(value);
    }
    ctx.absorb_label_count(RLC_CLAIMS_LABEL, values.len())?;
    for value in &values {
        ctx.absorb_value(value)?;
    }
    let rho = ctx.squeeze(SqueezeKind::Scalar)?;
    let rho_powers = powers(ctx, &rho, values.len());
    let mut joint = Accum::default();
    for (value, power) in values.iter().zip(&rho_powers) {
        let term = ctx.mul(power, value);
        joint.add(&term, Fr::one());
    }
    let joint_claim = joint.finish();

    ctx.section("stage8/dory");
    let links = dory::walk(ctx, opening_point, &joint_claim, &rho_powers)?;

    ctx.section("stage8/evaluation_claim");
    ctx.absorb_label_count(OPENING_POINT_LABEL, opening_point.len())?;
    for coordinate in opening_point {
        ctx.absorb_value(coordinate)?;
    }
    ctx.absorb_label(OPENING_EVAL_LABEL)?;
    ctx.absorb_value(&joint_claim)?;
    Ok(links)
}
