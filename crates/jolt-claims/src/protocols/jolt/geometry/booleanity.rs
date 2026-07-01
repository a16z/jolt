use jolt_field::RingCore;

use crate::{challenge, derived, opening};

use super::super::{
    BooleanityChallenge, BooleanityPublic, JoltExpr, JoltOpeningId, JoltRelationId,
    JoltVirtualPolynomial,
};
use super::ra::JoltRaPolynomialLayout;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BooleanityDimensions {
    pub layout: JoltRaPolynomialLayout,
    pub log_t: usize,
    pub log_k_chunk: usize,
}

impl BooleanityDimensions {
    pub const fn new(layout: JoltRaPolynomialLayout, log_t: usize, log_k_chunk: usize) -> Self {
        Self {
            layout,
            log_t,
            log_k_chunk,
        }
    }

    pub const fn sumcheck_rounds(self) -> usize {
        self.log_t + self.log_k_chunk
    }
}

pub(crate) fn booleanity_cycle_output<F>(dimensions: BooleanityDimensions) -> JoltExpr<F>
where
    F: RingCore,
{
    let gamma = challenge(BooleanityChallenge::Gamma);
    let eq_address_cycle = derived(BooleanityPublic::EqAddressCycle);
    let mut output = JoltExpr::zero();

    for (i, opening_id) in booleanity_output_openings(dimensions.layout)
        .into_iter()
        .enumerate()
    {
        let ra = opening(opening_id);
        output = output + gamma.clone().pow(2 * i) * (ra.clone() * ra.clone() - ra);
    }

    eq_address_cycle * output
}

pub fn booleanity_output_openings(layout: JoltRaPolynomialLayout) -> Vec<JoltOpeningId> {
    layout.openings(JoltRelationId::Booleanity).collect()
}

pub fn booleanity_address_phase_opening() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::BooleanityAddrClaim,
        JoltRelationId::Booleanity,
    )
}
