use jolt_field::Ring;

use crate::{derived, opening};

use super::super::{
    BooleanityPublic, JoltExpr, JoltOpeningId, JoltRelationId, JoltVirtualPolynomial,
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
    F: Ring,
{
    booleanity_output(booleanity_output_openings(dimensions.layout))
}

/// The booleanity fold `eq · Σ_i γ^{2i} (x_i² − x_i)` over any boolean-checked
/// opening list; shared by the base and lattice-mode variants of the relation
/// so the formula has one owner.
pub(crate) fn booleanity_output<F>(openings: impl IntoIterator<Item = JoltOpeningId>) -> JoltExpr<F>
where
    F: Ring,
{
    let eq_address_cycle = derived(BooleanityPublic::EqAddressCycle);
    let mut output = JoltExpr::zero();

    for (i, opening_id) in openings.into_iter().enumerate() {
        let x = opening(opening_id);
        let weight = if i == 0 {
            JoltExpr::one()
        } else {
            derived(BooleanityPublic::GammaPow { exponent: 2 * i })
        };
        output = output + weight * (x.clone() * x.clone() - x);
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
