use jolt_field::{Field, RingCore};

use crate::opening;

use super::super::{
    JoltAdviceKind, JoltCommittedPolynomial, JoltExpr, JoltOpeningId, JoltRelationId,
    JoltVirtualPolynomial, RamRaClaimReductionPublic, RamValCheckPublic,
};
use super::dimensions::ReadWriteDimensions;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RamRafEvaluationDimensions(ReadWriteDimensions);

impl TryFrom<ReadWriteDimensions> for RamRafEvaluationDimensions {
    type Error = super::dimensions::JoltFormulaDimensionsError;

    fn try_from(dimensions: ReadWriteDimensions) -> Result<Self, Self::Error> {
        if dimensions.phase1_num_rounds() > dimensions.log_t() {
            return Err(Self::Error::InvalidPhaseRounds {
                phase1_num_rounds: dimensions.phase1_num_rounds(),
                log_t: dimensions.log_t(),
            });
        }
        Ok(Self(dimensions))
    }
}

impl RamRafEvaluationDimensions {
    pub const fn read_write(self) -> ReadWriteDimensions {
        self.0
    }

    pub const fn phase3_cycle_rounds(self) -> usize {
        self.0.phase3_cycle_rounds()
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RamRaVirtualizationDimensions {
    log_t: usize,
    committed_ra_polys: usize,
}

impl RamRaVirtualizationDimensions {
    pub const fn new(log_t: usize, committed_ra_polys: usize) -> Self {
        Self {
            log_t,
            committed_ra_polys,
        }
    }

    pub const fn log_t(self) -> usize {
        self.log_t
    }

    pub const fn num_committed_ra_polys(self) -> usize {
        self.committed_ra_polys
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RamValCheckInit<F> {
    pub public_eval: F,
    pub contributions: Vec<RamValCheckInitContribution<F>>,
}

impl<F> RamValCheckInit<F> {
    pub fn full(init_eval: F) -> Self {
        Self {
            public_eval: init_eval,
            contributions: Vec::new(),
        }
    }

    pub fn decomposed<I>(public_eval: F, contributions: I) -> Self
    where
        I: IntoIterator<Item = RamValCheckInitContribution<F>>,
    {
        Self {
            public_eval,
            contributions: contributions.into_iter().collect(),
        }
    }
}

impl<F> From<F> for RamValCheckInit<F> {
    fn from(value: F) -> Self {
        Self::full(value)
    }
}

/// One staged-opening contribution to `Val_init(r_address)`: the init
/// evaluation gains `-neg_selector * opening`. Advice polynomials contribute
/// with their block-selector weight; in committed program mode the program
/// image contributes its staged scalar with weight one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RamValCheckInitContribution<F> {
    /// The `Public` selector id this contribution is weighted by in the symbolic
    /// `Val_init` decomposition (resolves to `neg_selector` on the verifier).
    pub selector: RamValCheckPublic,
    pub neg_selector: F,
    pub opening: JoltOpeningId,
}

impl<F> RamValCheckInitContribution<F> {
    pub fn new(selector: RamValCheckPublic, neg_selector: F, opening: JoltOpeningId) -> Self {
        Self {
            selector,
            neg_selector,
            opening,
        }
    }

    pub fn untrusted(neg_selector: F) -> Self {
        Self::new(
            RamValCheckPublic::InitSelector(JoltAdviceKind::Untrusted),
            neg_selector,
            JoltOpeningId::untrusted_advice(JoltRelationId::RamValCheck),
        )
    }

    pub fn trusted(neg_selector: F) -> Self {
        Self::new(
            RamValCheckPublic::InitSelector(JoltAdviceKind::Trusted),
            neg_selector,
            JoltOpeningId::trusted_advice(JoltRelationId::RamValCheck),
        )
    }

    pub fn program_image(neg_selector: F) -> Self {
        Self::new(
            RamValCheckPublic::InitSelectorProgramImage,
            neg_selector,
            super::claim_reductions::program_image::ram_val_check_contribution_opening(),
        )
    }
}

pub fn val_check_advice_opening(kind: JoltAdviceKind) -> JoltOpeningId {
    match kind {
        JoltAdviceKind::Trusted => JoltOpeningId::trusted_advice(JoltRelationId::RamValCheck),
        JoltAdviceKind::Untrusted => JoltOpeningId::untrusted_advice(JoltRelationId::RamValCheck),
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RamRaClaimReductionPublicValues<F: Field> {
    pub eq_cycle_raf: F,
    pub eq_cycle_read_write: F,
    pub eq_cycle_val_check: F,
}

impl<F: Field> RamRaClaimReductionPublicValues<F> {
    pub fn value(&self, id: RamRaClaimReductionPublic) -> F {
        match id {
            RamRaClaimReductionPublic::EqCycleRaf => self.eq_cycle_raf,
            RamRaClaimReductionPublic::EqCycleReadWrite => self.eq_cycle_read_write,
            RamRaClaimReductionPublic::EqCycleValCheck => self.eq_cycle_val_check,
        }
    }
}

pub fn ra_virtualization_committed_ram_ra_opening(index: usize) -> JoltOpeningId {
    committed_ram_ra(index)
}

pub(crate) fn committed_ram_ra_product<F>(dimensions: RamRaVirtualizationDimensions) -> JoltExpr<F>
where
    F: RingCore,
{
    let mut product = JoltExpr::one();
    for index in 0..dimensions.num_committed_ra_polys() {
        product = product * opening(committed_ram_ra(index));
    }
    product
}

pub(crate) fn ram_read_value() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RamReadValue,
        JoltRelationId::SpartanOuter,
    )
}

pub(crate) fn ram_write_value() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RamWriteValue,
        JoltRelationId::SpartanOuter,
    )
}

pub fn ram_ra() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RamRa,
        JoltRelationId::RamReadWriteChecking,
    )
}

pub fn ram_val() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RamVal,
        JoltRelationId::RamReadWriteChecking,
    )
}

pub fn ram_val_final() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RamValFinal,
        JoltRelationId::RamOutputCheck,
    )
}

pub fn ram_inc() -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::RamInc,
        JoltRelationId::RamReadWriteChecking,
    )
}

pub fn ram_inc_val_check() -> JoltOpeningId {
    JoltOpeningId::committed(JoltCommittedPolynomial::RamInc, JoltRelationId::RamValCheck)
}

pub fn ram_ra_val_check() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(JoltVirtualPolynomial::RamRa, JoltRelationId::RamValCheck)
}

pub(crate) fn ram_address_spartan() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RamAddress,
        JoltRelationId::SpartanOuter,
    )
}

pub fn ram_ra_raf_evaluation() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RamRa,
        JoltRelationId::RamRafEvaluation,
    )
}

pub fn ram_ra_claim_reduction() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RamRa,
        JoltRelationId::RamRaClaimReduction,
    )
}

pub(crate) fn committed_ram_ra(index: usize) -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::RamRa(index),
        JoltRelationId::RamRaVirtualization,
    )
}

pub fn ram_hamming_weight() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RamHammingWeight,
        JoltRelationId::RamHammingBooleanity,
    )
}

#[cfg(test)]
#[expect(clippy::panic)]
mod tests {
    use super::*;

    fn read_write_dimensions() -> ReadWriteDimensions {
        ReadWriteDimensions::new(5, 4, 2, 1)
    }

    fn raf_evaluation_dimensions() -> RamRafEvaluationDimensions {
        RamRafEvaluationDimensions::try_from(read_write_dimensions()).unwrap_or_else(|err| {
            panic!("test RAM RAF evaluation dimensions should be valid: {err}")
        })
    }

    #[test]
    fn raf_evaluation_rejects_invalid_dimensions() {
        assert!(
            RamRafEvaluationDimensions::try_from(ReadWriteDimensions::new(3, 4, 4, 0)).is_err()
        );

        let dimensions = raf_evaluation_dimensions();
        assert_eq!(dimensions.phase3_cycle_rounds(), 3);

        let large_dimensions =
            RamRafEvaluationDimensions::try_from(ReadWriteDimensions::new(usize::MAX, 0, 0, 0))
                .unwrap_or_else(|err| {
                    panic!("large RAM RAF evaluation dimensions was rejected: {err}")
                });
        assert_eq!(large_dimensions.phase3_cycle_rounds(), usize::MAX);
    }
}
