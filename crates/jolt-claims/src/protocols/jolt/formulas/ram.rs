use jolt_field::{Field, RingCore};
use jolt_poly::{EqPolynomial, Polynomial};

use crate::{challenge, opening, public};

use super::super::{
    JoltAdviceKind, JoltChallengeId, JoltCommittedPolynomial, JoltExpr, JoltOpeningId,
    JoltPublicId, JoltRelationId, JoltVirtualPolynomial, RamHammingBooleanityPublic,
    RamOutputCheckPublic, RamRaClaimReductionChallenge, RamRaClaimReductionPublic,
    RamRaVirtualizationPublic, RamRafEvaluationPublic, RamReadWriteChallenge, RamReadWritePublic,
    RamValCheckPublic,
};
use super::dimensions::{JoltSumcheckSpec, ReadWriteDimensions, TraceDimensions};

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

    pub const fn sumcheck(self) -> JoltSumcheckSpec {
        self.0.raf_evaluation_sumcheck()
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

    pub const fn sumcheck(self) -> JoltSumcheckSpec {
        JoltSumcheckSpec::boolean(self.log_t, self.committed_ra_polys + 1)
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

pub const fn val_check_sumcheck(dimensions: TraceDimensions) -> JoltSumcheckSpec {
    dimensions.sumcheck(3)
}

pub fn read_write_checking_output_openings() -> [JoltOpeningId; 3] {
    [ram_val(), ram_ra(), ram_inc()]
}

pub fn read_write_checking_input_openings() -> [JoltOpeningId; 2] {
    [ram_read_value(), ram_write_value()]
}

pub fn raf_evaluation_output_openings() -> [JoltOpeningId; 1] {
    [ram_ra_raf_evaluation()]
}

pub fn raf_evaluation_input_openings() -> [JoltOpeningId; 1] {
    [ram_address_spartan()]
}

pub fn output_check_output_openings() -> [JoltOpeningId; 1] {
    [ram_val_final()]
}

pub fn stage2_terminal_output_openings() -> [JoltOpeningId; 2] {
    [ram_ra_raf_evaluation(), ram_val_final()]
}

pub fn val_check_input_openings() -> [JoltOpeningId; 2] {
    [ram_val(), ram_val_final()]
}

pub fn val_check_output_openings() -> [JoltOpeningId; 2] {
    [ram_ra_val_check(), ram_inc_val_check()]
}

pub fn val_check_advice_opening(kind: JoltAdviceKind) -> JoltOpeningId {
    match kind {
        JoltAdviceKind::Trusted => JoltOpeningId::trusted_advice(JoltRelationId::RamValCheck),
        JoltAdviceKind::Untrusted => JoltOpeningId::untrusted_advice(JoltRelationId::RamValCheck),
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RamRafEvaluationPublicValues<F: Field> {
    pub unmap_address: F,
}

impl<F: Field> RamRafEvaluationPublicValues<F> {
    pub fn value(&self, id: RamRafEvaluationPublic) -> F {
        match id {
            RamRafEvaluationPublic::UnmapAddress => self.unmap_address,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RamOutputCheckPublicValues<F: Field> {
    pub eq_io_mask: F,
    pub neg_eq_io_mask_val_io: F,
}

impl<F: Field> RamOutputCheckPublicValues<F> {
    pub fn value(&self, id: RamOutputCheckPublic) -> F {
        match id {
            RamOutputCheckPublic::EqIoMask => self.eq_io_mask,
            RamOutputCheckPublic::NegEqIoMaskValIo => self.neg_eq_io_mask_val_io,
        }
    }
}

pub fn ra_claim_reduction_input_openings() -> [JoltOpeningId; 3] {
    [ram_ra_raf_evaluation(), ram_ra(), ram_ra_val_check()]
}

pub fn ra_claim_reduction_output_openings() -> [JoltOpeningId; 1] {
    [ram_ra_claim_reduction()]
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

pub fn ra_virtualization_eq_cycle_polynomial<F>(ram_reduced_cycle: &[F]) -> Polynomial<F>
where
    F: Field,
{
    let eq_point = ram_reduced_cycle.iter().rev().copied().collect::<Vec<_>>();
    Polynomial::new(EqPolynomial::<F>::evals(&eq_point, None))
}

pub fn hamming_booleanity_eq_cycle_polynomial<F>(stage1_cycle_binding: &[F]) -> Polynomial<F>
where
    F: Field,
{
    Polynomial::new(EqPolynomial::<F>::evals(stage1_cycle_binding, None))
}

pub fn ra_virtualization_input_openings() -> [JoltOpeningId; 1] {
    [ram_ra_claim_reduction()]
}

pub fn ra_virtualization_output_openings(
    dimensions: RamRaVirtualizationDimensions,
) -> Vec<JoltOpeningId> {
    (0..dimensions.num_committed_ra_polys())
        .map(ra_virtualization_committed_ram_ra_opening)
        .collect()
}

pub fn ra_virtualization_committed_ram_ra_opening(index: usize) -> JoltOpeningId {
    committed_ram_ra(index)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RamRaVirtualizationPublicValues<F: Field> {
    pub eq_cycle: F,
}

impl<F: Field> RamRaVirtualizationPublicValues<F> {
    pub fn value(&self, id: RamRaVirtualizationPublic) -> F {
        match id {
            RamRaVirtualizationPublic::EqCycle => self.eq_cycle,
        }
    }
}

pub fn hamming_booleanity_output_openings() -> [JoltOpeningId; 1] {
    [ram_hamming_weight()]
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RamHammingBooleanityPublicValues<F: Field> {
    pub eq_cycle: F,
}

impl<F: Field> RamHammingBooleanityPublicValues<F> {
    pub fn value(&self, id: RamHammingBooleanityPublic) -> F {
        match id {
            RamHammingBooleanityPublic::EqCycle => self.eq_cycle,
        }
    }
}

pub(crate) fn read_write_challenge<F>(id: RamReadWriteChallenge) -> JoltExpr<F>
where
    F: RingCore,
{
    challenge(JoltChallengeId::from(id))
}

pub(crate) fn read_write_public<F>(id: RamReadWritePublic) -> JoltExpr<F>
where
    F: RingCore,
{
    public(JoltPublicId::from(id))
}

pub(crate) fn ra_claim_reduction_challenge<F>(id: RamRaClaimReductionChallenge) -> JoltExpr<F>
where
    F: RingCore,
{
    challenge(JoltChallengeId::from(id))
}

pub(crate) fn ra_virtualization_public<F>(id: RamRaVirtualizationPublic) -> JoltExpr<F>
where
    F: RingCore,
{
    public(JoltPublicId::from(id))
}

pub(crate) fn hamming_booleanity_public<F>(id: RamHammingBooleanityPublic) -> JoltExpr<F>
where
    F: RingCore,
{
    public(JoltPublicId::from(id))
}

pub(crate) fn raf_evaluation_public<F>(id: RamRafEvaluationPublic) -> JoltExpr<F>
where
    F: RingCore,
{
    public(JoltPublicId::from(id))
}

pub(crate) fn output_check_public<F>(id: RamOutputCheckPublic) -> JoltExpr<F>
where
    F: RingCore,
{
    public(JoltPublicId::from(id))
}

pub(crate) fn ra_claim_reduction_public<F>(id: RamRaClaimReductionPublic) -> JoltExpr<F>
where
    F: RingCore,
{
    public(JoltPublicId::from(id))
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

pub(crate) fn ram_ra() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RamRa,
        JoltRelationId::RamReadWriteChecking,
    )
}

pub(crate) fn ram_val() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RamVal,
        JoltRelationId::RamReadWriteChecking,
    )
}

pub(crate) fn ram_val_final() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RamValFinal,
        JoltRelationId::RamOutputCheck,
    )
}

pub(crate) fn ram_inc() -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::RamInc,
        JoltRelationId::RamReadWriteChecking,
    )
}

pub(crate) fn ram_inc_val_check() -> JoltOpeningId {
    JoltOpeningId::committed(JoltCommittedPolynomial::RamInc, JoltRelationId::RamValCheck)
}

pub(crate) fn ram_ra_val_check() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(JoltVirtualPolynomial::RamRa, JoltRelationId::RamValCheck)
}

pub(crate) fn ram_address_spartan() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RamAddress,
        JoltRelationId::SpartanOuter,
    )
}

pub(crate) fn ram_ra_raf_evaluation() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RamRa,
        JoltRelationId::RamRafEvaluation,
    )
}

pub(crate) fn ram_ra_claim_reduction() -> JoltOpeningId {
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

pub(crate) fn ram_hamming_weight() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RamHammingWeight,
        JoltRelationId::RamHammingBooleanity,
    )
}

#[cfg(test)]
#[expect(clippy::panic)]
mod tests {
    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_poly::EqPolynomial;

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

    #[test]
    fn stage2_terminal_openings_are_ram_raf_then_output_check() {
        assert_eq!(
            stage2_terminal_output_openings(),
            [
                raf_evaluation_output_openings()[0],
                output_check_output_openings()[0]
            ]
        );
    }

    #[test]
    fn ra_virtualization_eq_cycle_polynomial_reverses_reduced_cycle() {
        let reduced_cycle = vec![Fr::from_u64(2), Fr::from_u64(3), Fr::from_u64(5)];
        let eq_point = vec![Fr::from_u64(5), Fr::from_u64(3), Fr::from_u64(2)];

        assert_eq!(
            ra_virtualization_eq_cycle_polynomial(&reduced_cycle).evals(),
            EqPolynomial::<Fr>::evals(&eq_point, None)
        );
    }

    #[test]
    fn hamming_booleanity_eq_cycle_polynomial_uses_stage1_cycle_binding() {
        let cycle_binding = vec![Fr::from_u64(2), Fr::from_u64(3), Fr::from_u64(5)];

        assert_eq!(
            hamming_booleanity_eq_cycle_polynomial(&cycle_binding).evals(),
            EqPolynomial::<Fr>::evals(&cycle_binding, None)
        );
    }
}
