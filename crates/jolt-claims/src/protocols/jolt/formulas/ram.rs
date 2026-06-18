use jolt_field::{Field, RingCore};

use crate::{challenge, constant, opening, public};

use super::super::{
    JoltAdviceKind, JoltChallengeId, JoltCommittedPolynomial, JoltExpr, JoltOpeningId,
    JoltPublicId, JoltRelationClaims, JoltRelationId, JoltVirtualPolynomial,
    RamHammingBooleanityChallenge, RamOutputCheckPublic, RamRaClaimReductionChallenge,
    RamRaClaimReductionPublic, RamRaVirtualizationChallenge, RamRafEvaluationPublic,
    RamReadWriteChallenge, RamValCheckChallenge,
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
    public_eval: F,
    contributions: Vec<RamValCheckInitContribution<F>>,
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
    pub neg_selector: F,
    pub opening: JoltOpeningId,
}

impl<F> RamValCheckInitContribution<F> {
    pub fn new(neg_selector: F, opening: JoltOpeningId) -> Self {
        Self {
            neg_selector,
            opening,
        }
    }

    pub fn untrusted(neg_selector: F) -> Self {
        Self::new(
            neg_selector,
            JoltOpeningId::untrusted_advice(JoltRelationId::RamValCheck),
        )
    }

    pub fn trusted(neg_selector: F) -> Self {
        Self::new(
            neg_selector,
            JoltOpeningId::trusted_advice(JoltRelationId::RamValCheck),
        )
    }

    pub fn program_image(neg_selector: F) -> Self {
        Self::new(
            neg_selector,
            super::claim_reductions::program_image::ram_val_check_contribution_opening(),
        )
    }
}

pub fn read_write_checking<F>(dimensions: ReadWriteDimensions) -> JoltRelationClaims<F>
where
    F: RingCore,
{
    let input = opening(ram_read_value())
        + read_write_challenge(RamReadWriteChallenge::Gamma) * opening(ram_write_value());

    let output = read_write_challenge(RamReadWriteChallenge::EqCycle)
        * opening(ram_ra())
        * opening(ram_val())
        + read_write_challenge(RamReadWriteChallenge::EqCycle)
            * read_write_challenge(RamReadWriteChallenge::Gamma)
            * opening(ram_ra())
            * opening(ram_val())
        + read_write_challenge(RamReadWriteChallenge::EqCycle)
            * read_write_challenge(RamReadWriteChallenge::Gamma)
            * opening(ram_ra())
            * opening(ram_inc());

    JoltRelationClaims::new(
        JoltRelationId::RamReadWriteChecking,
        dimensions.read_write_sumcheck(),
        input,
        output,
    )
}

pub const fn val_check_sumcheck(dimensions: TraceDimensions) -> JoltSumcheckSpec {
    dimensions.sumcheck(3)
}

pub fn val_check<F>(dimensions: TraceDimensions, init: RamValCheckInit<F>) -> JoltRelationClaims<F>
where
    F: RingCore,
{
    let gamma = val_check_challenge(RamValCheckChallenge::Gamma);
    let init_eval = ram_val_init_eval(init);

    let input = opening(ram_val()) + gamma.clone() * opening(ram_val_final())
        - (JoltExpr::one() + gamma) * init_eval;

    let output = val_check_challenge(RamValCheckChallenge::LtCyclePlusGamma)
        * opening(ram_inc_val_check())
        * opening(ram_ra_val_check());

    JoltRelationClaims::new(
        JoltRelationId::RamValCheck,
        val_check_sumcheck(dimensions),
        input,
        output,
    )
}

pub fn raf_evaluation<F>(dimensions: RamRafEvaluationDimensions) -> JoltRelationClaims<F>
where
    F: RingCore,
{
    let input =
        constant(F::pow2(dimensions.phase3_cycle_rounds())) * opening(ram_address_spartan());
    let output = raf_evaluation_public(RamRafEvaluationPublic::UnmapAddress)
        * opening(ram_ra_raf_evaluation());

    JoltRelationClaims::new(
        JoltRelationId::RamRafEvaluation,
        dimensions.sumcheck(),
        input,
        output,
    )
}

pub fn output_check<F>(dimensions: ReadWriteDimensions) -> JoltRelationClaims<F>
where
    F: RingCore,
{
    let output = output_check_public(RamOutputCheckPublic::EqIoMask) * opening(ram_val_final())
        + output_check_public(RamOutputCheckPublic::NegEqIoMaskValIo);

    JoltRelationClaims::new(
        JoltRelationId::RamOutputCheck,
        dimensions.output_check_sumcheck(),
        JoltExpr::zero(),
        output,
    )
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

pub fn ra_claim_reduction<F>(dimensions: TraceDimensions) -> JoltRelationClaims<F>
where
    F: RingCore,
{
    let gamma = ra_claim_reduction_challenge(RamRaClaimReductionChallenge::Gamma);
    let input = opening(ram_ra_raf_evaluation())
        + gamma.clone() * opening(ram_ra())
        + gamma.clone().pow(2) * opening(ram_ra_val_check());

    let output = (ra_claim_reduction_public(RamRaClaimReductionPublic::EqCycleRaf)
        + gamma.clone() * ra_claim_reduction_public(RamRaClaimReductionPublic::EqCycleReadWrite)
        + gamma.pow(2) * ra_claim_reduction_public(RamRaClaimReductionPublic::EqCycleValCheck))
        * opening(ram_ra_claim_reduction());

    JoltRelationClaims::new(
        JoltRelationId::RamRaClaimReduction,
        dimensions.sumcheck(2),
        input,
        output,
    )
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

pub fn ra_virtualization<F>(dimensions: RamRaVirtualizationDimensions) -> JoltRelationClaims<F>
where
    F: RingCore,
{
    let input = opening(ram_ra_claim_reduction());
    let output = ra_virtualization_challenge(RamRaVirtualizationChallenge::EqCycle)
        * committed_ram_ra_product(dimensions);

    JoltRelationClaims::new(
        JoltRelationId::RamRaVirtualization,
        dimensions.sumcheck(),
        input,
        output,
    )
}

pub fn hamming_booleanity<F>(dimensions: TraceDimensions) -> JoltRelationClaims<F>
where
    F: RingCore,
{
    let eq_cycle = hamming_booleanity_challenge(RamHammingBooleanityChallenge::EqCycle);
    let h = opening(ram_hamming_weight());
    let output = eq_cycle * (h.clone() * h.clone() - h);

    JoltRelationClaims::new(
        JoltRelationId::RamHammingBooleanity,
        dimensions.sumcheck(3),
        JoltExpr::zero(),
        output,
    )
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
pub struct RamRaVirtualizationChallengeValues<F: Field> {
    pub eq_cycle: F,
}

impl<F: Field> RamRaVirtualizationChallengeValues<F> {
    pub fn value(&self, id: RamRaVirtualizationChallenge) -> F {
        match id {
            RamRaVirtualizationChallenge::EqCycle => self.eq_cycle,
        }
    }
}

pub fn hamming_booleanity_output_openings() -> [JoltOpeningId; 1] {
    [ram_hamming_weight()]
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RamHammingBooleanityChallengeValues<F: Field> {
    pub eq_cycle: F,
}

impl<F: Field> RamHammingBooleanityChallengeValues<F> {
    pub fn value(&self, id: RamHammingBooleanityChallenge) -> F {
        match id {
            RamHammingBooleanityChallenge::EqCycle => self.eq_cycle,
        }
    }
}

fn read_write_challenge<F>(id: RamReadWriteChallenge) -> JoltExpr<F>
where
    F: RingCore,
{
    challenge(JoltChallengeId::from(id))
}

fn val_check_challenge<F>(id: RamValCheckChallenge) -> JoltExpr<F>
where
    F: RingCore,
{
    challenge(JoltChallengeId::from(id))
}

fn ra_claim_reduction_challenge<F>(id: RamRaClaimReductionChallenge) -> JoltExpr<F>
where
    F: RingCore,
{
    challenge(JoltChallengeId::from(id))
}

fn ra_virtualization_challenge<F>(id: RamRaVirtualizationChallenge) -> JoltExpr<F>
where
    F: RingCore,
{
    challenge(JoltChallengeId::from(id))
}

fn hamming_booleanity_challenge<F>(id: RamHammingBooleanityChallenge) -> JoltExpr<F>
where
    F: RingCore,
{
    challenge(JoltChallengeId::from(id))
}

fn raf_evaluation_public<F>(id: RamRafEvaluationPublic) -> JoltExpr<F>
where
    F: RingCore,
{
    public(JoltPublicId::from(id))
}

fn output_check_public<F>(id: RamOutputCheckPublic) -> JoltExpr<F>
where
    F: RingCore,
{
    public(JoltPublicId::from(id))
}

fn ra_claim_reduction_public<F>(id: RamRaClaimReductionPublic) -> JoltExpr<F>
where
    F: RingCore,
{
    public(JoltPublicId::from(id))
}

fn committed_ram_ra_product<F>(dimensions: RamRaVirtualizationDimensions) -> JoltExpr<F>
where
    F: RingCore,
{
    let mut product = JoltExpr::one();
    for index in 0..dimensions.num_committed_ra_polys() {
        product = product * opening(committed_ram_ra(index));
    }
    product
}

fn ram_val_init_eval<F>(init: RamValCheckInit<F>) -> JoltExpr<F>
where
    F: RingCore,
{
    let mut eval = JoltExpr::constant(init.public_eval);
    for contribution in init.contributions {
        eval = eval - JoltExpr::constant(contribution.neg_selector) * opening(contribution.opening);
    }
    eval
}

fn ram_read_value() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RamReadValue,
        JoltRelationId::SpartanOuter,
    )
}

fn ram_write_value() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RamWriteValue,
        JoltRelationId::SpartanOuter,
    )
}

fn ram_ra() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RamRa,
        JoltRelationId::RamReadWriteChecking,
    )
}

fn ram_val() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RamVal,
        JoltRelationId::RamReadWriteChecking,
    )
}

fn ram_val_final() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RamValFinal,
        JoltRelationId::RamOutputCheck,
    )
}

fn ram_inc() -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::RamInc,
        JoltRelationId::RamReadWriteChecking,
    )
}

fn ram_inc_val_check() -> JoltOpeningId {
    JoltOpeningId::committed(JoltCommittedPolynomial::RamInc, JoltRelationId::RamValCheck)
}

fn ram_ra_val_check() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(JoltVirtualPolynomial::RamRa, JoltRelationId::RamValCheck)
}

fn ram_address_spartan() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RamAddress,
        JoltRelationId::SpartanOuter,
    )
}

fn ram_ra_raf_evaluation() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RamRa,
        JoltRelationId::RamRafEvaluation,
    )
}

fn ram_ra_claim_reduction() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RamRa,
        JoltRelationId::RamRaClaimReduction,
    )
}

fn committed_ram_ra(index: usize) -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::RamRa(index),
        JoltRelationId::RamRaVirtualization,
    )
}

fn ram_hamming_weight() -> JoltOpeningId {
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

    fn trace_dimensions() -> TraceDimensions {
        TraceDimensions::new(5)
    }

    fn read_write_dimensions() -> ReadWriteDimensions {
        ReadWriteDimensions::new(5, 4, 2, 1)
    }

    fn raf_evaluation_dimensions() -> RamRafEvaluationDimensions {
        RamRafEvaluationDimensions::try_from(read_write_dimensions()).unwrap_or_else(|err| {
            panic!("test RAM RAF evaluation dimensions should be valid: {err}")
        })
    }

    fn ra_virtualization_dimensions(committed_ra_polys: usize) -> RamRaVirtualizationDimensions {
        RamRaVirtualizationDimensions::new(5, committed_ra_polys)
    }

    #[test]
    fn read_write_claims_expose_expected_dependencies() {
        let claims = read_write_checking::<Fr>(read_write_dimensions());

        assert_eq!(claims.id, JoltRelationId::RamReadWriteChecking);
        assert_eq!(
            claims.sumcheck,
            read_write_dimensions().read_write_sumcheck()
        );
        assert_eq!(
            claims.input.required_openings,
            read_write_checking_input_openings().to_vec()
        );
        assert_eq!(
            claims.output.required_openings,
            vec![ram_ra(), ram_val(), ram_inc()]
        );
        assert_eq!(
            claims.input.required_challenges,
            vec![JoltChallengeId::from(RamReadWriteChallenge::Gamma)]
        );
        assert_eq!(
            claims.output.required_challenges,
            vec![
                JoltChallengeId::from(RamReadWriteChallenge::EqCycle),
                JoltChallengeId::from(RamReadWriteChallenge::Gamma),
            ]
        );
        assert_eq!(
            claims.required_challenges(),
            vec![
                JoltChallengeId::from(RamReadWriteChallenge::Gamma),
                JoltChallengeId::from(RamReadWriteChallenge::EqCycle),
            ]
        );
        assert_eq!(
            claims.challenge_index(JoltChallengeId::from(RamReadWriteChallenge::EqCycle)),
            Some(1)
        );
        assert!(claims.required_publics().is_empty());
        assert_eq!(claims.num_challenges(), 2);
    }

    #[test]
    fn read_write_claims_evaluate_like_core_formula() {
        let claims = read_write_checking::<Fr>(read_write_dimensions());

        let read = Fr::from_u64(3);
        let write = Fr::from_u64(5);
        let ra = Fr::from_u64(7);
        let val = Fr::from_u64(11);
        let inc = Fr::from_u64(13);
        let gamma = Fr::from_u64(17);
        let eq = Fr::from_u64(19);
        let zero = Fr::from_u64(0);

        let input = claims.input.expression().evaluate(
            |id| match *id {
                id if id == ram_read_value() => read,
                id if id == ram_write_value() => write,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::RamReadWrite(RamReadWriteChallenge::Gamma) => gamma,
                JoltChallengeId::RamReadWrite(RamReadWriteChallenge::EqCycle)
                | JoltChallengeId::RamValCheck(_)
                | JoltChallengeId::RamRaClaimReduction(_)
                | JoltChallengeId::RamRaVirtualization(_)
                | JoltChallengeId::RamHammingBooleanity(_)
                | JoltChallengeId::RegistersReadWrite(_)
                | JoltChallengeId::RegistersValEvaluation(_)
                | JoltChallengeId::RegistersClaimReduction(_)
                | JoltChallengeId::InstructionClaimReduction(_)
                | JoltChallengeId::InstructionInput(_)
                | JoltChallengeId::InstructionReadRaf(_)
                | JoltChallengeId::InstructionRaVirtualization(_)
                | JoltChallengeId::Booleanity(_)
                | JoltChallengeId::IncClaimReduction(_)
                | JoltChallengeId::HammingWeightClaimReduction(_)
                | JoltChallengeId::BytecodeReadRaf(_)
                | JoltChallengeId::BytecodeClaimReduction(_)
                | JoltChallengeId::SpartanShift(_)
                | JoltChallengeId::FusedIncrementTranslation(_)
                | JoltChallengeId::FusedIncrementSourceLink(_)
                | JoltChallengeId::FusedIncrementInactiveZero(_)
                | JoltChallengeId::FusedIncrementInactiveSourceLink(_) => zero,
            },
            |_| zero,
        );

        let output = claims.output.expression().evaluate(
            |id| match *id {
                id if id == ram_ra() => ra,
                id if id == ram_val() => val,
                id if id == ram_inc() => inc,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::RamReadWrite(RamReadWriteChallenge::EqCycle) => eq,
                JoltChallengeId::RamReadWrite(RamReadWriteChallenge::Gamma) => gamma,
                JoltChallengeId::RamValCheck(_)
                | JoltChallengeId::RamRaClaimReduction(_)
                | JoltChallengeId::RamRaVirtualization(_)
                | JoltChallengeId::RamHammingBooleanity(_)
                | JoltChallengeId::RegistersReadWrite(_)
                | JoltChallengeId::RegistersValEvaluation(_)
                | JoltChallengeId::RegistersClaimReduction(_)
                | JoltChallengeId::InstructionClaimReduction(_)
                | JoltChallengeId::InstructionInput(_)
                | JoltChallengeId::InstructionReadRaf(_)
                | JoltChallengeId::InstructionRaVirtualization(_)
                | JoltChallengeId::Booleanity(_)
                | JoltChallengeId::IncClaimReduction(_)
                | JoltChallengeId::HammingWeightClaimReduction(_)
                | JoltChallengeId::BytecodeReadRaf(_)
                | JoltChallengeId::BytecodeClaimReduction(_)
                | JoltChallengeId::SpartanShift(_)
                | JoltChallengeId::FusedIncrementTranslation(_)
                | JoltChallengeId::FusedIncrementSourceLink(_)
                | JoltChallengeId::FusedIncrementInactiveZero(_)
                | JoltChallengeId::FusedIncrementInactiveSourceLink(_) => zero,
            },
            |_| zero,
        );

        assert_eq!(input, read + gamma * write);
        assert_eq!(output, eq * ra * (val + gamma * (val + inc)));
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
    fn raf_evaluation_exposes_expected_dependencies() {
        let dimensions = raf_evaluation_dimensions();
        let claims = raf_evaluation::<Fr>(dimensions);

        assert_eq!(claims.id, JoltRelationId::RamRafEvaluation);
        assert_eq!(claims.sumcheck, dimensions.sumcheck());
        assert_eq!(
            claims.input.required_openings,
            raf_evaluation_input_openings().to_vec()
        );
        assert_eq!(
            claims.output.required_openings,
            raf_evaluation_output_openings().to_vec()
        );
        assert!(claims.input.required_challenges.is_empty());
        assert!(claims.output.required_challenges.is_empty());
        assert!(claims.required_challenges().is_empty());
        assert_eq!(
            claims.output.required_publics,
            vec![JoltPublicId::from(RamRafEvaluationPublic::UnmapAddress)]
        );
        assert_eq!(
            claims.required_publics(),
            vec![JoltPublicId::from(RamRafEvaluationPublic::UnmapAddress)]
        );
        assert_eq!(claims.num_challenges(), 0);
    }

    #[test]
    fn raf_evaluation_evaluates_like_core_formula() {
        let dimensions = raf_evaluation_dimensions();
        let claims = raf_evaluation::<Fr>(dimensions);

        let address = Fr::from_u64(7);
        let ram_ra = Fr::from_u64(11);
        let unmap = Fr::from_u64(13);
        let zero = Fr::from_u64(0);

        let input = claims.input.expression().evaluate(
            |id| match *id {
                id if id == ram_address_spartan() => address,
                _ => zero,
            },
            |_| zero,
            |_| zero,
        );

        let output = claims.output.expression().evaluate(
            |id| match *id {
                id if id == ram_ra_raf_evaluation() => ram_ra,
                _ => zero,
            },
            |_| zero,
            |id| match *id {
                JoltPublicId::RamRafEvaluation(RamRafEvaluationPublic::UnmapAddress) => unmap,
                _ => zero,
            },
        );

        assert_eq!(input, address * Fr::from_u64(8));
        assert_eq!(output, unmap * ram_ra);
    }

    #[test]
    fn output_check_exposes_expected_dependencies() {
        let claims = output_check::<Fr>(read_write_dimensions());

        assert_eq!(claims.id, JoltRelationId::RamOutputCheck);
        assert_eq!(
            claims.sumcheck,
            read_write_dimensions().output_check_sumcheck()
        );
        assert!(claims.input.required_openings.is_empty());
        assert_eq!(
            claims.output.required_openings,
            output_check_output_openings().to_vec()
        );
        assert!(claims.input.required_challenges.is_empty());
        assert!(claims.output.required_challenges.is_empty());
        assert!(claims.required_challenges().is_empty());
        assert!(claims.input.required_publics.is_empty());
        assert_eq!(
            claims.output.required_publics,
            vec![
                JoltPublicId::from(RamOutputCheckPublic::EqIoMask),
                JoltPublicId::from(RamOutputCheckPublic::NegEqIoMaskValIo),
            ]
        );
        assert_eq!(
            claims.required_publics(),
            vec![
                JoltPublicId::from(RamOutputCheckPublic::EqIoMask),
                JoltPublicId::from(RamOutputCheckPublic::NegEqIoMaskValIo),
            ]
        );
        assert_eq!(claims.required_openings(), vec![ram_val_final()]);
        assert_eq!(claims.num_challenges(), 0);
    }

    #[test]
    fn output_check_evaluates_like_core_formula() {
        let claims = output_check::<Fr>(read_write_dimensions());

        let val_final = Fr::from_u64(7);
        let eq_io_mask = Fr::from_u64(11);
        let neg_eq_io_mask_val_io = -Fr::from_u64(13);
        let zero = Fr::from_u64(0);

        let input = claims
            .input
            .expression()
            .evaluate(|_| zero, |_| zero, |_| zero);
        let output = claims.output.expression().evaluate(
            |id| match *id {
                id if id == ram_val_final() => val_final,
                _ => zero,
            },
            |_| zero,
            |id| match *id {
                JoltPublicId::RamOutputCheck(RamOutputCheckPublic::EqIoMask) => eq_io_mask,
                JoltPublicId::RamOutputCheck(RamOutputCheckPublic::NegEqIoMaskValIo) => {
                    neg_eq_io_mask_val_io
                }
                _ => zero,
            },
        );

        assert_eq!(input, zero);
        assert_eq!(output, eq_io_mask * val_final + neg_eq_io_mask_val_io);
    }

    #[test]
    fn ra_claim_reduction_exposes_expected_dependencies() {
        let claims = ra_claim_reduction::<Fr>(trace_dimensions());

        assert_eq!(claims.id, JoltRelationId::RamRaClaimReduction);
        assert_eq!(claims.sumcheck, trace_dimensions().sumcheck(2));
        assert_eq!(
            claims.input.required_openings,
            ra_claim_reduction_input_openings().to_vec()
        );
        assert_eq!(
            claims.output.required_openings,
            ra_claim_reduction_output_openings().to_vec()
        );
        assert_eq!(
            claims.input.required_challenges,
            vec![JoltChallengeId::from(RamRaClaimReductionChallenge::Gamma)]
        );
        assert_eq!(
            claims.output.required_challenges,
            vec![JoltChallengeId::from(RamRaClaimReductionChallenge::Gamma)]
        );
        assert_eq!(
            claims.required_challenges(),
            vec![JoltChallengeId::from(RamRaClaimReductionChallenge::Gamma)]
        );
        assert_eq!(
            claims.output.required_publics,
            vec![
                JoltPublicId::from(RamRaClaimReductionPublic::EqCycleRaf),
                JoltPublicId::from(RamRaClaimReductionPublic::EqCycleReadWrite),
                JoltPublicId::from(RamRaClaimReductionPublic::EqCycleValCheck),
            ]
        );
        assert_eq!(
            claims.required_publics(),
            vec![
                JoltPublicId::from(RamRaClaimReductionPublic::EqCycleRaf),
                JoltPublicId::from(RamRaClaimReductionPublic::EqCycleReadWrite),
                JoltPublicId::from(RamRaClaimReductionPublic::EqCycleValCheck),
            ]
        );
        assert_eq!(claims.num_challenges(), 1);
    }

    #[test]
    fn ra_claim_reduction_evaluates_like_core_formula() {
        let claims = ra_claim_reduction::<Fr>(trace_dimensions());

        let raf = Fr::from_u64(3);
        let rw = Fr::from_u64(5);
        let val = Fr::from_u64(7);
        let gamma = Fr::from_u64(11);
        let reduced = Fr::from_u64(13);
        let eq_raf = Fr::from_u64(17);
        let eq_rw = Fr::from_u64(19);
        let eq_val = Fr::from_u64(23);
        let public_values = RamRaClaimReductionPublicValues {
            eq_cycle_raf: eq_raf,
            eq_cycle_read_write: eq_rw,
            eq_cycle_val_check: eq_val,
        };
        let zero = Fr::from_u64(0);

        let input = claims.input.expression().evaluate(
            |id| match *id {
                id if id == ram_ra_raf_evaluation() => raf,
                id if id == ram_ra() => rw,
                id if id == ram_ra_val_check() => val,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::RamRaClaimReduction(RamRaClaimReductionChallenge::Gamma) => gamma,
                JoltChallengeId::RamReadWrite(_)
                | JoltChallengeId::RamValCheck(_)
                | JoltChallengeId::RamRaVirtualization(_)
                | JoltChallengeId::RamHammingBooleanity(_)
                | JoltChallengeId::RegistersReadWrite(_)
                | JoltChallengeId::RegistersValEvaluation(_)
                | JoltChallengeId::RegistersClaimReduction(_)
                | JoltChallengeId::InstructionClaimReduction(_)
                | JoltChallengeId::InstructionInput(_)
                | JoltChallengeId::InstructionReadRaf(_)
                | JoltChallengeId::InstructionRaVirtualization(_)
                | JoltChallengeId::Booleanity(_)
                | JoltChallengeId::IncClaimReduction(_)
                | JoltChallengeId::HammingWeightClaimReduction(_)
                | JoltChallengeId::BytecodeReadRaf(_)
                | JoltChallengeId::BytecodeClaimReduction(_)
                | JoltChallengeId::SpartanShift(_)
                | JoltChallengeId::FusedIncrementTranslation(_)
                | JoltChallengeId::FusedIncrementSourceLink(_)
                | JoltChallengeId::FusedIncrementInactiveZero(_)
                | JoltChallengeId::FusedIncrementInactiveSourceLink(_) => zero,
            },
            |_| zero,
        );

        let output = claims.output.expression().evaluate(
            |id| match *id {
                id if id == ram_ra_claim_reduction() => reduced,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::RamRaClaimReduction(RamRaClaimReductionChallenge::Gamma) => gamma,
                JoltChallengeId::RamReadWrite(_)
                | JoltChallengeId::RamValCheck(_)
                | JoltChallengeId::RamRaVirtualization(_)
                | JoltChallengeId::RamHammingBooleanity(_)
                | JoltChallengeId::RegistersReadWrite(_)
                | JoltChallengeId::RegistersValEvaluation(_)
                | JoltChallengeId::RegistersClaimReduction(_)
                | JoltChallengeId::InstructionClaimReduction(_)
                | JoltChallengeId::InstructionInput(_)
                | JoltChallengeId::InstructionReadRaf(_)
                | JoltChallengeId::InstructionRaVirtualization(_)
                | JoltChallengeId::Booleanity(_)
                | JoltChallengeId::IncClaimReduction(_)
                | JoltChallengeId::HammingWeightClaimReduction(_)
                | JoltChallengeId::BytecodeReadRaf(_)
                | JoltChallengeId::BytecodeClaimReduction(_)
                | JoltChallengeId::SpartanShift(_)
                | JoltChallengeId::FusedIncrementTranslation(_)
                | JoltChallengeId::FusedIncrementSourceLink(_)
                | JoltChallengeId::FusedIncrementInactiveZero(_)
                | JoltChallengeId::FusedIncrementInactiveSourceLink(_) => zero,
            },
            |id| match *id {
                JoltPublicId::RamRaClaimReduction(id) => public_values.value(id),
                _ => zero,
            },
        );

        assert_eq!(input, raf + gamma * rw + gamma * gamma * val);
        assert_eq!(
            output,
            (eq_raf + gamma * eq_rw + gamma * gamma * eq_val) * reduced
        );
    }

    #[test]
    fn ra_virtualization_supports_empty_ra_product() {
        let claims = ra_virtualization::<Fr>(ra_virtualization_dimensions(0));

        assert!(claims.output.required_openings.is_empty());
    }

    #[test]
    fn ra_virtualization_exposes_expected_dependencies() {
        let dimensions = ra_virtualization_dimensions(3);
        let claims = ra_virtualization::<Fr>(dimensions);

        assert_eq!(claims.id, JoltRelationId::RamRaVirtualization);
        assert_eq!(claims.sumcheck, dimensions.sumcheck());
        assert_eq!(
            claims.input.required_openings,
            ra_virtualization_input_openings()
        );
        assert_eq!(
            claims.output.required_openings,
            ra_virtualization_output_openings(dimensions)
        );
        assert_eq!(
            claims.output.required_challenges,
            vec![JoltChallengeId::from(RamRaVirtualizationChallenge::EqCycle)]
        );
        assert_eq!(
            claims.required_challenges(),
            vec![JoltChallengeId::from(RamRaVirtualizationChallenge::EqCycle)]
        );
        assert!(claims.required_publics().is_empty());
        assert_eq!(
            claims.required_openings(),
            vec![
                ram_ra_claim_reduction(),
                committed_ram_ra(0),
                committed_ram_ra(1),
                committed_ram_ra(2),
            ]
        );
        assert_eq!(claims.num_challenges(), 1);
    }

    #[test]
    fn ra_virtualization_evaluates_like_core_formula() {
        let dimensions = ra_virtualization_dimensions(3);
        let claims = ra_virtualization::<Fr>(dimensions);

        let reduced = Fr::from_u64(3);
        let committed = [Fr::from_u64(5), Fr::from_u64(7), Fr::from_u64(11)];
        let eq_cycle = Fr::from_u64(13);
        let zero = Fr::from_u64(0);

        let input = claims.input.expression().evaluate(
            |id| match *id {
                id if id == ram_ra_claim_reduction() => reduced,
                _ => zero,
            },
            |_| zero,
            |_| zero,
        );

        let output = claims.output.expression().evaluate(
            |id| match *id {
                id if id == committed_ram_ra(0) => committed[0],
                id if id == committed_ram_ra(1) => committed[1],
                id if id == committed_ram_ra(2) => committed[2],
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::RamRaVirtualization(RamRaVirtualizationChallenge::EqCycle) => {
                    eq_cycle
                }
                _ => zero,
            },
            |_| zero,
        );

        assert_eq!(input, reduced);
        assert_eq!(
            output,
            eq_cycle * committed[0] * committed[1] * committed[2]
        );
    }

    #[test]
    fn hamming_booleanity_exposes_expected_dependencies() {
        let claims = hamming_booleanity::<Fr>(trace_dimensions());

        assert_eq!(claims.id, JoltRelationId::RamHammingBooleanity);
        assert_eq!(claims.sumcheck, trace_dimensions().sumcheck(3));
        assert!(claims.input.required_openings.is_empty());
        assert_eq!(
            claims.output.required_openings,
            hamming_booleanity_output_openings()
        );
        assert_eq!(
            claims.output.required_challenges,
            vec![JoltChallengeId::from(
                RamHammingBooleanityChallenge::EqCycle
            )]
        );
        assert_eq!(
            claims.required_challenges(),
            vec![JoltChallengeId::from(
                RamHammingBooleanityChallenge::EqCycle
            )]
        );
        assert!(claims.required_publics().is_empty());
        assert_eq!(claims.num_challenges(), 1);
    }

    #[test]
    fn hamming_booleanity_evaluates_like_core_formula() {
        let claims = hamming_booleanity::<Fr>(trace_dimensions());

        let h = Fr::from_u64(7);
        let eq_cycle = Fr::from_u64(11);
        let zero = Fr::from_u64(0);

        let input = claims
            .input
            .expression()
            .evaluate(|_| zero, |_| zero, |_| zero);
        let output = claims.output.expression().evaluate(
            |id| match *id {
                id if id == ram_hamming_weight() => h,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::RamHammingBooleanity(RamHammingBooleanityChallenge::EqCycle) => {
                    eq_cycle
                }
                _ => zero,
            },
            |_| zero,
        );

        assert_eq!(input, zero);
        assert_eq!(output, eq_cycle * (h * h - h));
    }

    #[test]
    fn val_check_full_init_exposes_expected_dependencies() {
        let claims = val_check::<Fr>(trace_dimensions(), Fr::from_u64(3).into());

        assert_eq!(claims.id, JoltRelationId::RamValCheck);
        assert_eq!(claims.sumcheck, val_check_sumcheck(trace_dimensions()));
        assert_eq!(
            claims.input.required_openings,
            val_check_input_openings().to_vec()
        );
        assert_eq!(
            claims.output.required_openings,
            vec![ram_inc_val_check(), ram_ra_val_check()]
        );
        assert_eq!(
            val_check_output_openings(),
            [ram_ra_val_check(), ram_inc_val_check()]
        );
        assert_eq!(
            claims.input.required_challenges,
            vec![JoltChallengeId::from(RamValCheckChallenge::Gamma)]
        );
        assert_eq!(
            claims.output.required_challenges,
            vec![JoltChallengeId::from(
                RamValCheckChallenge::LtCyclePlusGamma
            )]
        );
        assert_eq!(
            claims.required_challenges(),
            vec![
                JoltChallengeId::from(RamValCheckChallenge::Gamma),
                JoltChallengeId::from(RamValCheckChallenge::LtCyclePlusGamma),
            ]
        );
        assert_eq!(
            claims.challenge_index(JoltChallengeId::from(
                RamValCheckChallenge::LtCyclePlusGamma
            )),
            Some(1)
        );
        assert!(claims.required_publics().is_empty());
        assert_eq!(claims.num_challenges(), 2);
    }

    #[test]
    fn val_check_decomposed_init_exposes_advice_openings() {
        let init = RamValCheckInit::decomposed(
            Fr::from_u64(3),
            [
                RamValCheckInitContribution::untrusted(-Fr::from_u64(5)),
                RamValCheckInitContribution::trusted(-Fr::from_u64(7)),
            ],
        );
        let claims = val_check::<Fr>(trace_dimensions(), init);

        assert_eq!(
            claims.input.required_openings,
            vec![
                ram_val(),
                ram_val_final(),
                val_check_advice_opening(JoltAdviceKind::Untrusted),
                val_check_advice_opening(JoltAdviceKind::Trusted),
            ]
        );
        assert_eq!(
            claims.required_openings(),
            vec![
                ram_val(),
                ram_val_final(),
                val_check_advice_opening(JoltAdviceKind::Untrusted),
                val_check_advice_opening(JoltAdviceKind::Trusted),
                ram_inc_val_check(),
                ram_ra_val_check(),
            ]
        );
    }

    #[test]
    fn val_check_full_init_evaluates_like_core_formula() {
        let init_eval = Fr::from_u64(3);
        let claims = val_check::<Fr>(trace_dimensions(), init_eval.into());

        let val_rw = Fr::from_u64(5);
        let val_final = Fr::from_u64(7);
        let gamma = Fr::from_u64(11);
        let inc = Fr::from_u64(13);
        let wa = Fr::from_u64(17);
        let lt_plus_gamma = Fr::from_u64(19);
        let zero = Fr::from_u64(0);

        let input = claims.input.expression().evaluate(
            |id| match *id {
                id if id == ram_val() => val_rw,
                id if id == ram_val_final() => val_final,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::RamValCheck(RamValCheckChallenge::Gamma) => gamma,
                JoltChallengeId::RamValCheck(RamValCheckChallenge::LtCyclePlusGamma)
                | JoltChallengeId::RamReadWrite(_)
                | JoltChallengeId::RamRaClaimReduction(_)
                | JoltChallengeId::RamRaVirtualization(_)
                | JoltChallengeId::RamHammingBooleanity(_)
                | JoltChallengeId::RegistersReadWrite(_)
                | JoltChallengeId::RegistersValEvaluation(_)
                | JoltChallengeId::RegistersClaimReduction(_)
                | JoltChallengeId::InstructionClaimReduction(_)
                | JoltChallengeId::InstructionInput(_)
                | JoltChallengeId::InstructionReadRaf(_)
                | JoltChallengeId::InstructionRaVirtualization(_)
                | JoltChallengeId::Booleanity(_)
                | JoltChallengeId::IncClaimReduction(_)
                | JoltChallengeId::HammingWeightClaimReduction(_)
                | JoltChallengeId::BytecodeReadRaf(_)
                | JoltChallengeId::BytecodeClaimReduction(_)
                | JoltChallengeId::SpartanShift(_)
                | JoltChallengeId::FusedIncrementTranslation(_)
                | JoltChallengeId::FusedIncrementSourceLink(_)
                | JoltChallengeId::FusedIncrementInactiveZero(_)
                | JoltChallengeId::FusedIncrementInactiveSourceLink(_) => zero,
            },
            |_| zero,
        );

        let output = claims.output.expression().evaluate(
            |id| match *id {
                id if id == ram_inc_val_check() => inc,
                id if id == ram_ra_val_check() => wa,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::RamValCheck(RamValCheckChallenge::LtCyclePlusGamma) => {
                    lt_plus_gamma
                }
                JoltChallengeId::RamValCheck(RamValCheckChallenge::Gamma)
                | JoltChallengeId::RamReadWrite(_)
                | JoltChallengeId::RamRaClaimReduction(_)
                | JoltChallengeId::RamRaVirtualization(_)
                | JoltChallengeId::RamHammingBooleanity(_)
                | JoltChallengeId::RegistersReadWrite(_)
                | JoltChallengeId::RegistersValEvaluation(_)
                | JoltChallengeId::RegistersClaimReduction(_)
                | JoltChallengeId::InstructionClaimReduction(_)
                | JoltChallengeId::InstructionInput(_)
                | JoltChallengeId::InstructionReadRaf(_)
                | JoltChallengeId::InstructionRaVirtualization(_)
                | JoltChallengeId::Booleanity(_)
                | JoltChallengeId::IncClaimReduction(_)
                | JoltChallengeId::HammingWeightClaimReduction(_)
                | JoltChallengeId::BytecodeReadRaf(_)
                | JoltChallengeId::BytecodeClaimReduction(_)
                | JoltChallengeId::SpartanShift(_)
                | JoltChallengeId::FusedIncrementTranslation(_)
                | JoltChallengeId::FusedIncrementSourceLink(_)
                | JoltChallengeId::FusedIncrementInactiveZero(_)
                | JoltChallengeId::FusedIncrementInactiveSourceLink(_) => zero,
            },
            |_| zero,
        );

        assert_eq!(
            input,
            (val_rw - init_eval) + gamma * (val_final - init_eval)
        );
        assert_eq!(output, inc * wa * lt_plus_gamma);
    }

    #[test]
    fn val_check_decomposed_init_evaluates_like_full_init() {
        let public_eval = Fr::from_u64(3);
        let untrusted_neg_selector = -Fr::from_u64(5);
        let trusted_neg_selector = -Fr::from_u64(7);
        let init = RamValCheckInit::decomposed(
            public_eval,
            [
                RamValCheckInitContribution::untrusted(untrusted_neg_selector),
                RamValCheckInitContribution::trusted(trusted_neg_selector),
            ],
        );
        let claims = val_check::<Fr>(trace_dimensions(), init);

        let val_rw = Fr::from_u64(11);
        let val_final = Fr::from_u64(13);
        let gamma = Fr::from_u64(17);
        let untrusted_advice = Fr::from_u64(19);
        let trusted_advice = Fr::from_u64(23);
        let zero = Fr::from_u64(0);
        let init_eval = public_eval
            - untrusted_neg_selector * untrusted_advice
            - trusted_neg_selector * trusted_advice;

        let input = claims.input.expression().evaluate(
            |id| match *id {
                id if id == ram_val() => val_rw,
                id if id == ram_val_final() => val_final,
                id if id == JoltOpeningId::untrusted_advice(JoltRelationId::RamValCheck) => {
                    untrusted_advice
                }
                id if id == JoltOpeningId::trusted_advice(JoltRelationId::RamValCheck) => {
                    trusted_advice
                }
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::RamValCheck(RamValCheckChallenge::Gamma) => gamma,
                JoltChallengeId::RamValCheck(RamValCheckChallenge::LtCyclePlusGamma)
                | JoltChallengeId::RamReadWrite(_)
                | JoltChallengeId::RamRaClaimReduction(_)
                | JoltChallengeId::RamRaVirtualization(_)
                | JoltChallengeId::RamHammingBooleanity(_)
                | JoltChallengeId::RegistersReadWrite(_)
                | JoltChallengeId::RegistersValEvaluation(_)
                | JoltChallengeId::RegistersClaimReduction(_)
                | JoltChallengeId::InstructionClaimReduction(_)
                | JoltChallengeId::InstructionInput(_)
                | JoltChallengeId::InstructionReadRaf(_)
                | JoltChallengeId::InstructionRaVirtualization(_)
                | JoltChallengeId::Booleanity(_)
                | JoltChallengeId::IncClaimReduction(_)
                | JoltChallengeId::HammingWeightClaimReduction(_)
                | JoltChallengeId::BytecodeReadRaf(_)
                | JoltChallengeId::BytecodeClaimReduction(_)
                | JoltChallengeId::SpartanShift(_)
                | JoltChallengeId::FusedIncrementTranslation(_)
                | JoltChallengeId::FusedIncrementSourceLink(_)
                | JoltChallengeId::FusedIncrementInactiveZero(_)
                | JoltChallengeId::FusedIncrementInactiveSourceLink(_) => zero,
            },
            |_| zero,
        );

        assert_eq!(
            input,
            (val_rw - init_eval) + gamma * (val_final - init_eval)
        );
    }
}
