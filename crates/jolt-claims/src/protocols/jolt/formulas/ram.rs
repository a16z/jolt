use jolt_field::{Field, RingCore};

use crate::{challenge, constant, opening, pow2, public};

use super::super::{
    JoltAdviceKind, JoltChallengeId, JoltCommittedPolynomial, JoltExpr, JoltOpeningId,
    JoltPublicId, JoltStageClaims, JoltStageId, JoltVirtualPolynomial, RamHammingBooleanityPublic,
    RamOutputCheckPublic, RamRaClaimReductionChallenge, RamRaClaimReductionPublic,
    RamRaVirtualizationPublic, RamRafEvaluationPublic, RamReadWriteChallenge, RamValCheckChallenge,
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

impl From<(usize, usize)> for RamRaVirtualizationDimensions {
    fn from((log_t, committed_ra_polys): (usize, usize)) -> Self {
        Self::new(log_t, committed_ra_polys)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RamValCheckInit<F> {
    public_eval: F,
    advice_contributions: Vec<RamValCheckAdviceContribution<F>>,
}

impl<F> RamValCheckInit<F> {
    pub fn full(init_eval: F) -> Self {
        Self {
            public_eval: init_eval,
            advice_contributions: Vec::new(),
        }
    }

    pub fn decomposed<I>(public_eval: F, advice_contributions: I) -> Self
    where
        I: IntoIterator<Item = RamValCheckAdviceContribution<F>>,
    {
        Self {
            public_eval,
            advice_contributions: advice_contributions.into_iter().collect(),
        }
    }
}

impl<F> From<F> for RamValCheckInit<F> {
    fn from(value: F) -> Self {
        Self::full(value)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RamValCheckAdviceContribution<F> {
    pub neg_selector: F,
    pub opening: JoltOpeningId,
}

impl<F> RamValCheckAdviceContribution<F> {
    pub fn new(neg_selector: F, opening: JoltOpeningId) -> Self {
        Self {
            neg_selector,
            opening,
        }
    }

    pub fn untrusted(neg_selector: F) -> Self {
        Self::new(
            neg_selector,
            JoltOpeningId::untrusted_advice(JoltStageId::RamValCheck),
        )
    }

    pub fn trusted(neg_selector: F) -> Self {
        Self::new(
            neg_selector,
            JoltOpeningId::trusted_advice(JoltStageId::RamValCheck),
        )
    }
}

pub fn read_write_checking<F>(dimensions: ReadWriteDimensions) -> JoltStageClaims<F>
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

    JoltStageClaims::new(
        JoltStageId::RamReadWriteChecking,
        dimensions.read_write_sumcheck(),
        input,
        output,
    )
}

pub fn val_check<F>(dimensions: TraceDimensions, init: RamValCheckInit<F>) -> JoltStageClaims<F>
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

    JoltStageClaims::new(
        JoltStageId::RamValCheck,
        dimensions.sumcheck(3),
        input,
        output,
    )
}

pub fn raf_evaluation<F>(dimensions: RamRafEvaluationDimensions) -> JoltStageClaims<F>
where
    F: RingCore,
{
    let input =
        constant(pow2::<F>(dimensions.phase3_cycle_rounds())) * opening(ram_address_spartan());
    let output = raf_evaluation_public(RamRafEvaluationPublic::UnmapAddress)
        * opening(ram_ra_raf_evaluation());

    JoltStageClaims::new(
        JoltStageId::RamRafEvaluation,
        dimensions.sumcheck(),
        input,
        output,
    )
}

pub fn output_check<F>(dimensions: ReadWriteDimensions) -> JoltStageClaims<F>
where
    F: RingCore,
{
    let output = output_check_public(RamOutputCheckPublic::EqIoMask) * opening(ram_val_final())
        + output_check_public(RamOutputCheckPublic::NegEqIoMaskValIo);

    JoltStageClaims::new(
        JoltStageId::RamOutputCheck,
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
        JoltAdviceKind::Trusted => JoltOpeningId::trusted_advice(JoltStageId::RamValCheck),
        JoltAdviceKind::Untrusted => JoltOpeningId::untrusted_advice(JoltStageId::RamValCheck),
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

pub fn ra_claim_reduction<F>(dimensions: TraceDimensions) -> JoltStageClaims<F>
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

    JoltStageClaims::new(
        JoltStageId::RamRaClaimReduction,
        dimensions.sumcheck(2),
        input,
        output,
    )
}

pub fn ra_virtualization<F>(dimensions: RamRaVirtualizationDimensions) -> JoltStageClaims<F>
where
    F: RingCore,
{
    let input = opening(ram_ra_claim_reduction());
    let output = ra_virtualization_public(RamRaVirtualizationPublic::EqCycle)
        * committed_ram_ra_product(dimensions);

    JoltStageClaims::new(
        JoltStageId::RamRaVirtualization,
        dimensions.sumcheck(),
        input,
        output,
    )
}

pub fn hamming_booleanity<F>(dimensions: TraceDimensions) -> JoltStageClaims<F>
where
    F: RingCore,
{
    let eq_cycle = hamming_booleanity_public(RamHammingBooleanityPublic::EqCycle);
    let h = opening(ram_hamming_weight());
    let output = eq_cycle * (h.clone() * h.clone() - h);

    JoltStageClaims::new(
        JoltStageId::RamHammingBooleanity,
        dimensions.sumcheck(3),
        JoltExpr::zero(),
        output,
    )
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

fn ra_virtualization_public<F>(id: RamRaVirtualizationPublic) -> JoltExpr<F>
where
    F: RingCore,
{
    public(JoltPublicId::from(id))
}

fn hamming_booleanity_public<F>(id: RamHammingBooleanityPublic) -> JoltExpr<F>
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
    for contribution in init.advice_contributions {
        eval = eval - JoltExpr::constant(contribution.neg_selector) * opening(contribution.opening);
    }
    eval
}

fn ram_read_value() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RamReadValue,
        JoltStageId::SpartanOuter,
    )
}

fn ram_write_value() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RamWriteValue,
        JoltStageId::SpartanOuter,
    )
}

fn ram_ra() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RamRa,
        JoltStageId::RamReadWriteChecking,
    )
}

fn ram_val() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RamVal,
        JoltStageId::RamReadWriteChecking,
    )
}

fn ram_val_final() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RamValFinal,
        JoltStageId::RamOutputCheck,
    )
}

fn ram_inc() -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::RamInc,
        JoltStageId::RamReadWriteChecking,
    )
}

fn ram_inc_val_check() -> JoltOpeningId {
    JoltOpeningId::committed(JoltCommittedPolynomial::RamInc, JoltStageId::RamValCheck)
}

fn ram_ra_val_check() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(JoltVirtualPolynomial::RamRa, JoltStageId::RamValCheck)
}

fn ram_address_spartan() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(JoltVirtualPolynomial::RamAddress, JoltStageId::SpartanOuter)
}

fn ram_ra_raf_evaluation() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(JoltVirtualPolynomial::RamRa, JoltStageId::RamRafEvaluation)
}

fn ram_ra_claim_reduction() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RamRa,
        JoltStageId::RamRaClaimReduction,
    )
}

fn committed_ram_ra(index: usize) -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::RamRa(index),
        JoltStageId::RamRaVirtualization,
    )
}

fn ram_hamming_weight() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RamHammingWeight,
        JoltStageId::RamHammingBooleanity,
    )
}

#[cfg(test)]
#[expect(clippy::panic)]
mod tests {
    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};

    fn trace_dimensions() -> TraceDimensions {
        5.into()
    }

    fn read_write_dimensions() -> ReadWriteDimensions {
        (5, 4, 2, 1).into()
    }

    fn raf_evaluation_dimensions() -> RamRafEvaluationDimensions {
        RamRafEvaluationDimensions::try_from(read_write_dimensions()).unwrap_or_else(|err| {
            panic!("test RAM RAF evaluation dimensions should be valid: {err}")
        })
    }

    fn ra_virtualization_dimensions(committed_ra_polys: usize) -> RamRaVirtualizationDimensions {
        (5, committed_ra_polys).into()
    }

    #[test]
    fn read_write_claims_expose_expected_dependencies() {
        let claims = read_write_checking::<Fr>(read_write_dimensions());

        assert_eq!(claims.id, JoltStageId::RamReadWriteChecking);
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

        let input = claims.input.expression.evaluate(
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
                | JoltChallengeId::SpartanShift(_) => zero,
            },
            |_| zero,
        );

        let output = claims.output.expression.evaluate(
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
                | JoltChallengeId::SpartanShift(_) => zero,
            },
            |_| zero,
        );

        assert_eq!(input, read + gamma * write);
        assert_eq!(output, eq * ra * (val + gamma * (val + inc)));
    }

    #[test]
    fn raf_evaluation_rejects_invalid_dimensions() {
        assert!(
            RamRafEvaluationDimensions::try_from(ReadWriteDimensions::from((3, 4, 4, 0))).is_err()
        );

        let dimensions = raf_evaluation_dimensions();
        assert_eq!(dimensions.phase3_cycle_rounds(), 3);

        let large_dimensions =
            RamRafEvaluationDimensions::try_from(ReadWriteDimensions::from((usize::MAX, 0, 0, 0)))
                .unwrap_or_else(|err| {
                    panic!("large RAM RAF evaluation dimensions was rejected: {err}")
                });
        assert_eq!(large_dimensions.phase3_cycle_rounds(), usize::MAX);
    }

    #[test]
    fn raf_evaluation_exposes_expected_dependencies() {
        let dimensions = raf_evaluation_dimensions();
        let claims = raf_evaluation::<Fr>(dimensions);

        assert_eq!(claims.id, JoltStageId::RamRafEvaluation);
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

        let input = claims.input.expression.evaluate(
            |id| match *id {
                id if id == ram_address_spartan() => address,
                _ => zero,
            },
            |_| zero,
            |_| zero,
        );

        let output = claims.output.expression.evaluate(
            |id| match *id {
                id if id == ram_ra_raf_evaluation() => ram_ra,
                _ => zero,
            },
            |_| zero,
            |id| match *id {
                JoltPublicId::RamRafEvaluation(RamRafEvaluationPublic::UnmapAddress) => unmap,
                JoltPublicId::TraceLength
                | JoltPublicId::PaddedTraceLength
                | JoltPublicId::BytecodeLength
                | JoltPublicId::MemorySize
                | JoltPublicId::RamOutputCheck(_)
                | JoltPublicId::RamRaClaimReduction(_)
                | JoltPublicId::RamRaVirtualization(_)
                | JoltPublicId::RamHammingBooleanity(_)
                | JoltPublicId::PublicInput(_)
                | JoltPublicId::PublicOutput(_)
                | JoltPublicId::Booleanity(_)
                | JoltPublicId::IncClaimReduction(_)
                | JoltPublicId::HammingWeightClaimReduction(_)
                | JoltPublicId::BytecodeReadRaf(_)
                | JoltPublicId::AdviceClaimReduction(_)
                | JoltPublicId::SpartanShift(_)
                | JoltPublicId::SpartanProductVirtualization(_)
                | JoltPublicId::SpartanOuter(_) => zero,
            },
        );

        assert_eq!(input, address * Fr::from_u64(8));
        assert_eq!(output, unmap * ram_ra);
    }

    #[test]
    fn output_check_exposes_expected_dependencies() {
        let claims = output_check::<Fr>(read_write_dimensions());

        assert_eq!(claims.id, JoltStageId::RamOutputCheck);
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
            .expression
            .evaluate(|_| zero, |_| zero, |_| zero);
        let output = claims.output.expression.evaluate(
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
                JoltPublicId::TraceLength
                | JoltPublicId::PaddedTraceLength
                | JoltPublicId::BytecodeLength
                | JoltPublicId::MemorySize
                | JoltPublicId::RamRafEvaluation(_)
                | JoltPublicId::RamRaClaimReduction(_)
                | JoltPublicId::RamRaVirtualization(_)
                | JoltPublicId::RamHammingBooleanity(_)
                | JoltPublicId::PublicInput(_)
                | JoltPublicId::PublicOutput(_)
                | JoltPublicId::Booleanity(_)
                | JoltPublicId::IncClaimReduction(_)
                | JoltPublicId::HammingWeightClaimReduction(_)
                | JoltPublicId::BytecodeReadRaf(_)
                | JoltPublicId::AdviceClaimReduction(_)
                | JoltPublicId::SpartanShift(_)
                | JoltPublicId::SpartanProductVirtualization(_)
                | JoltPublicId::SpartanOuter(_) => zero,
            },
        );

        assert_eq!(input, zero);
        assert_eq!(output, eq_io_mask * val_final + neg_eq_io_mask_val_io);
    }

    #[test]
    fn ra_claim_reduction_exposes_expected_dependencies() {
        let claims = ra_claim_reduction::<Fr>(trace_dimensions());

        assert_eq!(claims.id, JoltStageId::RamRaClaimReduction);
        assert_eq!(claims.sumcheck, trace_dimensions().sumcheck(2));
        assert_eq!(
            claims.input.required_openings,
            vec![ram_ra_raf_evaluation(), ram_ra(), ram_ra_val_check()]
        );
        assert_eq!(
            claims.output.required_openings,
            vec![ram_ra_claim_reduction()]
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
        let zero = Fr::from_u64(0);

        let input = claims.input.expression.evaluate(
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
                | JoltChallengeId::SpartanShift(_) => zero,
            },
            |_| zero,
        );

        let output = claims.output.expression.evaluate(
            |id| match *id {
                id if id == ram_ra_claim_reduction() => reduced,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::RamRaClaimReduction(RamRaClaimReductionChallenge::Gamma) => gamma,
                JoltChallengeId::RamReadWrite(_)
                | JoltChallengeId::RamValCheck(_)
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
                | JoltChallengeId::SpartanShift(_) => zero,
            },
            |id| match *id {
                JoltPublicId::RamRaClaimReduction(RamRaClaimReductionPublic::EqCycleRaf) => eq_raf,
                JoltPublicId::RamRaClaimReduction(RamRaClaimReductionPublic::EqCycleReadWrite) => {
                    eq_rw
                }
                JoltPublicId::RamRaClaimReduction(RamRaClaimReductionPublic::EqCycleValCheck) => {
                    eq_val
                }
                JoltPublicId::TraceLength
                | JoltPublicId::PaddedTraceLength
                | JoltPublicId::BytecodeLength
                | JoltPublicId::MemorySize
                | JoltPublicId::RamRafEvaluation(_)
                | JoltPublicId::RamOutputCheck(_)
                | JoltPublicId::RamRaVirtualization(_)
                | JoltPublicId::RamHammingBooleanity(_)
                | JoltPublicId::PublicInput(_)
                | JoltPublicId::PublicOutput(_)
                | JoltPublicId::Booleanity(_)
                | JoltPublicId::IncClaimReduction(_)
                | JoltPublicId::HammingWeightClaimReduction(_)
                | JoltPublicId::BytecodeReadRaf(_)
                | JoltPublicId::AdviceClaimReduction(_)
                | JoltPublicId::SpartanShift(_)
                | JoltPublicId::SpartanProductVirtualization(_)
                | JoltPublicId::SpartanOuter(_) => zero,
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

        assert_eq!(claims.id, JoltStageId::RamRaVirtualization);
        assert_eq!(claims.sumcheck, dimensions.sumcheck());
        assert_eq!(
            claims.input.required_openings,
            vec![ram_ra_claim_reduction()]
        );
        assert_eq!(
            claims.output.required_openings,
            vec![
                committed_ram_ra(0),
                committed_ram_ra(1),
                committed_ram_ra(2),
            ]
        );
        assert!(claims.required_challenges().is_empty());
        assert_eq!(
            claims.output.required_publics,
            vec![JoltPublicId::from(RamRaVirtualizationPublic::EqCycle)]
        );
        assert_eq!(
            claims.required_openings(),
            vec![
                ram_ra_claim_reduction(),
                committed_ram_ra(0),
                committed_ram_ra(1),
                committed_ram_ra(2),
            ]
        );
        assert_eq!(claims.num_challenges(), 0);
    }

    #[test]
    fn ra_virtualization_evaluates_like_core_formula() {
        let dimensions = ra_virtualization_dimensions(3);
        let claims = ra_virtualization::<Fr>(dimensions);

        let reduced = Fr::from_u64(3);
        let committed = [Fr::from_u64(5), Fr::from_u64(7), Fr::from_u64(11)];
        let eq_cycle = Fr::from_u64(13);
        let zero = Fr::from_u64(0);

        let input = claims.input.expression.evaluate(
            |id| match *id {
                id if id == ram_ra_claim_reduction() => reduced,
                _ => zero,
            },
            |_| zero,
            |_| zero,
        );

        let output = claims.output.expression.evaluate(
            |id| match *id {
                id if id == committed_ram_ra(0) => committed[0],
                id if id == committed_ram_ra(1) => committed[1],
                id if id == committed_ram_ra(2) => committed[2],
                _ => zero,
            },
            |_| zero,
            |id| match *id {
                JoltPublicId::RamRaVirtualization(RamRaVirtualizationPublic::EqCycle) => eq_cycle,
                JoltPublicId::TraceLength
                | JoltPublicId::PaddedTraceLength
                | JoltPublicId::BytecodeLength
                | JoltPublicId::MemorySize
                | JoltPublicId::RamRafEvaluation(_)
                | JoltPublicId::RamOutputCheck(_)
                | JoltPublicId::RamRaClaimReduction(_)
                | JoltPublicId::RamHammingBooleanity(_)
                | JoltPublicId::PublicInput(_)
                | JoltPublicId::PublicOutput(_)
                | JoltPublicId::Booleanity(_)
                | JoltPublicId::IncClaimReduction(_)
                | JoltPublicId::HammingWeightClaimReduction(_)
                | JoltPublicId::BytecodeReadRaf(_)
                | JoltPublicId::AdviceClaimReduction(_)
                | JoltPublicId::SpartanShift(_)
                | JoltPublicId::SpartanProductVirtualization(_)
                | JoltPublicId::SpartanOuter(_) => zero,
            },
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

        assert_eq!(claims.id, JoltStageId::RamHammingBooleanity);
        assert_eq!(claims.sumcheck, trace_dimensions().sumcheck(3));
        assert!(claims.input.required_openings.is_empty());
        assert_eq!(claims.output.required_openings, vec![ram_hamming_weight()]);
        assert!(claims.required_challenges().is_empty());
        assert_eq!(
            claims.output.required_publics,
            vec![JoltPublicId::from(RamHammingBooleanityPublic::EqCycle)]
        );
        assert_eq!(
            claims.required_publics(),
            vec![JoltPublicId::from(RamHammingBooleanityPublic::EqCycle)]
        );
        assert_eq!(claims.num_challenges(), 0);
    }

    #[test]
    fn hamming_booleanity_evaluates_like_core_formula() {
        let claims = hamming_booleanity::<Fr>(trace_dimensions());

        let h = Fr::from_u64(7);
        let eq_cycle = Fr::from_u64(11);
        let zero = Fr::from_u64(0);

        let input = claims
            .input
            .expression
            .evaluate(|_| zero, |_| zero, |_| zero);
        let output = claims.output.expression.evaluate(
            |id| match *id {
                id if id == ram_hamming_weight() => h,
                _ => zero,
            },
            |_| zero,
            |id| match *id {
                JoltPublicId::RamHammingBooleanity(RamHammingBooleanityPublic::EqCycle) => eq_cycle,
                JoltPublicId::TraceLength
                | JoltPublicId::PaddedTraceLength
                | JoltPublicId::BytecodeLength
                | JoltPublicId::MemorySize
                | JoltPublicId::RamRafEvaluation(_)
                | JoltPublicId::RamOutputCheck(_)
                | JoltPublicId::RamRaClaimReduction(_)
                | JoltPublicId::RamRaVirtualization(_)
                | JoltPublicId::PublicInput(_)
                | JoltPublicId::PublicOutput(_)
                | JoltPublicId::Booleanity(_)
                | JoltPublicId::IncClaimReduction(_)
                | JoltPublicId::HammingWeightClaimReduction(_)
                | JoltPublicId::BytecodeReadRaf(_)
                | JoltPublicId::AdviceClaimReduction(_)
                | JoltPublicId::SpartanShift(_)
                | JoltPublicId::SpartanProductVirtualization(_)
                | JoltPublicId::SpartanOuter(_) => zero,
            },
        );

        assert_eq!(input, zero);
        assert_eq!(output, eq_cycle * (h * h - h));
    }

    #[test]
    fn val_check_full_init_exposes_expected_dependencies() {
        let claims = val_check::<Fr>(trace_dimensions(), Fr::from_u64(3).into());

        assert_eq!(claims.id, JoltStageId::RamValCheck);
        assert_eq!(claims.sumcheck, trace_dimensions().sumcheck(3));
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
                RamValCheckAdviceContribution::untrusted(-Fr::from_u64(5)),
                RamValCheckAdviceContribution::trusted(-Fr::from_u64(7)),
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

        let input = claims.input.expression.evaluate(
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
                | JoltChallengeId::SpartanShift(_) => zero,
            },
            |_| zero,
        );

        let output = claims.output.expression.evaluate(
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
                | JoltChallengeId::SpartanShift(_) => zero,
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
                RamValCheckAdviceContribution::untrusted(untrusted_neg_selector),
                RamValCheckAdviceContribution::trusted(trusted_neg_selector),
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

        let input = claims.input.expression.evaluate(
            |id| match *id {
                id if id == ram_val() => val_rw,
                id if id == ram_val_final() => val_final,
                id if id == JoltOpeningId::untrusted_advice(JoltStageId::RamValCheck) => {
                    untrusted_advice
                }
                id if id == JoltOpeningId::trusted_advice(JoltStageId::RamValCheck) => {
                    trusted_advice
                }
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::RamValCheck(RamValCheckChallenge::Gamma) => gamma,
                JoltChallengeId::RamValCheck(RamValCheckChallenge::LtCyclePlusGamma)
                | JoltChallengeId::RamReadWrite(_)
                | JoltChallengeId::RamRaClaimReduction(_)
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
                | JoltChallengeId::SpartanShift(_) => zero,
            },
            |_| zero,
        );

        assert_eq!(
            input,
            (val_rw - init_eval) + gamma * (val_final - init_eval)
        );
    }
}
