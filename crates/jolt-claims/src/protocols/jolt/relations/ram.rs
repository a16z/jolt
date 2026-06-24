//! RAM symbolic sumcheck relations.

use jolt_field::RingCore;

use crate::protocols::jolt::formulas::ram::{
    committed_ram_ra_product, hamming_booleanity_public, output_check_public,
    ra_claim_reduction_challenge, ra_claim_reduction_public, ra_virtualization_public,
    raf_evaluation_public, ram_address_spartan, ram_hamming_weight, ram_inc, ram_ra,
    ram_ra_claim_reduction, ram_ra_raf_evaluation, ram_ra_val_check, ram_read_value, ram_val,
    ram_val_final, ram_write_value, read_write_challenge, read_write_public,
    RamRaVirtualizationDimensions, RamRafEvaluationDimensions,
};
use crate::protocols::jolt::{
    JoltExpr, JoltRelationId, JoltSumcheckSpec, RamHammingBooleanityPublic, RamOutputCheckPublic,
    RamRaClaimReductionChallenge, RamRaClaimReductionPublic, RamRaVirtualizationPublic,
    RamRafEvaluationPublic, RamReadWriteChallenge, RamReadWritePublic, ReadWriteDimensions,
    TraceDimensions,
};
use crate::SymbolicSumcheck;
use crate::{constant, opening};

/// The RAM read/write-checking sumcheck: folds the read and write values by
/// `gamma` on the input side, and reconstructs them from `ra`, `val`, and `inc`
/// weighted by the cycle-`eq` public on the output side.
pub struct ReadWriteChecking {
    shape: ReadWriteDimensions,
}

impl SymbolicSumcheck for ReadWriteChecking {
    type RelationId = JoltRelationId;
    type OpeningId = crate::protocols::jolt::JoltOpeningId;
    type PublicId = crate::protocols::jolt::JoltPublicId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = ReadWriteDimensions;

    fn new(shape: ReadWriteDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::RamReadWriteChecking
    }

    fn sumcheck(&self) -> JoltSumcheckSpec {
        self.shape.read_write_sumcheck()
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        opening(ram_read_value())
            + read_write_challenge(RamReadWriteChallenge::Gamma) * opening(ram_write_value())
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        read_write_public(RamReadWritePublic::EqCycle) * opening(ram_ra()) * opening(ram_val())
            + read_write_public(RamReadWritePublic::EqCycle)
                * read_write_challenge(RamReadWriteChallenge::Gamma)
                * opening(ram_ra())
                * opening(ram_val())
            + read_write_public(RamReadWritePublic::EqCycle)
                * read_write_challenge(RamReadWriteChallenge::Gamma)
                * opening(ram_ra())
                * opening(ram_inc())
    }
}

/// The RAM RAF-evaluation sumcheck: scales the Spartan RAM address opening by
/// `2^phase3_cycle_rounds` on the input side, and matches it against `ra`
/// weighted by the `UnmapAddress` public on the output side.
pub struct RafEvaluation {
    shape: RamRafEvaluationDimensions,
}

impl SymbolicSumcheck for RafEvaluation {
    type RelationId = JoltRelationId;
    type OpeningId = crate::protocols::jolt::JoltOpeningId;
    type PublicId = crate::protocols::jolt::JoltPublicId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = RamRafEvaluationDimensions;

    fn new(shape: RamRafEvaluationDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::RamRafEvaluation
    }

    fn sumcheck(&self) -> JoltSumcheckSpec {
        self.shape.sumcheck()
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        constant(F::pow2(self.shape.phase3_cycle_rounds())) * opening(ram_address_spartan())
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        raf_evaluation_public(RamRafEvaluationPublic::UnmapAddress)
            * opening(ram_ra_raf_evaluation())
    }
}

/// The RAM output-check sumcheck: a degree-one output that pins `Val_final` on
/// the I/O region (via `EqIoMask`) and offsets by the masked I/O value; no
/// input claim.
pub struct OutputCheck {
    shape: ReadWriteDimensions,
}

impl SymbolicSumcheck for OutputCheck {
    type RelationId = JoltRelationId;
    type OpeningId = crate::protocols::jolt::JoltOpeningId;
    type PublicId = crate::protocols::jolt::JoltPublicId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = ReadWriteDimensions;

    fn new(shape: ReadWriteDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::RamOutputCheck
    }

    fn sumcheck(&self) -> JoltSumcheckSpec {
        self.shape.output_check_sumcheck()
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        JoltExpr::zero()
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        output_check_public(RamOutputCheckPublic::EqIoMask) * opening(ram_val_final())
            + output_check_public(RamOutputCheckPublic::NegEqIoMaskValIo)
    }
}

/// The RAM `ra` claim-reduction sumcheck: folds the three `ra` openings (RAF,
/// read/write, val-check) by `gamma` on the input side, and matches the reduced
/// `ra` opening weighted by the matching cycle-`eq` publics on the output side.
pub struct RaClaimReduction {
    shape: TraceDimensions,
}

impl SymbolicSumcheck for RaClaimReduction {
    type RelationId = JoltRelationId;
    type OpeningId = crate::protocols::jolt::JoltOpeningId;
    type PublicId = crate::protocols::jolt::JoltPublicId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = TraceDimensions;

    fn new(shape: TraceDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::RamRaClaimReduction
    }

    fn sumcheck(&self) -> JoltSumcheckSpec {
        self.shape.sumcheck(2)
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = ra_claim_reduction_challenge(RamRaClaimReductionChallenge::Gamma);
        opening(ram_ra_raf_evaluation())
            + gamma.clone() * opening(ram_ra())
            + gamma.clone().pow(2) * opening(ram_ra_val_check())
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = ra_claim_reduction_challenge(RamRaClaimReductionChallenge::Gamma);
        (ra_claim_reduction_public(RamRaClaimReductionPublic::EqCycleRaf)
            + gamma.clone()
                * ra_claim_reduction_public(RamRaClaimReductionPublic::EqCycleReadWrite)
            + gamma.pow(2) * ra_claim_reduction_public(RamRaClaimReductionPublic::EqCycleValCheck))
            * opening(ram_ra_claim_reduction())
    }
}

/// The RAM `ra` virtualization sumcheck: equates the reduced `ra` opening on the
/// input side with the product of the committed per-`d` `ra` openings, weighted
/// by the cycle-`eq` public, on the output side.
pub struct RaVirtualization {
    shape: RamRaVirtualizationDimensions,
}

impl SymbolicSumcheck for RaVirtualization {
    type RelationId = JoltRelationId;
    type OpeningId = crate::protocols::jolt::JoltOpeningId;
    type PublicId = crate::protocols::jolt::JoltPublicId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = RamRaVirtualizationDimensions;

    fn new(shape: RamRaVirtualizationDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::RamRaVirtualization
    }

    fn sumcheck(&self) -> JoltSumcheckSpec {
        self.shape.sumcheck()
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        opening(ram_ra_claim_reduction())
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        ra_virtualization_public(RamRaVirtualizationPublic::EqCycle)
            * committed_ram_ra_product(self.shape)
    }
}

/// The RAM Hamming-booleanity sumcheck: a degree-three output enforcing that the
/// Hamming-weight opening is boolean (`h^2 - h == 0`) at each cycle, weighted by
/// the cycle-`eq` public; no input claim.
pub struct HammingBooleanity {
    shape: TraceDimensions,
}

impl SymbolicSumcheck for HammingBooleanity {
    type RelationId = JoltRelationId;
    type OpeningId = crate::protocols::jolt::JoltOpeningId;
    type PublicId = crate::protocols::jolt::JoltPublicId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = TraceDimensions;

    fn new(shape: TraceDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::RamHammingBooleanity
    }

    fn sumcheck(&self) -> JoltSumcheckSpec {
        self.shape.sumcheck(3)
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        JoltExpr::zero()
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let eq_cycle = hamming_booleanity_public(RamHammingBooleanityPublic::EqCycle);
        let h = opening(ram_hamming_weight());
        eq_cycle * (h.clone() * h.clone() - h)
    }
}

#[cfg(test)]
#[expect(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::protocols::jolt::formulas::ram::committed_ram_ra;
    use crate::protocols::jolt::{JoltChallengeId, JoltPublicId};
    use jolt_field::Fr;

    fn trace_dimensions() -> TraceDimensions {
        TraceDimensions::new(5)
    }

    fn read_write_dimensions() -> ReadWriteDimensions {
        ReadWriteDimensions::new(5, 4, 2, 1)
    }

    fn raf_evaluation_dimensions() -> RamRafEvaluationDimensions {
        RamRafEvaluationDimensions::try_from(read_write_dimensions())
            .expect("test RAM RAF evaluation dimensions should be valid")
    }

    fn ra_virtualization_dimensions(committed_ra_polys: usize) -> RamRaVirtualizationDimensions {
        RamRaVirtualizationDimensions::new(5, committed_ra_polys)
    }

    #[test]
    fn read_write_symbolic_matches_dependencies() {
        let relation = ReadWriteChecking::new(read_write_dimensions());

        assert_eq!(
            ReadWriteChecking::id(),
            JoltRelationId::RamReadWriteChecking
        );
        assert_eq!(
            relation.sumcheck(),
            read_write_dimensions().read_write_sumcheck()
        );
        assert_eq!(
            relation.required_openings::<Fr>(),
            vec![
                ram_read_value(),
                ram_write_value(),
                ram_ra(),
                ram_val(),
                ram_inc()
            ]
        );
        assert_eq!(
            relation.required_challenges::<Fr>(),
            vec![JoltChallengeId::from(RamReadWriteChallenge::Gamma)]
        );
        assert_eq!(
            relation.required_publics::<Fr>(),
            vec![JoltPublicId::from(RamReadWritePublic::EqCycle)]
        );
    }

    #[test]
    fn raf_evaluation_symbolic_matches_dependencies() {
        let relation = RafEvaluation::new(raf_evaluation_dimensions());

        assert_eq!(RafEvaluation::id(), JoltRelationId::RamRafEvaluation);
        assert_eq!(relation.sumcheck(), raf_evaluation_dimensions().sumcheck());
        assert_eq!(
            relation.required_openings::<Fr>(),
            vec![ram_address_spartan(), ram_ra_raf_evaluation()]
        );
        assert!(relation.required_challenges::<Fr>().is_empty());
        assert_eq!(
            relation.required_publics::<Fr>(),
            vec![JoltPublicId::from(RamRafEvaluationPublic::UnmapAddress)]
        );
    }

    #[test]
    fn output_check_symbolic_matches_dependencies() {
        let relation = OutputCheck::new(read_write_dimensions());

        assert_eq!(OutputCheck::id(), JoltRelationId::RamOutputCheck);
        assert_eq!(
            relation.sumcheck(),
            read_write_dimensions().output_check_sumcheck()
        );
        assert_eq!(relation.required_openings::<Fr>(), vec![ram_val_final()]);
        assert!(relation.required_challenges::<Fr>().is_empty());
        assert_eq!(
            relation.required_publics::<Fr>(),
            vec![
                JoltPublicId::from(RamOutputCheckPublic::EqIoMask),
                JoltPublicId::from(RamOutputCheckPublic::NegEqIoMaskValIo),
            ]
        );
    }

    #[test]
    fn ra_claim_reduction_symbolic_matches_dependencies() {
        let relation = RaClaimReduction::new(trace_dimensions());

        assert_eq!(RaClaimReduction::id(), JoltRelationId::RamRaClaimReduction);
        assert_eq!(relation.sumcheck(), trace_dimensions().sumcheck(2));
        assert_eq!(
            relation.required_openings::<Fr>(),
            vec![
                ram_ra_raf_evaluation(),
                ram_ra(),
                ram_ra_val_check(),
                ram_ra_claim_reduction(),
            ]
        );
        assert_eq!(
            relation.required_challenges::<Fr>(),
            vec![JoltChallengeId::from(RamRaClaimReductionChallenge::Gamma)]
        );
        assert_eq!(
            relation.required_publics::<Fr>(),
            vec![
                JoltPublicId::from(RamRaClaimReductionPublic::EqCycleRaf),
                JoltPublicId::from(RamRaClaimReductionPublic::EqCycleReadWrite),
                JoltPublicId::from(RamRaClaimReductionPublic::EqCycleValCheck),
            ]
        );
    }

    #[test]
    fn ra_virtualization_supports_empty_ra_product() {
        let relation = RaVirtualization::new(ra_virtualization_dimensions(0));

        assert_eq!(
            relation.required_openings::<Fr>(),
            vec![ram_ra_claim_reduction()]
        );
    }

    #[test]
    fn ra_virtualization_symbolic_matches_dependencies() {
        let relation = RaVirtualization::new(ra_virtualization_dimensions(3));

        assert_eq!(RaVirtualization::id(), JoltRelationId::RamRaVirtualization);
        assert_eq!(
            relation.sumcheck(),
            ra_virtualization_dimensions(3).sumcheck()
        );
        assert_eq!(
            relation.required_openings::<Fr>(),
            vec![
                ram_ra_claim_reduction(),
                committed_ram_ra(0),
                committed_ram_ra(1),
                committed_ram_ra(2),
            ]
        );
        assert!(relation.required_challenges::<Fr>().is_empty());
        assert_eq!(
            relation.required_publics::<Fr>(),
            vec![JoltPublicId::from(RamRaVirtualizationPublic::EqCycle)]
        );
    }

    #[test]
    fn hamming_booleanity_symbolic_matches_dependencies() {
        let relation = HammingBooleanity::new(trace_dimensions());

        assert_eq!(
            HammingBooleanity::id(),
            JoltRelationId::RamHammingBooleanity
        );
        assert_eq!(relation.sumcheck(), trace_dimensions().sumcheck(3));
        assert_eq!(
            relation.required_openings::<Fr>(),
            vec![ram_hamming_weight()]
        );
        assert!(relation.required_challenges::<Fr>().is_empty());
        assert_eq!(
            relation.required_publics::<Fr>(),
            vec![JoltPublicId::from(RamHammingBooleanityPublic::EqCycle)]
        );
    }
}
