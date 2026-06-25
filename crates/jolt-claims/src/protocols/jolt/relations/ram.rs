//! RAM symbolic sumcheck relations.

use jolt_field::RingCore;

use crate::protocols::jolt::geometry::ram::{
    committed_ram_ra_product, ram_address_spartan, ram_hamming_weight, ram_inc, ram_inc_val_check,
    ram_ra, ram_ra_claim_reduction, ram_ra_raf_evaluation, ram_ra_val_check, ram_read_value,
    ram_val, ram_val_final, ram_write_value, RamRaVirtualizationDimensions,
    RamRafEvaluationDimensions,
};
use crate::protocols::jolt::{
    JoltChallengeId, JoltExpr, JoltOpeningId, JoltPublicId, JoltRelationId, JoltSumcheckSpec,
    RamHammingBooleanityPublic, RamOutputCheckPublic, RamRaClaimReductionChallenge,
    RamRaClaimReductionPublic, RamRaVirtualizationPublic, RamRafEvaluationPublic,
    RamReadWriteChallenge, RamReadWritePublic, RamValCheckChallenge, RamValCheckPublic,
    ReadWriteDimensions, TraceDimensions,
};
use crate::SymbolicSumcheck;
use crate::{challenge, constant, opening, public};

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

    fn spec(&self) -> JoltSumcheckSpec {
        self.shape.read_write_sumcheck()
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        opening(ram_read_value())
            + challenge(RamReadWriteChallenge::Gamma) * opening(ram_write_value())
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        public(RamReadWritePublic::EqCycle) * opening(ram_ra()) * opening(ram_val())
            + public(RamReadWritePublic::EqCycle)
                * challenge(RamReadWriteChallenge::Gamma)
                * opening(ram_ra())
                * opening(ram_val())
            + public(RamReadWritePublic::EqCycle)
                * challenge(RamReadWriteChallenge::Gamma)
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

    fn spec(&self) -> JoltSumcheckSpec {
        self.shape.sumcheck()
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        constant(F::pow2(self.shape.phase3_cycle_rounds())) * opening(ram_address_spartan())
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        public(RamRafEvaluationPublic::UnmapAddress) * opening(ram_ra_raf_evaluation())
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

    fn spec(&self) -> JoltSumcheckSpec {
        self.shape.output_check_sumcheck()
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        JoltExpr::zero()
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        public(RamOutputCheckPublic::EqIoMask) * opening(ram_val_final())
            + public(RamOutputCheckPublic::NegEqIoMaskValIo)
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

    fn spec(&self) -> JoltSumcheckSpec {
        self.shape.sumcheck(2)
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = challenge(RamRaClaimReductionChallenge::Gamma);
        opening(ram_ra_raf_evaluation())
            + gamma.clone() * opening(ram_ra())
            + gamma.clone().pow(2) * opening(ram_ra_val_check())
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = challenge(RamRaClaimReductionChallenge::Gamma);
        (public(RamRaClaimReductionPublic::EqCycleRaf)
            + gamma.clone() * public(RamRaClaimReductionPublic::EqCycleReadWrite)
            + gamma.pow(2) * public(RamRaClaimReductionPublic::EqCycleValCheck))
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

    fn spec(&self) -> JoltSumcheckSpec {
        self.shape.sumcheck()
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        opening(ram_ra_claim_reduction())
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        public(RamRaVirtualizationPublic::EqCycle) * committed_ram_ra_product(self.shape)
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

    fn spec(&self) -> JoltSumcheckSpec {
        self.shape.sumcheck(3)
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        JoltExpr::zero()
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let eq_cycle = public(RamHammingBooleanityPublic::EqCycle);
        let h = opening(ram_hamming_weight());
        eq_cycle * (h.clone() * h.clone() - h)
    }
}

/// One committed contribution to the `Val_init(r_address)` decomposition: a
/// `Public` selector weighting a committed advice / program-image `opening`. The
/// selector *value* is supplied by the concrete side (`resolve_public`); the
/// symbolic shape carries only the `(selector_id, opening_id)` structure, keeping
/// the relation field-independent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RamValContribution {
    pub selector: RamValCheckPublic,
    pub opening: JoltOpeningId,
}

/// The RAM value-check shape: the trace dimensions plus the present `Val_init`
/// contributions, in the canonical order the BlindFold constraint also uses
/// (program image first, then advice). An empty `contributions` is the full-init
/// form (`Val_init` is wholly public).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RamValCheckShape {
    pub dimensions: TraceDimensions,
    pub contributions: Vec<RamValContribution>,
}

/// The RAM value-check sumcheck. The input reconstructs `Val_init(r_address)` as
/// `public(InitEval) - Σ public(InitSelector)·opening(advice)` and folds it
/// against the read-write `val` and output-check `val_final` by `gamma`; the
/// output is the degree-two `LtCyclePlusGamma·inc·ra`. The `Val_init` scalars are
/// `Public` symbols resolved by the verifier — value-preserving versus the prior
/// baked coefficients (BlindFold bakes `Public` factors as matrix coefficients),
/// so the relation stays field-independent. See `specs/symbolic-sumcheck.md` §4.1.
pub struct RamValCheck {
    shape: RamValCheckShape,
}

impl SymbolicSumcheck for RamValCheck {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type PublicId = JoltPublicId;
    type ChallengeId = JoltChallengeId;
    type Shape = RamValCheckShape;

    fn new(shape: RamValCheckShape) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::RamValCheck
    }

    fn spec(&self) -> JoltSumcheckSpec {
        self.shape.dimensions.sumcheck(3)
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = challenge(JoltChallengeId::from(RamValCheckChallenge::Gamma));
        let mut init = public(JoltPublicId::from(RamValCheckPublic::InitEval));
        for contribution in &self.shape.contributions {
            init = init
                - public(JoltPublicId::from(contribution.selector)) * opening(contribution.opening);
        }
        opening(ram_val()) + gamma.clone() * opening(ram_val_final())
            - (JoltExpr::one() + gamma) * init
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        public(JoltPublicId::from(RamValCheckPublic::LtCyclePlusGamma))
            * opening(ram_inc_val_check())
            * opening(ram_ra_val_check())
    }
}

#[cfg(test)]
#[expect(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::protocols::jolt::geometry::ram::{
        committed_ram_ra, RamRaClaimReductionPublicValues,
    };
    use crate::protocols::jolt::{JoltChallengeId, JoltPublicId};
    use jolt_field::{Fr, FromPrimitiveInt};

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
    fn read_write_claims_evaluate_like_core_formula() {
        let relation = ReadWriteChecking::new(read_write_dimensions());

        let read = Fr::from_u64(3);
        let write = Fr::from_u64(5);
        let ra = Fr::from_u64(7);
        let val = Fr::from_u64(11);
        let inc = Fr::from_u64(13);
        let gamma = Fr::from_u64(17);
        let eq = Fr::from_u64(19);
        let zero = Fr::from_u64(0);

        let input = relation.input_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == ram_read_value() => read,
                id if id == ram_write_value() => write,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::RamReadWrite(RamReadWriteChallenge::Gamma) => gamma,
                JoltChallengeId::RamValCheck(_)
                | JoltChallengeId::RamRaClaimReduction(_)
                | JoltChallengeId::RegistersReadWrite(_)
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
                | JoltChallengeId::SpartanShift(_) => zero,
            },
            |_| zero,
        );

        let output = relation.output_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == ram_ra() => ra,
                id if id == ram_val() => val,
                id if id == ram_inc() => inc,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::RamReadWrite(RamReadWriteChallenge::Gamma) => gamma,
                JoltChallengeId::RamValCheck(_)
                | JoltChallengeId::RamRaClaimReduction(_)
                | JoltChallengeId::RegistersReadWrite(_)
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
                | JoltChallengeId::SpartanShift(_) => zero,
            },
            |id| match *id {
                JoltPublicId::RamReadWrite(RamReadWritePublic::EqCycle) => eq,
                _ => zero,
            },
        );

        assert_eq!(input, read + gamma * write);
        assert_eq!(output, eq * ra * (val + gamma * (val + inc)));
    }

    #[test]
    fn raf_evaluation_evaluates_like_core_formula() {
        let dimensions = raf_evaluation_dimensions();
        let relation = RafEvaluation::new(dimensions);

        let address = Fr::from_u64(7);
        let ram_ra = Fr::from_u64(11);
        let unmap = Fr::from_u64(13);
        let zero = Fr::from_u64(0);

        let input = relation.input_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == ram_address_spartan() => address,
                _ => zero,
            },
            |_| zero,
            |_| zero,
        );

        let output = relation.output_expression::<Fr>().evaluate(
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
    fn output_check_evaluates_like_core_formula() {
        let relation = OutputCheck::new(read_write_dimensions());

        let val_final = Fr::from_u64(7);
        let eq_io_mask = Fr::from_u64(11);
        let neg_eq_io_mask_val_io = -Fr::from_u64(13);
        let zero = Fr::from_u64(0);

        let input = relation
            .input_expression::<Fr>()
            .evaluate(|_| zero, |_| zero, |_| zero);
        let output = relation.output_expression::<Fr>().evaluate(
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
    fn ra_claim_reduction_evaluates_like_core_formula() {
        let relation = RaClaimReduction::new(trace_dimensions());

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

        let input = relation.input_expression::<Fr>().evaluate(
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
                | JoltChallengeId::SpartanShift(_) => zero,
            },
            |_| zero,
        );

        let output = relation.output_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == ram_ra_claim_reduction() => reduced,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::RamRaClaimReduction(RamRaClaimReductionChallenge::Gamma) => gamma,
                JoltChallengeId::RamReadWrite(_)
                | JoltChallengeId::RamValCheck(_)
                | JoltChallengeId::RegistersReadWrite(_)
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
                | JoltChallengeId::SpartanShift(_) => zero,
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
    fn ra_virtualization_evaluates_like_core_formula() {
        let dimensions = ra_virtualization_dimensions(3);
        let relation = RaVirtualization::new(dimensions);

        let reduced = Fr::from_u64(3);
        let committed = [Fr::from_u64(5), Fr::from_u64(7), Fr::from_u64(11)];
        let eq_cycle = Fr::from_u64(13);
        let zero = Fr::from_u64(0);

        let input = relation.input_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == ram_ra_claim_reduction() => reduced,
                _ => zero,
            },
            |_| zero,
            |_| zero,
        );

        let output = relation.output_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == committed_ram_ra(0) => committed[0],
                id if id == committed_ram_ra(1) => committed[1],
                id if id == committed_ram_ra(2) => committed[2],
                _ => zero,
            },
            |_| zero,
            |id| match *id {
                JoltPublicId::RamRaVirtualization(RamRaVirtualizationPublic::EqCycle) => eq_cycle,
                _ => zero,
            },
        );

        assert_eq!(input, reduced);
        assert_eq!(
            output,
            eq_cycle * committed[0] * committed[1] * committed[2]
        );
    }

    #[test]
    fn hamming_booleanity_evaluates_like_core_formula() {
        let relation = HammingBooleanity::new(trace_dimensions());

        let h = Fr::from_u64(7);
        let eq_cycle = Fr::from_u64(11);
        let zero = Fr::from_u64(0);

        let input = relation
            .input_expression::<Fr>()
            .evaluate(|_| zero, |_| zero, |_| zero);
        let output = relation.output_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == ram_hamming_weight() => h,
                _ => zero,
            },
            |_| zero,
            |id| match *id {
                JoltPublicId::RamHammingBooleanity(RamHammingBooleanityPublic::EqCycle) => eq_cycle,
                _ => zero,
            },
        );

        assert_eq!(input, zero);
        assert_eq!(output, eq_cycle * (h * h - h));
    }

    #[test]
    fn read_write_symbolic_matches_dependencies() {
        let relation = ReadWriteChecking::new(read_write_dimensions());

        assert_eq!(
            ReadWriteChecking::id(),
            JoltRelationId::RamReadWriteChecking
        );
        assert_eq!(
            relation.spec(),
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
        assert_eq!(relation.spec(), raf_evaluation_dimensions().sumcheck());
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
            relation.spec(),
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
        assert_eq!(relation.spec(), trace_dimensions().sumcheck(2));
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
        assert_eq!(relation.spec(), ra_virtualization_dimensions(3).sumcheck());
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
        assert_eq!(relation.spec(), trace_dimensions().sumcheck(3));
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

    #[test]
    fn ram_val_check_symbolic_matches_dependencies() {
        let relation = RamValCheck::new(RamValCheckShape {
            dimensions: trace_dimensions(),
            contributions: vec![],
        });

        assert_eq!(RamValCheck::id(), JoltRelationId::RamValCheck);
        assert_eq!(relation.spec(), trace_dimensions().sumcheck(3));
        // Full-init form (no committed contributions): only the read-write and
        // output-check openings on the input side.
        assert_eq!(
            relation.required_openings::<Fr>(),
            vec![
                ram_val(),
                ram_val_final(),
                ram_inc_val_check(),
                ram_ra_val_check(),
            ]
        );
        assert_eq!(
            relation.required_challenges::<Fr>(),
            vec![JoltChallengeId::from(RamValCheckChallenge::Gamma)]
        );
        assert_eq!(
            relation.required_publics::<Fr>(),
            vec![
                JoltPublicId::from(RamValCheckPublic::InitEval),
                JoltPublicId::from(RamValCheckPublic::LtCyclePlusGamma),
            ]
        );
    }

    /// The remodel's soundness anchor: the `Public`-symbol input expression must
    /// evaluate to the same value the pre-remodel baked-constant decomposition did
    /// (proven equal to the full-init formula in `geometry::ram`'s tests). With
    /// `InitEval = public_eval` and `InitSelector = neg_selector`, the
    /// `public·opening` term equals the old `constant·opening` term.
    #[test]
    fn ram_val_check_symbolic_evaluates_like_decomposed_init() {
        use crate::protocols::jolt::geometry::ram::val_check_advice_opening;
        use crate::protocols::jolt::JoltAdviceKind;

        let public_eval = Fr::from_u64(3);
        let untrusted_neg_selector = -Fr::from_u64(5);
        let trusted_neg_selector = -Fr::from_u64(7);

        let relation = RamValCheck::new(RamValCheckShape {
            dimensions: trace_dimensions(),
            contributions: vec![
                RamValContribution {
                    selector: RamValCheckPublic::InitSelector(JoltAdviceKind::Untrusted),
                    opening: val_check_advice_opening(JoltAdviceKind::Untrusted),
                },
                RamValContribution {
                    selector: RamValCheckPublic::InitSelector(JoltAdviceKind::Trusted),
                    opening: val_check_advice_opening(JoltAdviceKind::Trusted),
                },
            ],
        });

        let val_rw = Fr::from_u64(11);
        let val_final = Fr::from_u64(13);
        let gamma = Fr::from_u64(17);
        let untrusted_advice = Fr::from_u64(19);
        let trusted_advice = Fr::from_u64(23);
        let zero = Fr::from_u64(0);
        let init_eval = public_eval
            - untrusted_neg_selector * untrusted_advice
            - trusted_neg_selector * trusted_advice;

        let input = relation.input_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == ram_val() => val_rw,
                id if id == ram_val_final() => val_final,
                id if id == val_check_advice_opening(JoltAdviceKind::Untrusted) => untrusted_advice,
                id if id == val_check_advice_opening(JoltAdviceKind::Trusted) => trusted_advice,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::RamValCheck(RamValCheckChallenge::Gamma) => gamma,
                _ => zero,
            },
            |id| match *id {
                JoltPublicId::RamValCheck(RamValCheckPublic::InitEval) => public_eval,
                JoltPublicId::RamValCheck(RamValCheckPublic::InitSelector(
                    JoltAdviceKind::Untrusted,
                )) => untrusted_neg_selector,
                JoltPublicId::RamValCheck(RamValCheckPublic::InitSelector(
                    JoltAdviceKind::Trusted,
                )) => trusted_neg_selector,
                _ => zero,
            },
        );

        assert_eq!(
            input,
            (val_rw - init_eval) + gamma * (val_final - init_eval)
        );
    }
}
