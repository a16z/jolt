//! Lattice-mode bytecode read-RAF: the base two-phase relation with a sixth
//! val stage binding the fused-inc store selector.
//!
//! `OpFlags(Store)@IncVirtualization` is consumed like the five base flag
//! stages — its claim joins the address-phase input fold and its
//! verifier-evaluated `StageValue(5)` joins the cycle-phase output fold —
//! which is only sound because `IncVirtualization` runs strictly before
//! this relation's address phase (the consumer's cycle point is baked into
//! the address-phase state at construction).
//!
//! Only the address phase needs a lattice variant (the extra consumed store
//! claim); the cycle phases are the base `ReadRafCyclePhase` /
//! `ReadRafCyclePhaseCommitted` relations at six staged vals.

use jolt_field::{Field, RingCore};

use crate::protocols::jolt::geometry::bytecode::{
    bytecode_read_raf_address_phase_opening, read_raf_address_input_fold, BytecodeReadRafDimensions,
};
use crate::protocols::jolt::relations::bytecode::{
    BytecodeReadRafAddressPhaseChallenges, BytecodeReadRafAddressPhaseInputClaims,
    BytecodeReadRafAddressPhaseOutputClaims,
};
use crate::protocols::jolt::{
    JoltChallengeId, JoltDerivedId, JoltExpr, JoltOpeningId, JoltRelationId,
};
use crate::{opening, InputClaims, SymbolicSumcheck};

use super::inc_virtualization::fused_inc_store_opening;

/// The base address-phase inputs plus the store selector claim. Input
/// claims never cross the wire (verifier-assembled), so no serde.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LatticeReadRafAddressPhaseInputClaims<C> {
    pub base: BytecodeReadRafAddressPhaseInputClaims<C>,
    pub store: C,
}

impl<F: Field> InputClaims<F> for LatticeReadRafAddressPhaseInputClaims<F> {
    fn canonical_order(&self) -> Vec<JoltOpeningId> {
        let mut order = self.base.canonical_order();
        order.push(fused_inc_store_opening());
        order
    }

    fn resolve_input(&self, id: &JoltOpeningId) -> Option<F> {
        if *id == fused_inc_store_opening() {
            return Some(self.store);
        }
        self.base.resolve_input(id)
    }
}

/// The address phase with the store claim as the sixth stage: the base fold
/// with the store term at `γ⁵` and the pc/shift/entry terms shifted to
/// `γ⁶..γ⁸`.
pub struct LatticeReadRafAddressPhase {
    shape: BytecodeReadRafDimensions,
}

impl SymbolicSumcheck for LatticeReadRafAddressPhase {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = JoltDerivedId;
    type ChallengeId = JoltChallengeId;
    type Shape = BytecodeReadRafDimensions;
    type Challenges<F> = BytecodeReadRafAddressPhaseChallenges<F>;
    type Inputs<C> = LatticeReadRafAddressPhaseInputClaims<C>;
    type Outputs<C> = BytecodeReadRafAddressPhaseOutputClaims<C>;

    fn new(shape: BytecodeReadRafDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::BytecodeReadRaf
    }

    fn rounds(&self) -> usize {
        self.shape.log_k()
    }

    fn degree(&self) -> usize {
        self.shape.num_committed_ra_polys() + 1
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        read_raf_address_input_fold(Some(opening(fused_inc_store_opening())))
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        opening(bytecode_read_raf_address_phase_opening())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::jolt::geometry::bytecode::{bytecode_ra, pc_spartan_outer};
    use crate::protocols::jolt::geometry::claim_reductions::bytecode::bytecode_val_stage_opening;
    use crate::protocols::jolt::geometry::spartan::pc_shift;
    use crate::protocols::jolt::relations::bytecode::{
        ReadRafCyclePhase, ReadRafCyclePhaseCommitted,
    };
    use crate::protocols::jolt::BytecodeReadRafPublic;
    use crate::SymbolicSumcheck;
    use jolt_field::{Fr, FromPrimitiveInt};

    use super::super::super::LATTICE_BYTECODE_VAL_STAGES;

    fn dimensions() -> BytecodeReadRafDimensions {
        BytecodeReadRafDimensions::new(5, 10, 2)
    }

    fn pow(base: Fr, exponent: usize) -> Fr {
        (0..exponent).fold(Fr::from_u64(1), |acc, _| acc * base)
    }

    /// With the five staged claims zeroed, the address-phase input collapses
    /// to the store/pc/shift/entry placements:
    /// `γ⁵·store + γ⁶·pc + γ⁷·shift + γ⁸`.
    #[test]
    fn lattice_address_phase_folds_the_store_stage_at_gamma_5() {
        let relation = LatticeReadRafAddressPhase::new(dimensions());
        assert_eq!(
            LatticeReadRafAddressPhase::id(),
            JoltRelationId::BytecodeReadRaf
        );
        assert_eq!(relation.rounds(), dimensions().log_k());
        assert_eq!(relation.degree(), dimensions().num_committed_ra_polys() + 1);

        let gamma = Fr::from_u64(3);
        let store = Fr::from_u64(5);
        let pc_outer = Fr::from_u64(7);
        let pc_shifted = Fr::from_u64(11);
        let intermediate = Fr::from_u64(13);
        let zero = Fr::from_u64(0);

        let input = relation.input_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == fused_inc_store_opening() => store,
                id if id == pc_spartan_outer() => pc_outer,
                id if id == pc_shift() => pc_shifted,
                _ => zero,
            },
            |_| gamma,
            |_| zero,
        );
        assert_eq!(
            input,
            pow(gamma, 5) * store
                + pow(gamma, 6) * pc_outer
                + pow(gamma, 7) * pc_shifted
                + pow(gamma, 8)
        );

        let output = relation.output_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == bytecode_read_raf_address_phase_opening() => intermediate,
                _ => zero,
            },
            |_| zero,
            |_| zero,
        );
        assert_eq!(output, intermediate);
    }

    /// Six verifier-evaluated stage values fold at `γ^0..5`, the RAF and
    /// entry publics shift to `γ^6..8`, all against the committed RA product.
    #[test]
    fn lattice_cycle_phase_shifts_raf_past_six_stage_values() {
        let relation = ReadRafCyclePhase::new((dimensions(), LATTICE_BYTECODE_VAL_STAGES));
        assert_eq!(relation.rounds(), dimensions().log_t());

        let gamma = Fr::from_u64(3);
        let stage_values: Vec<Fr> = (0..LATTICE_BYTECODE_VAL_STAGES)
            .map(|stage| Fr::from_u64(5 + stage as u64))
            .collect();
        let outer_raf = Fr::from_u64(17);
        let shift_raf = Fr::from_u64(19);
        let entry = Fr::from_u64(23);
        let ra = [Fr::from_u64(29), Fr::from_u64(31)];
        let zero = Fr::from_u64(0);

        let output = relation.output_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == bytecode_ra(0) => ra[0],
                id if id == bytecode_ra(1) => ra[1],
                _ => zero,
            },
            |_| gamma,
            |id| match *id {
                JoltDerivedId::BytecodeReadRaf(BytecodeReadRafPublic::StageValue(stage)) => {
                    stage_values[stage]
                }
                JoltDerivedId::BytecodeReadRaf(BytecodeReadRafPublic::SpartanOuterRaf) => outer_raf,
                JoltDerivedId::BytecodeReadRaf(BytecodeReadRafPublic::SpartanShiftRaf) => shift_raf,
                JoltDerivedId::BytecodeReadRaf(BytecodeReadRafPublic::Entry) => entry,
                _ => zero,
            },
        );
        let mut coefficient = zero;
        for (stage, value) in stage_values.iter().enumerate() {
            coefficient += pow(gamma, stage) * *value;
        }
        coefficient +=
            pow(gamma, 6) * outer_raf + pow(gamma, 7) * shift_raf + pow(gamma, 8) * entry;
        assert_eq!(output, coefficient * ra[0] * ra[1]);
    }

    /// The committed cycle output stages six `BytecodeValStage` openings, each
    /// weighted by its cycle-eq public at `γ^0..5`; RAF and entry at `γ^6..8`.
    #[test]
    fn lattice_committed_cycle_phase_stages_six_vals() {
        let relation = ReadRafCyclePhaseCommitted::new((dimensions(), LATTICE_BYTECODE_VAL_STAGES));
        assert_eq!(relation.rounds(), dimensions().log_t());

        let gamma = Fr::from_u64(3);
        let val_stages: Vec<Fr> = (0..LATTICE_BYTECODE_VAL_STAGES)
            .map(|stage| Fr::from_u64(5 + stage as u64))
            .collect();
        let cycle_eqs: Vec<Fr> = (0..LATTICE_BYTECODE_VAL_STAGES)
            .map(|stage| Fr::from_u64(37 + stage as u64))
            .collect();
        let outer_raf = Fr::from_u64(17);
        let shift_raf = Fr::from_u64(19);
        let entry = Fr::from_u64(23);
        let ra = [Fr::from_u64(29), Fr::from_u64(31)];
        let zero = Fr::from_u64(0);

        let output = relation.output_expression::<Fr>().evaluate(
            |id| {
                (0..LATTICE_BYTECODE_VAL_STAGES)
                    .find(|&stage| *id == bytecode_val_stage_opening(stage))
                    .map(|stage| val_stages[stage])
                    .or((*id == bytecode_ra(0)).then_some(ra[0]))
                    .or((*id == bytecode_ra(1)).then_some(ra[1]))
                    .unwrap_or(zero)
            },
            |_| gamma,
            |id| match *id {
                JoltDerivedId::BytecodeReadRaf(BytecodeReadRafPublic::StageCycleEq(stage)) => {
                    cycle_eqs[stage]
                }
                JoltDerivedId::BytecodeReadRaf(BytecodeReadRafPublic::SpartanOuterRaf) => outer_raf,
                JoltDerivedId::BytecodeReadRaf(BytecodeReadRafPublic::SpartanShiftRaf) => shift_raf,
                JoltDerivedId::BytecodeReadRaf(BytecodeReadRafPublic::Entry) => entry,
                _ => zero,
            },
        );
        let ra_product = ra[0] * ra[1];
        let mut expected = zero;
        for stage in 0..LATTICE_BYTECODE_VAL_STAGES {
            expected += pow(gamma, stage) * cycle_eqs[stage] * ra_product * val_stages[stage];
        }
        expected += (pow(gamma, 6) * outer_raf + pow(gamma, 7) * shift_raf + pow(gamma, 8) * entry)
            * ra_product;
        assert_eq!(output, expected);
    }

    #[test]
    fn composite_input_claims_resolve_base_and_store() {
        let claims = LatticeReadRafAddressPhaseInputClaims::<Fr> {
            base: BytecodeReadRafAddressPhaseInputClaims::default(),
            store: Fr::from_u64(7),
        };
        let order = InputClaims::<Fr>::canonical_order(&claims);
        assert_eq!(order.last(), Some(&fused_inc_store_opening()));
        assert_eq!(
            InputClaims::<Fr>::resolve_input(&claims, &fused_inc_store_opening()),
            Some(Fr::from_u64(7))
        );
        assert_eq!(
            order.len(),
            InputClaims::<Fr>::canonical_order(&claims.base).len() + 1
        );
    }
}
