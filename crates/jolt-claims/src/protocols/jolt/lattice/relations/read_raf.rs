//! Lattice-mode bytecode read-RAF: the base two-phase relation with a sixth
//! val stage binding the fused-inc store selector.
//!
//! `OpFlags(Store)@IncVirtualization` is consumed like the five base flag
//! stages — its claim joins the address-phase input fold and its
//! verifier-evaluated `StageValue(5)` joins the cycle-phase output fold —
//! which is only sound because `IncVirtualization` runs strictly before
//! this relation's address phase (the consumer's cycle point is baked into
//! the address-phase state at construction).

use jolt_field::{Field, RingCore};

use crate::protocols::jolt::geometry::bytecode::{
    bytecode_ra_product, bytecode_read_raf_address_phase_opening, pc_spartan_outer,
    pc_spartan_shift, stage1_claim, stage2_claim, stage3_claim, stage4_claim, stage5_claim,
    BytecodeReadRafDimensions,
};
use crate::protocols::jolt::relations::bytecode::{
    BytecodeReadRafAddressPhaseChallenges, BytecodeReadRafAddressPhaseInputClaims,
    BytecodeReadRafAddressPhaseOutputClaims, BytecodeReadRafCyclePhaseChallenges,
    BytecodeReadRafInputClaims, BytecodeReadRafOutputClaims,
};
use crate::protocols::jolt::{
    BytecodeReadRafChallenge, BytecodeReadRafPublic, JoltChallengeId, JoltDerivedId, JoltExpr,
    JoltOpeningId, JoltRelationId,
};
use crate::{challenge, derived, opening, GetValue, InputClaims, SymbolicSumcheck};

use super::inc_virtualization::fused_inc_store_opening;

/// The base address-phase inputs plus the store selector claim. Input
/// claims never cross the wire (verifier-assembled), so no serde.
#[derive(Clone, Debug)]
pub struct LatticeReadRafAddressPhaseInputClaims<C> {
    pub base: BytecodeReadRafAddressPhaseInputClaims<C>,
    pub store: C,
}

impl<F, C> InputClaims<F> for LatticeReadRafAddressPhaseInputClaims<C>
where
    F: Field,
    C: GetValue<F>,
{
    fn canonical_order(&self) -> Vec<JoltOpeningId> {
        let mut order = self.base.canonical_order();
        order.push(fused_inc_store_opening());
        order
    }

    fn resolve_input(&self, id: &JoltOpeningId) -> Option<F> {
        if *id == fused_inc_store_opening() {
            return Some(self.store.value());
        }
        self.base.resolve_input(id)
    }
}

/// The address phase with the store claim as the sixth stage: the base fold
/// with the store term at `γ⁵` and the pc/entry terms shifted to `γ⁶..γ⁸`.
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
        let gamma = challenge(BytecodeReadRafChallenge::Gamma);

        gamma.clone().pow(8)
            + stage1_claim()
            + gamma.clone() * stage2_claim()
            + gamma.clone().pow(2) * stage3_claim()
            + gamma.clone().pow(3) * stage4_claim()
            + gamma.clone().pow(4) * stage5_claim::<F>()
            + gamma.clone().pow(5) * opening(fused_inc_store_opening())
            + gamma.clone().pow(6) * opening(pc_spartan_outer())
            + gamma.pow(7) * opening(pc_spartan_shift())
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        opening(bytecode_read_raf_address_phase_opening())
    }
}

/// The cycle phase with six verifier-evaluated stage values; the RAF and
/// entry publics shift past the store stage.
pub struct LatticeReadRafCyclePhase {
    shape: BytecodeReadRafDimensions,
}

impl SymbolicSumcheck for LatticeReadRafCyclePhase {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = JoltDerivedId;
    type ChallengeId = JoltChallengeId;
    type Shape = BytecodeReadRafDimensions;
    type Challenges<F> = BytecodeReadRafCyclePhaseChallenges<F>;
    type Inputs<C> = BytecodeReadRafInputClaims<C>;
    type Outputs<C> = BytecodeReadRafOutputClaims<C>;

    fn new(shape: BytecodeReadRafDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::BytecodeReadRaf
    }

    fn rounds(&self) -> usize {
        self.shape.log_t()
    }

    fn degree(&self) -> usize {
        self.shape.num_committed_ra_polys() + 1
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        opening(bytecode_read_raf_address_phase_opening())
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = challenge(BytecodeReadRafChallenge::Gamma);
        let output_coeff = derived(BytecodeReadRafPublic::StageValue(0))
            + gamma.clone() * derived(BytecodeReadRafPublic::StageValue(1))
            + gamma.clone().pow(2) * derived(BytecodeReadRafPublic::StageValue(2))
            + gamma.clone().pow(3) * derived(BytecodeReadRafPublic::StageValue(3))
            + gamma.clone().pow(4) * derived(BytecodeReadRafPublic::StageValue(4))
            + gamma.clone().pow(5) * derived(BytecodeReadRafPublic::StageValue(5))
            + gamma.clone().pow(6) * derived(BytecodeReadRafPublic::SpartanOuterRaf)
            + gamma.clone().pow(7) * derived(BytecodeReadRafPublic::SpartanShiftRaf)
            + gamma.pow(8) * derived(BytecodeReadRafPublic::Entry);

        output_coeff * bytecode_ra_product(self.shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Fr;

    fn dimensions() -> BytecodeReadRafDimensions {
        BytecodeReadRafDimensions::new(5, 10, 2)
    }

    #[test]
    fn lattice_address_phase_appends_the_store_stage() {
        let relation = LatticeReadRafAddressPhase::new(dimensions());
        assert_eq!(
            LatticeReadRafAddressPhase::id(),
            JoltRelationId::BytecodeReadRaf
        );
        assert_eq!(relation.rounds(), dimensions().log_k());
        assert_eq!(relation.degree(), dimensions().num_committed_ra_polys() + 1);
        let openings = relation.input_expression::<Fr>().required_openings();
        assert!(openings.contains(&fused_inc_store_opening()));
        assert!(openings.contains(&pc_spartan_outer()));
        assert_eq!(
            relation.output_expression::<Fr>().required_openings(),
            vec![bytecode_read_raf_address_phase_opening()]
        );
    }

    #[test]
    fn lattice_cycle_phase_extends_to_six_stage_values() {
        let relation = LatticeReadRafCyclePhase::new(dimensions());
        assert_eq!(relation.rounds(), dimensions().log_t());
        let deriveds = relation.required_deriveds::<Fr>();
        for stage in 0..6 {
            assert!(
                deriveds.contains(&JoltDerivedId::from(BytecodeReadRafPublic::StageValue(
                    stage
                )))
            );
        }
        assert!(deriveds.contains(&JoltDerivedId::from(BytecodeReadRafPublic::SpartanOuterRaf)));
        assert!(deriveds.contains(&JoltDerivedId::from(BytecodeReadRafPublic::Entry)));
    }
}
