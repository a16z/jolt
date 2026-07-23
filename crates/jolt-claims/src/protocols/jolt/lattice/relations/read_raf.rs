//! Lattice-mode bytecode read-RAF: the base two-phase relation extended with
//! four fused-inc consumer val stages.
//!
//! The four reduced `Inc` claims (`RamInc` from RAM read-write / val-check,
//! `RdInc` from register read-write / val-evaluation) join the address-phase
//! input fold at `γ^5..8`, replacing the base `IncClaimReduction` member and
//! the former standalone `IncVirtualization` phase: since a cycle's bytecode
//! row is one-hot, `Store(j) = Σ_k val_store(k)·ra(k,j)` substitutes the
//! store selector directly into the fused-inc identities
//! (`FusedInc·Store = RamInc`, `FusedInc·(1−Store) = RdInc`), so each inc
//! claim is exactly a read-raf-shaped stage — a bytecode val column
//! (`store`/`¬store`), the consuming relation's cycle point, and the shared
//! RA product — with one extra `FusedInc` cycle factor (degree +1).
//!
//! The cycle phase therefore produces the `FusedInc` opening at the shared
//! stage-6b cycle point (consumed by the stage-7 hamming-weight decode leg),
//! and no store-selector opening exists anywhere.
//!
//! The per-cycle store/rd disjointness comes from `jolt-program`'s memory
//! expansion: ISA S-type stores carry no `rd`, and every read-modify-write
//! instruction is lowered into a virtual sequence whose RAM-writing step is a
//! plain store, with the `rd` write on a separate cycle. The offline store/rd
//! disjointness check on the public bytecode re-verifies this per row at
//! preprocessing.

use jolt_field::{Field, RingCore};
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::bytecode::{
    bytecode_read_raf_address_phase_opening, read_raf_address_input_fold,
    read_raf_cycle_output_committed_lattice, read_raf_cycle_output_lattice,
    BytecodeReadRafDimensions, BYTECODE_STAGE_GAMMA_COUNTS, LATTICE_FUSED_INC_STAGES,
};
use crate::protocols::jolt::geometry::ram::{ram_inc, ram_inc_val_check};
use crate::protocols::jolt::geometry::registers::{rd_inc_read_write, rd_inc_val_evaluation};
use crate::protocols::jolt::relations::bytecode::{
    BytecodeReadRafAddressPhaseChallenges, BytecodeReadRafAddressPhaseInputClaims,
    BytecodeReadRafAddressPhaseOutputClaims, BytecodeReadRafCyclePhaseChallenges,
    BytecodeReadRafCyclePhaseCommittedChallenges, BytecodeReadRafInputClaims,
};
use crate::protocols::jolt::relations::claim_reductions::increments::IncClaimReductionInputClaims;
use crate::protocols::jolt::{
    JoltChallengeId, JoltDerivedId, JoltExpr, JoltOpeningId, JoltRelationId,
};
use crate::{opening, InputClaims, OutputClaims, SymbolicSumcheck};

/// Total lattice read-raf val stages: the five base flag stages plus the four
/// fused-inc consumer stages.
pub const LATTICE_READ_RAF_STAGES: usize =
    BYTECODE_STAGE_GAMMA_COUNTS.len() + LATTICE_FUSED_INC_STAGES;

/// The base address-phase inputs plus the four consumed inc claims (the same
/// struct the base `IncClaimReduction` consumes — the producing relations are
/// identical). Input claims never cross the wire (verifier-assembled), so no
/// serde.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LatticeReadRafAddressPhaseInputClaims<C> {
    pub base: BytecodeReadRafAddressPhaseInputClaims<C>,
    pub inc: IncClaimReductionInputClaims<C>,
}

impl<F: Field> InputClaims<F> for LatticeReadRafAddressPhaseInputClaims<F> {
    fn canonical_order(&self) -> Vec<JoltOpeningId> {
        let mut order = self.base.canonical_order();
        order.extend(InputClaims::<F>::canonical_order(&self.inc));
        order
    }

    fn resolve_input(&self, id: &JoltOpeningId) -> Option<F> {
        self.inc
            .resolve_input(id)
            .or_else(|| self.base.resolve_input(id))
    }
}

/// The four consumed inc claims in stage order (`γ^5..8`).
fn fused_inc_stage_claims<F: RingCore>() -> Vec<JoltExpr<F>> {
    vec![
        opening(ram_inc()),
        opening(ram_inc_val_check()),
        opening(rd_inc_read_write()),
        opening(rd_inc_val_evaluation()),
    ]
}

/// The address phase with the four inc claims as stages `γ^5..8` and the
/// pc/shift/entry terms shifted to `γ^9..11`.
#[derive(Clone)]
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
        read_raf_address_input_fold(fused_inc_stage_claims())
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        opening(bytecode_read_raf_address_phase_opening())
    }
}

/// The lattice cycle-phase produced openings: the committed `BytecodeRa`
/// chunks plus the `FusedInc` stream at the bound cycle point.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(BytecodeReadRaf)]
pub struct LatticeBytecodeReadRafOutputClaims<C> {
    #[opening(committed = BytecodeRa)]
    pub bytecode_ra: Vec<C>,
    #[opening(FusedInc)]
    pub fused_inc: C,
}

/// Lattice full-program cycle phase: nine verifier-evaluated stage values, the
/// last four carrying the `FusedInc` opening as a cycle factor.
#[derive(Clone)]
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
    type Outputs<C> = LatticeBytecodeReadRafOutputClaims<C>;

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
        self.shape.num_committed_ra_polys() + 2
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        opening(bytecode_read_raf_address_phase_opening())
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        read_raf_cycle_output_lattice(self.shape)
    }
}

/// Lattice committed-program cycle phase: the base staged vals plus the four
/// fused stages resolving through the staged *store* val and its complement.
#[derive(Clone)]
pub struct LatticeReadRafCyclePhaseCommitted {
    shape: BytecodeReadRafDimensions,
}

impl SymbolicSumcheck for LatticeReadRafCyclePhaseCommitted {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = JoltDerivedId;
    type ChallengeId = JoltChallengeId;
    type Shape = BytecodeReadRafDimensions;
    type Challenges<F> = BytecodeReadRafCyclePhaseCommittedChallenges<F>;
    type Inputs<C> = BytecodeReadRafInputClaims<C>;
    type Outputs<C> = LatticeBytecodeReadRafOutputClaims<C>;

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
        self.shape.num_committed_ra_polys() + 2
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        opening(bytecode_read_raf_address_phase_opening())
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        read_raf_cycle_output_committed_lattice(self.shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::jolt::geometry::bytecode::{
        bytecode_ra, fused_inc_read_raf_opening, pc_spartan_outer,
    };
    use crate::protocols::jolt::geometry::claim_reductions::bytecode::bytecode_val_stage_opening;
    use crate::protocols::jolt::geometry::spartan::pc_shift;
    use crate::protocols::jolt::BytecodeReadRafPublic;
    use crate::SymbolicSumcheck;
    use jolt_field::{Fr, FromPrimitiveInt};

    fn dimensions() -> BytecodeReadRafDimensions {
        BytecodeReadRafDimensions::new(5, 10, 2)
    }

    fn pow(base: Fr, exponent: usize) -> Fr {
        (0..exponent).fold(Fr::from_u64(1), |acc, _| acc * base)
    }

    /// With the five staged claims zeroed, the address-phase input collapses
    /// to the inc/pc/shift/entry placements:
    /// `γ⁵·ram_rw + γ⁶·ram_val + γ⁷·rd_rw + γ⁸·rd_val + γ⁹·pc + γ¹⁰·shift + γ¹¹`.
    #[test]
    fn lattice_address_phase_folds_the_inc_stages_at_gamma_5_to_8() {
        let relation = LatticeReadRafAddressPhase::new(dimensions());
        assert_eq!(
            LatticeReadRafAddressPhase::id(),
            JoltRelationId::BytecodeReadRaf
        );
        assert_eq!(relation.rounds(), dimensions().log_k());
        assert_eq!(relation.degree(), dimensions().num_committed_ra_polys() + 1);

        let gamma = Fr::from_u64(3);
        let ram_rw = Fr::from_u64(5);
        let ram_val = Fr::from_u64(23);
        let rd_rw = Fr::from_u64(29);
        let rd_val = Fr::from_u64(31);
        let pc_outer = Fr::from_u64(7);
        let pc_shifted = Fr::from_u64(11);
        let intermediate = Fr::from_u64(13);
        let zero = Fr::from_u64(0);

        let input = relation.input_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == ram_inc() => ram_rw,
                id if id == ram_inc_val_check() => ram_val,
                id if id == rd_inc_read_write() => rd_rw,
                id if id == rd_inc_val_evaluation() => rd_val,
                id if id == pc_spartan_outer() => pc_outer,
                id if id == pc_shift() => pc_shifted,
                _ => zero,
            },
            |_| gamma,
            |_| zero,
        );
        assert_eq!(
            input,
            pow(gamma, 5) * ram_rw
                + pow(gamma, 6) * ram_val
                + pow(gamma, 7) * rd_rw
                + pow(gamma, 8) * rd_val
                + pow(gamma, 9) * pc_outer
                + pow(gamma, 10) * pc_shifted
                + pow(gamma, 11)
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

    /// Nine stage values fold at `γ^0..8` — the last four against the fused
    /// opening — with RAF and entry publics at `γ^9..11`, all against the
    /// committed RA product.
    #[test]
    fn lattice_cycle_phase_carries_the_fused_factor_on_stages_5_to_8() {
        let relation = LatticeReadRafCyclePhase::new(dimensions());
        assert_eq!(relation.rounds(), dimensions().log_t());
        assert_eq!(relation.degree(), dimensions().num_committed_ra_polys() + 2);

        let gamma = Fr::from_u64(3);
        let stage_values: Vec<Fr> = (0..LATTICE_READ_RAF_STAGES)
            .map(|stage| Fr::from_u64(5 + stage as u64))
            .collect();
        let outer_raf = Fr::from_u64(17);
        let shift_raf = Fr::from_u64(19);
        let entry = Fr::from_u64(23);
        let ra = [Fr::from_u64(29), Fr::from_u64(31)];
        let fused = Fr::from_u64(37);
        let zero = Fr::from_u64(0);

        let output = relation.output_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == bytecode_ra(0) => ra[0],
                id if id == bytecode_ra(1) => ra[1],
                id if id == fused_inc_read_raf_opening() => fused,
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
        for (stage, value) in stage_values.iter().take(5).enumerate() {
            coefficient += pow(gamma, stage) * *value;
        }
        for (stage, value) in stage_values.iter().enumerate().skip(5) {
            coefficient += pow(gamma, stage) * *value * fused;
        }
        coefficient +=
            pow(gamma, 9) * outer_raf + pow(gamma, 10) * shift_raf + pow(gamma, 11) * entry;
        assert_eq!(output, coefficient * ra[0] * ra[1]);
    }

    /// The committed cycle output resolves the four fused stages through the
    /// staged store val (index 5) and its complement, against the fused
    /// opening; the base five stage their own vals; RAF and entry at `γ^9..11`.
    #[test]
    fn lattice_committed_cycle_phase_reuses_the_staged_store_val() {
        let relation = LatticeReadRafCyclePhaseCommitted::new(dimensions());
        assert_eq!(relation.rounds(), dimensions().log_t());
        assert_eq!(relation.degree(), dimensions().num_committed_ra_polys() + 2);

        let gamma = Fr::from_u64(3);
        let val_stages: Vec<Fr> = (0..6).map(|stage| Fr::from_u64(5 + stage as u64)).collect();
        let cycle_eqs: Vec<Fr> = (0..LATTICE_READ_RAF_STAGES)
            .map(|stage| Fr::from_u64(37 + stage as u64))
            .collect();
        let outer_raf = Fr::from_u64(17);
        let shift_raf = Fr::from_u64(19);
        let entry = Fr::from_u64(23);
        let ra = [Fr::from_u64(29), Fr::from_u64(31)];
        let fused = Fr::from_u64(41);
        let zero = Fr::from_u64(0);
        let one = Fr::from_u64(1);

        let output = relation.output_expression::<Fr>().evaluate(
            |id| {
                (0..6)
                    .find(|&stage| *id == bytecode_val_stage_opening(stage))
                    .map(|stage| val_stages[stage])
                    .or((*id == bytecode_ra(0)).then_some(ra[0]))
                    .or((*id == bytecode_ra(1)).then_some(ra[1]))
                    .or((*id == fused_inc_read_raf_opening()).then_some(fused))
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
        let store = val_stages[5];
        let mut expected = zero;
        for stage in 0..5 {
            expected += pow(gamma, stage) * cycle_eqs[stage] * ra_product * val_stages[stage];
        }
        expected += (pow(gamma, 5) * cycle_eqs[5] + pow(gamma, 6) * cycle_eqs[6])
            * ra_product
            * fused
            * store;
        expected += (pow(gamma, 7) * cycle_eqs[7] + pow(gamma, 8) * cycle_eqs[8])
            * ra_product
            * fused
            * (one - store);
        expected +=
            (pow(gamma, 9) * outer_raf + pow(gamma, 10) * shift_raf + pow(gamma, 11) * entry)
                * ra_product;
        assert_eq!(output, expected);
    }

    #[test]
    fn composite_input_claims_resolve_base_and_inc() {
        let claims = LatticeReadRafAddressPhaseInputClaims::<Fr> {
            base: BytecodeReadRafAddressPhaseInputClaims::default(),
            inc: IncClaimReductionInputClaims {
                ram_inc_read_write: Fr::from_u64(7),
                ram_inc_val_check: Fr::from_u64(11),
                rd_inc_read_write: Fr::from_u64(13),
                rd_inc_val_evaluation: Fr::from_u64(17),
            },
        };
        let order = InputClaims::<Fr>::canonical_order(&claims);
        assert_eq!(order.last(), Some(&rd_inc_val_evaluation()));
        assert_eq!(
            InputClaims::<Fr>::resolve_input(&claims, &ram_inc()),
            Some(Fr::from_u64(7))
        );
        assert_eq!(
            order.len(),
            InputClaims::<Fr>::canonical_order(&claims.base).len() + 4
        );
    }
}
