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

use jolt_field::{JoltField, Ring};
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

impl<F: JoltField> InputClaims<F> for LatticeReadRafAddressPhaseInputClaims<F> {
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
fn fused_inc_stage_claims<F: Ring>() -> Vec<JoltExpr<F>> {
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

    fn input_expression<F: Ring>(&self) -> JoltExpr<F> {
        read_raf_address_input_fold(fused_inc_stage_claims())
    }

    fn output_expression<F: Ring>(&self) -> JoltExpr<F> {
        opening(bytecode_read_raf_address_phase_opening())
    }
}

/// The lattice cycle-phase produced openings: the committed `BytecodeRa`
/// chunks plus the `FusedInc` stream at the bound cycle point.
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
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

    fn input_expression<F: Ring>(&self) -> JoltExpr<F> {
        opening(bytecode_read_raf_address_phase_opening())
    }

    fn output_expression<F: Ring>(&self) -> JoltExpr<F> {
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

    fn input_expression<F: Ring>(&self) -> JoltExpr<F> {
        opening(bytecode_read_raf_address_phase_opening())
    }

    fn output_expression<F: Ring>(&self) -> JoltExpr<F> {
        read_raf_cycle_output_committed_lattice(self.shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Fr, Ring};

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
