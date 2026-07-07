//! Fused increment virtualization: the lattice-mode replacement for the base
//! `IncClaimReduction` relation.

use jolt_field::RingCore;
use jolt_riscv::CircuitFlags;
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::claim_reductions::increments::inc_consumers_input;
use crate::protocols::jolt::relations::claim_reductions::increments::IncClaimReductionInputClaims;
use crate::protocols::jolt::{
    IncVirtualizationChallenge, IncVirtualizationPublic, JoltExpr, JoltOpeningId, JoltRelationId,
    JoltVirtualPolynomial, TraceDimensions,
};
use crate::{challenge, derived, opening, OutputClaims, SumcheckChallenges, SymbolicSumcheck};

/// The fused increment stream and its destination selector, produced at the
/// bound cycle point. A cycle increments RAM (`store = 1`) or a register
/// (`store = 0`), never both, so one committed one-hot decomposition serves
/// both consumers — this halves the packed inc polynomials.
///
/// The per-cycle disjointness comes from `jolt-program`'s memory expansion:
/// ISA S-type stores carry no `rd`, and every read-modify-write instruction
/// (the RV64A atomics, `crates/jolt-program/src/expand/memory/`) is lowered
/// into a virtual sequence whose RAM-writing step is a plain store, with the
/// `rd` write on a separate cycle. The offline store/rd disjointness check on
/// the public bytecode re-verifies this per row at preprocessing.
///
/// The selector is the existing `OpFlags(Store)` virtual polynomial, so its
/// opening is bound to the actual bytecode Store flag by the same read-raf
/// val-stage machinery that binds every other flag consumer — no
/// dedicated store-binding relation is needed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(IncVirtualization)]
pub struct IncVirtualizationOutputClaims<C> {
    #[opening(FusedInc)]
    pub fused_inc: C,
    #[opening(OpFlags(CircuitFlags::Store))]
    pub store: C,
}

#[derive(Clone, Copy, Debug, SumcheckChallenges)]
pub struct IncVirtualizationChallenges<F> {
    #[challenge(IncVirtualizationChallenge::Gamma)]
    pub gamma: F,
}

/// Batches the four inc consumer claims by `gamma` and virtualizes them into
/// the fused stream: `FusedInc · (ram_coeff · store + γ² · rd_coeff ·
/// (1 − store))`, with `ram_coeff`/`rd_coeff` the γ-paired eq deriveds of the
/// consuming relations' cycle points. `RdInc`/`RamInc` are never PCS-opened in
/// lattice mode; this relation is where their claims leave the base PIOP.
pub struct IncVirtualization {
    shape: TraceDimensions,
}

impl SymbolicSumcheck for IncVirtualization {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = crate::protocols::jolt::JoltDerivedId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = TraceDimensions;
    type Challenges<F> = IncVirtualizationChallenges<F>;
    /// The same consumed-claim struct as the base `IncClaimReduction`: the
    /// lattice relation replaces it and must consume exactly the same four
    /// claims (input-claim ids carry the producing relations, not the
    /// consumer, so the struct is shared verbatim).
    type Inputs<C> = IncClaimReductionInputClaims<C>;
    type Outputs<C> = IncVirtualizationOutputClaims<C>;

    fn new(shape: TraceDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::IncVirtualization
    }

    fn rounds(&self) -> usize {
        self.shape.log_t()
    }

    fn degree(&self) -> usize {
        3
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        inc_consumers_input(challenge(IncVirtualizationChallenge::Gamma))
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = challenge(IncVirtualizationChallenge::Gamma);

        let ram_coeff = derived(IncVirtualizationPublic::EqRamReadWrite)
            + gamma.clone() * derived(IncVirtualizationPublic::EqRamValCheck);
        let rd_coeff = derived(IncVirtualizationPublic::EqRegistersReadWrite)
            + gamma.clone() * derived(IncVirtualizationPublic::EqRegistersValEvaluation);
        let store = opening(fused_inc_store_opening());

        opening(fused_inc_opening())
            * (ram_coeff * store.clone() + gamma.pow(2) * rd_coeff * (JoltExpr::one() - store))
    }
}

pub fn fused_inc_opening() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::FusedInc,
        JoltRelationId::IncVirtualization,
    )
}

pub fn fused_inc_store_opening() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::OpFlags(CircuitFlags::Store),
        JoltRelationId::IncVirtualization,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::jolt::geometry::claim_reductions::increments::{
        ram_inc_read_write, ram_inc_val_check, rd_inc_read_write, rd_inc_val_evaluation,
    };
    use crate::protocols::jolt::{JoltChallengeId, JoltDerivedId};
    use jolt_field::{Fr, FromPrimitiveInt};

    #[test]
    fn inc_virtualization_evaluates_like_core_formula() {
        let relation = IncVirtualization::new(TraceDimensions::new(5));

        let ram_rw = Fr::from_u64(3);
        let ram_val = Fr::from_u64(5);
        let rd_rw = Fr::from_u64(7);
        let rd_val = Fr::from_u64(11);
        let fused_inc = Fr::from_u64(13);
        let store = Fr::from_u64(17);
        let eq_ram_rw = Fr::from_u64(19);
        let eq_ram_val = Fr::from_u64(23);
        let eq_rd_rw = Fr::from_u64(29);
        let eq_rd_val = Fr::from_u64(31);
        let gamma = Fr::from_u64(37);
        let zero = Fr::from_u64(0);

        let input = relation.input_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == ram_inc_read_write() => ram_rw,
                id if id == ram_inc_val_check() => ram_val,
                id if id == rd_inc_read_write() => rd_rw,
                id if id == rd_inc_val_evaluation() => rd_val,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::IncVirtualization(IncVirtualizationChallenge::Gamma) => gamma,
                _ => zero,
            },
            |_| zero,
        );
        let gamma_2 = gamma * gamma;
        assert_eq!(
            input,
            ram_rw + gamma * ram_val + gamma_2 * rd_rw + gamma_2 * gamma * rd_val
        );

        let output = relation.output_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == fused_inc_opening() => fused_inc,
                id if id == fused_inc_store_opening() => store,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::IncVirtualization(IncVirtualizationChallenge::Gamma) => gamma,
                _ => zero,
            },
            |id| match *id {
                JoltDerivedId::IncVirtualization(IncVirtualizationPublic::EqRamReadWrite) => {
                    eq_ram_rw
                }
                JoltDerivedId::IncVirtualization(IncVirtualizationPublic::EqRamValCheck) => {
                    eq_ram_val
                }
                JoltDerivedId::IncVirtualization(IncVirtualizationPublic::EqRegistersReadWrite) => {
                    eq_rd_rw
                }
                JoltDerivedId::IncVirtualization(
                    IncVirtualizationPublic::EqRegistersValEvaluation,
                ) => eq_rd_val,
                _ => zero,
            },
        );
        assert_eq!(
            output,
            fused_inc
                * ((eq_ram_rw + gamma * eq_ram_val) * store
                    + gamma_2 * (eq_rd_rw + gamma * eq_rd_val) * (Fr::from_u64(1) - store))
        );
    }

    #[test]
    fn inc_virtualization_exposes_expected_dependencies() {
        let relation = IncVirtualization::new(TraceDimensions::new(5));

        assert_eq!(IncVirtualization::id(), JoltRelationId::IncVirtualization);
        assert_eq!(relation.rounds(), 5);
        assert_eq!(relation.degree(), 3);
        assert_eq!(
            relation.input_expression::<Fr>().required_openings(),
            vec![
                ram_inc_read_write(),
                ram_inc_val_check(),
                rd_inc_read_write(),
                rd_inc_val_evaluation(),
            ]
        );
        assert_eq!(
            relation.output_expression::<Fr>().required_openings(),
            vec![fused_inc_opening(), fused_inc_store_opening()]
        );
        assert_eq!(
            relation.required_challenges::<Fr>(),
            vec![JoltChallengeId::from(IncVirtualizationChallenge::Gamma)]
        );
        assert_eq!(
            relation.required_deriveds::<Fr>(),
            vec![
                JoltDerivedId::from(IncVirtualizationPublic::EqRamReadWrite),
                JoltDerivedId::from(IncVirtualizationPublic::EqRamValCheck),
                JoltDerivedId::from(IncVirtualizationPublic::EqRegistersReadWrite),
                JoltDerivedId::from(IncVirtualizationPublic::EqRegistersValEvaluation),
            ]
        );
    }
}
