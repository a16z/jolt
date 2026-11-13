use jolt_field::Ring;
use jolt_riscv::CircuitFlags;
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::{
    JoltCommittedPolynomial, JoltExpr, JoltRelationId, JoltVirtualPolynomial,
    RamReadWriteChallenge, RamReadWritePublic, ReadWriteDimensions, UnbatchedClaim,
    UnbatchedClaimExpr, UnbatchedRelation,
};
use crate::SymbolicSumcheck;
use crate::{InputClaims, OutputClaims, SumcheckChallenges};

/// Produced RAM read-write openings (`val`, `ra`, committed `inc`), all sharing
/// the single read-write opening point. Generic over the opening cell (`F` for the
/// serialized wire value, `Vec<F>` for the derived opening point).
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(RamReadWriteChecking)]
pub struct RamReadWriteOutputClaims<C> {
    #[opening(RamVal)]
    pub val: C,
    #[opening(RamRa)]
    pub ra: C,
    #[opening(committed = RamInc)]
    pub inc: C,
}

/// Consumed RAM values and Load/Store flags from stage 1's outer sumcheck, reduced
/// by the read-write checking sumcheck. The relation reads only these values (its
/// output points come from its own sumcheck point and `product_tau_low`), so the
/// input points are left empty. Generic over the cell.
#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
pub struct RamReadWriteInputClaims<C> {
    #[opening(RamReadValue, from = SpartanOuter)]
    pub ram_read_value: C,
    #[opening(RamWriteValue, from = SpartanOuter)]
    pub ram_write_value: C,
    #[opening(OpFlags(CircuitFlags::Load), from = SpartanOuter)]
    pub load: C,
    #[opening(OpFlags(CircuitFlags::Store), from = SpartanOuter)]
    pub store: C,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, SumcheckChallenges)]
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
pub struct RamReadWriteChallenges<F> {
    #[challenge(RamReadWriteChallenge::Gamma)]
    pub gamma: F,
}

/// RAM read-write checking batches value consistency, code preservation and
/// address selection. Its public weights are cycle equality and that equality
/// times the code-word mask; all four identities share `ra`, `val` and `inc`.
#[derive(Clone)]
pub struct ReadWriteChecking {
    shape: ReadWriteDimensions,
}

impl ReadWriteChecking {
    pub fn unbatched_relation() -> UnbatchedRelation {
        let v = UnbatchedClaimExpr::polynomial;
        let c = |polynomial: JoltCommittedPolynomial| UnbatchedClaimExpr::polynomial(polynomial);
        UnbatchedRelation {
            output_relation: JoltRelationId::RamReadWriteChecking,
            gamma: RamReadWriteChallenge::Gamma.into(),
            claims: vec![
                UnbatchedClaim {
                    input_relation: JoltRelationId::SpartanOuter,
                    input: v(JoltVirtualPolynomial::RamReadValue),
                    output: v(JoltVirtualPolynomial::RamRa) * v(JoltVirtualPolynomial::RamVal),
                    output_weight: RamReadWritePublic::EqCycle.into(),
                    offset: false,
                },
                UnbatchedClaim {
                    input_relation: JoltRelationId::SpartanOuter,
                    input: v(JoltVirtualPolynomial::RamWriteValue),
                    output: v(JoltVirtualPolynomial::RamRa)
                        * (v(JoltVirtualPolynomial::RamVal) + c(JoltCommittedPolynomial::RamInc)),
                    output_weight: RamReadWritePublic::EqCycle.into(),
                    offset: false,
                },
                UnbatchedClaim {
                    input_relation: JoltRelationId::SpartanOuter,
                    input: UnbatchedClaimExpr::constant(0),
                    output: v(JoltVirtualPolynomial::RamRa) * c(JoltCommittedPolynomial::RamInc),
                    output_weight: RamReadWritePublic::CodeEqCycle.into(),
                    offset: false,
                },
                // The only RAM one-hotness check. With Booleanity of every committed
                // `RamRa` chunk, `Σ_k ra(k, j) = Load(j) + Store(j)` (exclusive Boolean
                // flags; `load_store_exclusive` in jolt-riscv) makes `ra = Π_i ra_i`
                // select one word on memory rows and none on other rows. Chunk weights
                // are not checked individually.
                UnbatchedClaim {
                    input_relation: JoltRelationId::SpartanOuter,
                    input: v(JoltVirtualPolynomial::OpFlags(CircuitFlags::Load))
                        + v(JoltVirtualPolynomial::OpFlags(CircuitFlags::Store)),
                    output: v(JoltVirtualPolynomial::RamRa),
                    output_weight: RamReadWritePublic::EqCycle.into(),
                    offset: false,
                },
            ],
        }
    }
}

impl SymbolicSumcheck for ReadWriteChecking {
    type RelationId = JoltRelationId;
    type OpeningId = crate::protocols::jolt::JoltOpeningId;
    type DerivedId = crate::protocols::jolt::JoltDerivedId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = ReadWriteDimensions;
    type Challenges<F> = RamReadWriteChallenges<F>;
    type Inputs<C> = RamReadWriteInputClaims<C>;
    type Outputs<C> = RamReadWriteOutputClaims<C>;

    fn new(shape: ReadWriteDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::RamReadWriteChecking
    }

    fn rounds(&self) -> usize {
        self.shape.read_write_rounds()
    }

    fn degree(&self) -> usize {
        3
    }

    fn input_expression<F: Ring>(&self) -> JoltExpr<F> {
        Self::unbatched_relation().folded_input()
    }

    fn output_expression<F: Ring>(&self) -> JoltExpr<F> {
        Self::unbatched_relation().folded_output()
    }
}
