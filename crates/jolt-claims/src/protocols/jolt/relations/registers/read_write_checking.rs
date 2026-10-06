use jolt_field::Ring;
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::{
    JoltCommittedPolynomial, JoltExpr, JoltRelationId, JoltVirtualPolynomial, ReadWriteDimensions,
    RegistersReadWriteChallenge, RegistersReadWritePublic, UnbatchedClaim, UnbatchedClaimExpr,
    UnbatchedRelation,
};
use crate::SymbolicSumcheck;
use crate::{InputClaims, OutputClaims, SumcheckChallenges};

/// Produced register read-write openings, all sharing the single read-write
/// opening point. Generic over the opening cell (`F` for the serialized wire
/// value, `Vec<F>` for the derived opening point).
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(RegistersReadWriteChecking)]
pub struct RegistersReadWriteOutputClaims<C> {
    #[opening(RegistersVal)]
    pub registers_val: C,
    #[opening(Rs1Ra)]
    pub rs1_ra: C,
    #[opening(Rs2Ra)]
    pub rs2_ra: C,
    #[opening(RdWa)]
    pub rd_wa: C,
    #[opening(committed = RdInc)]
    pub rd_inc: C,
}

/// Consumed register openings reduced by the read-write checking sumcheck, wired
/// from the upstream registers claim-reduction relation (stage 3). Generic over
/// the cell.
#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
pub struct RegistersReadWriteInputClaims<C> {
    #[opening(RdWriteValue, from = RegistersClaimReduction)]
    pub rd_write_value: C,
    #[opening(Rs1Value, from = RegistersClaimReduction)]
    pub rs1_value: C,
    #[opening(Rs2Value, from = RegistersClaimReduction)]
    pub rs2_value: C,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, SumcheckChallenges)]
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
pub struct RegistersReadWriteChallenges<F> {
    #[challenge(RegistersReadWriteChallenge::Gamma)]
    pub gamma: F,
}

/// The registers read/write checking sumcheck: relates the read-value claims
/// (`RdWriteValue`, `Rs1Value`, `Rs2Value`) folded by `gamma` to the register
/// `val`/`ra`/`inc` openings weighted by the `EqCycle` public.
#[derive(Clone)]
pub struct ReadWriteChecking {
    shape: ReadWriteDimensions,
}

impl ReadWriteChecking {
    pub fn unbatched_relation() -> UnbatchedRelation {
        let v = UnbatchedClaimExpr::polynomial;
        let c = |polynomial: JoltCommittedPolynomial| UnbatchedClaimExpr::polynomial(polynomial);
        UnbatchedRelation {
            output_relation: JoltRelationId::RegistersReadWriteChecking,
            gamma: RegistersReadWriteChallenge::Gamma.into(),
            claims: vec![
                UnbatchedClaim {
                    input_relation: JoltRelationId::RegistersClaimReduction,
                    input: v(JoltVirtualPolynomial::RdWriteValue),
                    output: v(JoltVirtualPolynomial::RdWa)
                        * (v(JoltVirtualPolynomial::RegistersVal)
                            + c(JoltCommittedPolynomial::RdInc)),
                    output_weight: RegistersReadWritePublic::EqCycle.into(),
                    offset: false,
                },
                UnbatchedClaim {
                    input_relation: JoltRelationId::RegistersClaimReduction,
                    input: v(JoltVirtualPolynomial::Rs1Value),
                    output: v(JoltVirtualPolynomial::Rs1Ra)
                        * v(JoltVirtualPolynomial::RegistersVal),
                    output_weight: RegistersReadWritePublic::EqCycle.into(),
                    offset: false,
                },
                UnbatchedClaim {
                    input_relation: JoltRelationId::RegistersClaimReduction,
                    input: v(JoltVirtualPolynomial::Rs2Value),
                    output: v(JoltVirtualPolynomial::Rs2Ra)
                        * v(JoltVirtualPolynomial::RegistersVal),
                    output_weight: RegistersReadWritePublic::EqCycle.into(),
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
    type Challenges<F> = RegistersReadWriteChallenges<F>;
    type Inputs<C> = RegistersReadWriteInputClaims<C>;
    type Outputs<C> = RegistersReadWriteOutputClaims<C>;

    fn new(shape: ReadWriteDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::RegistersReadWriteChecking
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
