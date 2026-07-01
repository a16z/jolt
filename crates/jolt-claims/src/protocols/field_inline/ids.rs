use derive_more::From;
use serde::{Deserialize, Serialize};

use crate::Expr;

/// The field-inline protocol's expression type: an [`Expr`](crate::Expr) over the
/// field-inline id families (openings, deriveds, challenges).
pub type FieldInlineExpr<F> =
    Expr<F, FieldInlineOpeningId, FieldInlineDerivedId, FieldInlineChallengeId>;

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FieldInlineRelationId {
    FieldRegistersSpartanOuter,
    FieldRegistersClaimReduction,
    FieldRegistersProduct,
    FieldRegistersReadWriteChecking,
    FieldRegistersValEvaluation,
    FieldRegistersIncClaimReduction,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FieldRegistersClaimReductionChallenge {
    Gamma,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FieldRegistersClaimReductionPublic {
    EqSpartan,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FieldRegistersReadWriteChallenge {
    Gamma,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FieldRegistersReadWritePublic {
    EqCycle,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FieldRegistersValEvaluationPublic {
    LtCycle,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FieldRegistersIncClaimReductionChallenge {
    Gamma,
}

#[derive(
    Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize, From,
)]
pub enum FieldInlineChallengeId {
    FieldRegistersClaimReduction(FieldRegistersClaimReductionChallenge),
    FieldRegistersReadWrite(FieldRegistersReadWriteChallenge),
    FieldRegistersIncClaimReduction(FieldRegistersIncClaimReductionChallenge),
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FieldInlineCommittedPolynomial {
    FieldRdInc,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FieldInlineOpFlag {
    Add,
    Sub,
    Mul,
    Inv,
    AssertEq,
    LoadFromX,
    StoreToX,
    LoadImm,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FieldInlineVirtualPolynomial {
    FieldRs1Value,
    FieldRs2Value,
    FieldRdValue,
    FieldProduct,
    FieldInvProduct,
    FieldRs1Ra,
    FieldRs2Ra,
    FieldRdWa,
    FieldRegistersVal,
    FieldOpFlag(FieldInlineOpFlag),
}

#[derive(
    Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize, From,
)]
pub enum FieldInlinePolynomialId {
    Committed(FieldInlineCommittedPolynomial),
    Virtual(FieldInlineVirtualPolynomial),
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FieldInlineOpeningId {
    Polynomial {
        polynomial: FieldInlinePolynomialId,
        relation: FieldInlineRelationId,
    },
}

impl FieldInlineOpeningId {
    pub fn polynomial(
        polynomial: impl Into<FieldInlinePolynomialId>,
        relation: FieldInlineRelationId,
    ) -> Self {
        Self::Polynomial {
            polynomial: polynomial.into(),
            relation,
        }
    }

    pub fn committed(
        polynomial: FieldInlineCommittedPolynomial,
        relation: FieldInlineRelationId,
    ) -> Self {
        Self::polynomial(polynomial, relation)
    }

    pub fn virtual_polynomial(
        polynomial: FieldInlineVirtualPolynomial,
        relation: FieldInlineRelationId,
    ) -> Self {
        Self::polynomial(polynomial, relation)
    }
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FieldRegistersIncClaimReductionPublic {
    EqReadWrite,
    EqValEvaluation,
}

#[derive(
    Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize, From,
)]
pub enum FieldInlineDerivedId {
    FieldRegistersClaimReduction(FieldRegistersClaimReductionPublic),
    FieldRegistersReadWrite(FieldRegistersReadWritePublic),
    FieldRegistersValEvaluation(FieldRegistersValEvaluationPublic),
    FieldRegistersIncClaimReduction(FieldRegistersIncClaimReductionPublic),
}
