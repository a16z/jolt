use derive_more::From;
use serde::{Deserialize, Serialize};

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
    EqSpartan,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FieldRegistersReadWriteChallenge {
    Gamma,
    EqCycle,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FieldRegistersValEvaluationChallenge {
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
    FieldRegistersValEvaluation(FieldRegistersValEvaluationChallenge),
    FieldRegistersIncClaimReduction(FieldRegistersIncClaimReductionChallenge),
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FieldInlineCommittedPolynomial {
    FieldRdInc,
    FieldRegistersRa(usize),
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FieldInlineVirtualPolynomial {
    FieldRs1Value,
    FieldRs2Value,
    FieldRdValue,
    FieldProduct,
    FieldRs1Ra,
    FieldRs2Ra,
    FieldRdWa,
    FieldRegistersVal,
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
pub enum FieldInlinePublicId {
    FieldRegistersIncClaimReduction(FieldRegistersIncClaimReductionPublic),
}
