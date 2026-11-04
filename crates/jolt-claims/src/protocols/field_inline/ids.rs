use derive_more::{From, TryInto};
use serde::{Deserialize, Serialize};

use crate::Expr;

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

impl FieldInlineCommittedPolynomial {
    /// All field-inline committed polynomials in proof-payload order.
    pub const ALL: [Self; 1] = [Self::FieldRdInc];
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

// `TryInto` is the `From` embeddings' inverse: each relation's
// `derive_output_term` projects the family id onto its own public enum through
// the generated `TryFrom` impls instead of a hand-written destructure.
#[derive(
    Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize, From, TryInto,
)]
pub enum FieldInlineDerivedId {
    FieldRegistersClaimReduction(FieldRegistersClaimReductionPublic),
    FieldRegistersReadWrite(FieldRegistersReadWritePublic),
    FieldRegistersValEvaluation(FieldRegistersValEvaluationPublic),
    FieldRegistersIncClaimReduction(FieldRegistersIncClaimReductionPublic),
}

#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::field_inline; FieldInlineRelationId {
    FieldRegistersSpartanOuter = 0,
    FieldRegistersClaimReduction = 1,
    FieldRegistersProduct = 2,
    FieldRegistersReadWriteChecking = 3,
    FieldRegistersValEvaluation = 4,
    FieldRegistersIncClaimReduction = 5,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::field_inline; FieldRegistersClaimReductionChallenge {
    Gamma = 0,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::field_inline; FieldRegistersClaimReductionPublic {
    EqSpartan = 0,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::field_inline; FieldRegistersReadWriteChallenge {
    Gamma = 0,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::field_inline; FieldRegistersReadWritePublic {
    EqCycle = 0,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::field_inline; FieldRegistersValEvaluationPublic {
    LtCycle = 0,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::field_inline; FieldRegistersIncClaimReductionChallenge {
    Gamma = 0,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::field_inline; FieldInlineChallengeId {
    FieldRegistersClaimReduction(value0: FieldRegistersClaimReductionChallenge) = 0,
    FieldRegistersReadWrite(value0: FieldRegistersReadWriteChallenge) = 1,
    FieldRegistersIncClaimReduction(value0: FieldRegistersIncClaimReductionChallenge) = 2,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::field_inline; FieldInlineCommittedPolynomial {
    FieldRdInc = 0,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::field_inline; FieldInlineVirtualPolynomial {
    FieldRs1Value = 0,
    FieldRs2Value = 1,
    FieldRdValue = 2,
    FieldProduct = 3,
    FieldInvProduct = 4,
    FieldRs1Ra = 5,
    FieldRs2Ra = 6,
    FieldRdWa = 7,
    FieldRegistersVal = 8,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::field_inline; FieldInlinePolynomialId {
    Committed(value0: FieldInlineCommittedPolynomial) = 0,
    Virtual(value0: FieldInlineVirtualPolynomial) = 1,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::field_inline; FieldInlineOpeningId {
    Polynomial { polynomial: FieldInlinePolynomialId, relation: FieldInlineRelationId } = 0,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::field_inline; FieldRegistersIncClaimReductionPublic {
    EqReadWrite = 0,
    EqValEvaluation = 1,
} }
#[cfg(feature = "immutable-recipes")]
crate::recipe_id! { crate::protocols::field_inline; FieldInlineDerivedId {
    FieldRegistersClaimReduction(value0: FieldRegistersClaimReductionPublic) = 0,
    FieldRegistersReadWrite(value0: FieldRegistersReadWritePublic) = 1,
    FieldRegistersValEvaluation(value0: FieldRegistersValEvaluationPublic) = 2,
    FieldRegistersIncClaimReduction(value0: FieldRegistersIncClaimReductionPublic) = 3,
} }
