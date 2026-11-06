use derive_more::{From, TryInto};
use serde::{Deserialize, Serialize};

use crate::Expr;

pub type FieldInlineExpr<F> =
    Expr<F, FieldInlineOpeningId, FieldInlineDerivedId, FieldInlineChallengeId>;

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
#[cfg_attr(feature = "immutable-recipes", derive(jolt_claims_derive::KeyVisit))]
#[cfg_attr(feature = "immutable-recipes", recipe_namespace(crate::protocols::field_inline))]
pub enum FieldInlineRelationId {
    FieldRegistersSpartanOuter,
    FieldRegistersClaimReduction,
    FieldRegistersProduct,
    FieldRegistersReadWriteChecking,
    FieldRegistersValEvaluation,
    FieldRegistersIncClaimReduction,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
#[cfg_attr(feature = "immutable-recipes", derive(jolt_claims_derive::KeyVisit))]
#[cfg_attr(feature = "immutable-recipes", recipe_namespace(crate::protocols::field_inline))]
pub enum FieldRegistersClaimReductionChallenge {
    Gamma,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
#[cfg_attr(feature = "immutable-recipes", derive(jolt_claims_derive::KeyVisit))]
#[cfg_attr(feature = "immutable-recipes", recipe_namespace(crate::protocols::field_inline))]
pub enum FieldRegistersClaimReductionPublic {
    EqSpartan,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
#[cfg_attr(feature = "immutable-recipes", derive(jolt_claims_derive::KeyVisit))]
#[cfg_attr(feature = "immutable-recipes", recipe_namespace(crate::protocols::field_inline))]
pub enum FieldRegistersReadWriteChallenge {
    Gamma,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
#[cfg_attr(feature = "immutable-recipes", derive(jolt_claims_derive::KeyVisit))]
#[cfg_attr(feature = "immutable-recipes", recipe_namespace(crate::protocols::field_inline))]
pub enum FieldRegistersReadWritePublic {
    EqCycle,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
#[cfg_attr(feature = "immutable-recipes", derive(jolt_claims_derive::KeyVisit))]
#[cfg_attr(feature = "immutable-recipes", recipe_namespace(crate::protocols::field_inline))]
pub enum FieldRegistersValEvaluationPublic {
    LtCycle,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
#[cfg_attr(feature = "immutable-recipes", derive(jolt_claims_derive::KeyVisit))]
#[cfg_attr(feature = "immutable-recipes", recipe_namespace(crate::protocols::field_inline))]
pub enum FieldRegistersIncClaimReductionChallenge {
    Gamma,
}

#[derive(
    Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize, From,
)]
#[cfg_attr(feature = "immutable-recipes", derive(jolt_claims_derive::KeyVisit))]
#[cfg_attr(feature = "immutable-recipes", recipe_namespace(crate::protocols::field_inline))]
pub enum FieldInlineChallengeId {
    FieldRegistersClaimReduction(FieldRegistersClaimReductionChallenge),
    FieldRegistersReadWrite(FieldRegistersReadWriteChallenge),
    FieldRegistersIncClaimReduction(FieldRegistersIncClaimReductionChallenge),
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
#[cfg_attr(feature = "immutable-recipes", derive(jolt_claims_derive::KeyVisit))]
#[cfg_attr(feature = "immutable-recipes", recipe_namespace(crate::protocols::field_inline))]
pub enum FieldInlineCommittedPolynomial {
    FieldRdInc,
}

impl FieldInlineCommittedPolynomial {
    /// All field-inline committed polynomials in proof-payload order.
    pub const ALL: [Self; 1] = [Self::FieldRdInc];
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
#[cfg_attr(feature = "immutable-recipes", derive(jolt_claims_derive::KeyVisit))]
#[cfg_attr(feature = "immutable-recipes", recipe_namespace(crate::protocols::field_inline))]
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
#[cfg_attr(feature = "immutable-recipes", derive(jolt_claims_derive::KeyVisit))]
#[cfg_attr(feature = "immutable-recipes", recipe_namespace(crate::protocols::field_inline))]
pub enum FieldInlinePolynomialId {
    Committed(FieldInlineCommittedPolynomial),
    Virtual(FieldInlineVirtualPolynomial),
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
#[cfg_attr(feature = "immutable-recipes", derive(jolt_claims_derive::KeyVisit))]
#[cfg_attr(feature = "immutable-recipes", recipe_namespace(crate::protocols::field_inline))]
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
#[cfg_attr(feature = "immutable-recipes", derive(jolt_claims_derive::KeyVisit))]
#[cfg_attr(feature = "immutable-recipes", recipe_namespace(crate::protocols::field_inline))]
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
#[cfg_attr(feature = "immutable-recipes", derive(jolt_claims_derive::KeyVisit))]
#[cfg_attr(feature = "immutable-recipes", recipe_namespace(crate::protocols::field_inline))]
pub enum FieldInlineDerivedId {
    FieldRegistersClaimReduction(FieldRegistersClaimReductionPublic),
    FieldRegistersReadWrite(FieldRegistersReadWritePublic),
    FieldRegistersValEvaluation(FieldRegistersValEvaluationPublic),
    FieldRegistersIncClaimReduction(FieldRegistersIncClaimReductionPublic),
}
