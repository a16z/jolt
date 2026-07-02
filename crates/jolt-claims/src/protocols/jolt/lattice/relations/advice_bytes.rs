//! Untrusted-advice byte validity: proves the prover-supplied advice byte
//! column is a well-formed byte one-hot encoding. This is simultaneously the
//! range check behind the advice decode view — byte one-hot cells force every
//! decoded byte below 256 and every decoded word below 2^64.
//!
//! One sumcheck over the column's `(symbol ‖ limb ‖ word)` cell domain
//! γ-batches two legs:
//!
//! - **booleanity**: `Σ_cells eq(cell, r) · (cell² − cell) = 0`,
//! - **hamming**: `Σ_cells eq((limb, word), r_lw) · cell = 1` — the symbol
//!   variables are summed rather than eq-bound, so each byte position holds
//!   exactly one hot symbol.
//!
//! Trusted advice shares the encoding but is precommitted by a party the
//! verifier trusts, so no relation is spent on it.

use jolt_field::RingCore;
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::{
    AdviceBytesValidityChallenge, AdviceBytesValidityPublic, JoltExpr, JoltOpeningId,
    JoltRelationId,
};
use crate::{challenge, derived, opening, OutputClaims, SumcheckChallenges, SymbolicSumcheck};

use super::super::geometry::{BYTE_SYMBOL_BITS, WORD_BYTE_LIMB_BITS};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdviceBytesDimensions {
    pub word_vars: usize,
}

impl AdviceBytesDimensions {
    pub const fn new(word_vars: usize) -> Self {
        Self { word_vars }
    }

    pub const fn cell_vars(self) -> usize {
        BYTE_SYMBOL_BITS + WORD_BYTE_LIMB_BITS + self.word_vars
    }
}

/// The untrusted advice byte-column opening at the bound cell point — the
/// leaf claim the packed opening consumes for the advice byte column.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(AdviceBytesValidity)]
pub struct AdviceBytesValidityOutputClaims<C> {
    #[opening(committed = UntrustedAdviceBytes)]
    pub bytes: C,
}

#[derive(Clone, Copy, Debug, SumcheckChallenges)]
pub struct AdviceBytesValidityChallenges<F> {
    #[challenge(AdviceBytesValidityChallenge::Gamma)]
    pub gamma: F,
}

pub struct AdviceBytesValidity {
    shape: AdviceBytesDimensions,
}

impl SymbolicSumcheck for AdviceBytesValidity {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = crate::protocols::jolt::JoltDerivedId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = AdviceBytesDimensions;
    type Challenges<F> = AdviceBytesValidityChallenges<F>;
    type Inputs<C> = crate::NoInputs<C>;
    type Outputs<C> = AdviceBytesValidityOutputClaims<C>;

    fn new(shape: AdviceBytesDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::AdviceBytesValidity
    }

    fn rounds(&self) -> usize {
        self.shape.cell_vars()
    }

    fn degree(&self) -> usize {
        3
    }

    /// The booleanity leg sums to zero; the hamming leg's claimed sum is the
    /// eq-weighted count of hot cells, `Σ_{l,w} eq((l,w), r_lw) · 1 = 1`.
    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        challenge(AdviceBytesValidityChallenge::Gamma)
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = challenge(AdviceBytesValidityChallenge::Gamma);
        let bytes = opening(advice_bytes_validity_opening());

        derived(AdviceBytesValidityPublic::EqCell) * (bytes.clone() * bytes.clone() - bytes.clone())
            + gamma * derived(AdviceBytesValidityPublic::EqLimbWord) * bytes
    }
}

pub fn advice_bytes_validity_opening() -> JoltOpeningId {
    JoltOpeningId::committed(
        crate::protocols::jolt::JoltCommittedPolynomial::UntrustedAdviceBytes,
        JoltRelationId::AdviceBytesValidity,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::jolt::{JoltChallengeId, JoltDerivedId};
    use jolt_field::{Fr, FromPrimitiveInt};

    #[test]
    fn validity_evaluates_like_core_formula() {
        let relation = AdviceBytesValidity::new(AdviceBytesDimensions::new(4));

        let bytes = Fr::from_u64(3);
        let gamma = Fr::from_u64(5);
        let eq_cell = Fr::from_u64(7);
        let eq_limb_word = Fr::from_u64(11);
        let zero = Fr::from_u64(0);

        let input = relation
            .input_expression::<Fr>()
            .evaluate(|_| zero, |_| gamma, |_| zero);
        assert_eq!(input, gamma);

        let output = relation.output_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == advice_bytes_validity_opening() => bytes,
                _ => zero,
            },
            |_| gamma,
            |id| match *id {
                JoltDerivedId::AdviceBytesValidity(AdviceBytesValidityPublic::EqCell) => eq_cell,
                JoltDerivedId::AdviceBytesValidity(AdviceBytesValidityPublic::EqLimbWord) => {
                    eq_limb_word
                }
                _ => zero,
            },
        );
        assert_eq!(
            output,
            eq_cell * (bytes * bytes - bytes) + gamma * eq_limb_word * bytes
        );
    }

    #[test]
    fn validity_exposes_expected_dependencies() {
        let relation = AdviceBytesValidity::new(AdviceBytesDimensions::new(4));

        assert_eq!(
            AdviceBytesValidity::id(),
            JoltRelationId::AdviceBytesValidity
        );
        assert_eq!(relation.rounds(), 8 + 3 + 4);
        assert_eq!(relation.degree(), 3);
        assert!(relation
            .input_expression::<Fr>()
            .required_openings()
            .is_empty());
        assert_eq!(
            relation.output_expression::<Fr>().required_openings(),
            vec![advice_bytes_validity_opening()]
        );
        assert_eq!(
            relation.required_challenges::<Fr>(),
            vec![JoltChallengeId::from(AdviceBytesValidityChallenge::Gamma)]
        );
        assert_eq!(
            relation.required_deriveds::<Fr>(),
            vec![
                JoltDerivedId::from(AdviceBytesValidityPublic::EqCell),
                JoltDerivedId::from(AdviceBytesValidityPublic::EqLimbWord),
            ]
        );
    }
}
