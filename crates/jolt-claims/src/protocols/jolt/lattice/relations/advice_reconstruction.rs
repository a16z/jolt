//! Advice word virtualization: in lattice mode only the byte one-hot advice
//! polynomials are committed, so the word-valued
//! `TrustedAdvice`/`UntrustedAdvice` claims produced by the RAM sumchecks
//! (and reduced by the base `AdviceClaimReduction`) are settled against the
//! byte one-hots by reconstruction sumchecks — the word polynomials are
//! never PCS-opened.
//!
//! The decode identity: `advice(word) = Σ_{byte, place} value(byte) ·
//! 256^place · Bytes(byte ‖ place ‖ word)`; the weight is multilinear per
//! variable, so it sumchecks cleanly
//! ([`byte_decode_weight`](super::super::geometry::byte_decode_weight) is
//! its bound evaluation).
//!
//! **Untrusted** advice bytes are prover-supplied, so the same sumcheck also
//! proves the polynomial is a well-formed byte one-hot encoding — which is
//! simultaneously the range check the decode leg relies on (each byte < 256,
//! each word < 2^64). Three γ-batched legs, summed over the full
//! `(byte ‖ place ‖ word)` domain with `B` the byte one-hot:
//!
//! - **booleanity**: `Σ_z eq(z, r) · (B(z)² − B(z)) = 0`,
//! - **hamming**: `Σ_z eq((place, word), r_pw) · B(z) = 1` — the byte
//!   variables are summed rather than eq-bound, so each byte place holds
//!   exactly one hot byte,
//! - **decode**: `Σ_z eq(word, r_advice) · value(byte) · 256^place ·
//!   B(z) = advice(r_advice)` — the incoming word claim.
//!
//! The legs must share one sumcheck: the packed witness admits exactly one
//! claim per slot, so a standalone validity relation and a standalone decode
//! reduction would each pin `UntrustedAdviceBytes` at a different point.
//!
//! **Trusted** advice shares the encoding but is precommitted by a party the
//! verifier trusts, so no validity legs are spent on it; its relation is the
//! decode leg alone, binding only the `(byte ‖ place)` variables (the word
//! point is fixed by the incoming claim, mirroring how the inc chunk
//! reconstruction fixes its cycle point).

use jolt_field::RingCore;
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::claim_reductions::advice::final_advice_opening;
use crate::protocols::jolt::{
    JoltAdviceKind, JoltCommittedPolynomial, JoltExpr, JoltOpeningId, JoltRelationId,
    TrustedAdviceReconstructionPublic, UntrustedAdviceReconstructionChallenge,
    UntrustedAdviceReconstructionPublic,
};
use crate::{
    challenge, derived, opening, InputClaims, OutputClaims, SumcheckChallenges, SymbolicSumcheck,
};

use super::super::geometry::{byte_place_vars, word_byte_num_vars};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdviceReconstructionDimensions {
    pub word_vars: usize,
}

impl AdviceReconstructionDimensions {
    /// The untrusted sumcheck round count — the byte one-hot's slot variable
    /// count by construction (both come from [`word_byte_num_vars`]).
    pub fn num_vars(self) -> usize {
        word_byte_num_vars(self.word_vars)
    }
}

/// The untrusted advice byte one-hot opening at the bound point — the final
/// claim the packed opening consumes for the slot.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(UntrustedAdviceReconstruction)]
pub struct UntrustedAdviceReconstructionOutputClaims<C> {
    #[opening(committed = UntrustedAdviceBytes)]
    pub bytes: C,
}

/// The consumed word claim: the base advice reduction's untrusted terminus.
#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
pub struct UntrustedAdviceReconstructionInputClaims<C> {
    #[opening(untrusted_advice, from = AdviceClaimReduction)]
    pub word: C,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, SumcheckChallenges)]
pub struct UntrustedAdviceReconstructionChallenges<F> {
    #[challenge(UntrustedAdviceReconstructionChallenge::Gamma)]
    pub gamma: F,
}

#[derive(Clone)]
pub struct UntrustedAdviceReconstruction {
    shape: AdviceReconstructionDimensions,
}

impl SymbolicSumcheck for UntrustedAdviceReconstruction {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = crate::protocols::jolt::JoltDerivedId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = AdviceReconstructionDimensions;
    type Challenges<F> = UntrustedAdviceReconstructionChallenges<F>;
    type Inputs<C> = UntrustedAdviceReconstructionInputClaims<C>;
    type Outputs<C> = UntrustedAdviceReconstructionOutputClaims<C>;

    fn new(shape: AdviceReconstructionDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::UntrustedAdviceReconstruction
    }

    fn rounds(&self) -> usize {
        self.shape.num_vars()
    }

    fn degree(&self) -> usize {
        3
    }

    /// The booleanity leg sums to zero, the hamming leg to one, and the
    /// decode leg to the incoming word claim.
    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = challenge(UntrustedAdviceReconstructionChallenge::Gamma);
        gamma.clone() + gamma.pow(2) * opening(final_advice_opening(JoltAdviceKind::Untrusted))
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = challenge(UntrustedAdviceReconstructionChallenge::Gamma);
        let bytes = opening(untrusted_advice_bytes_opening());

        derived(UntrustedAdviceReconstructionPublic::EqBytePlaceWord)
            * (bytes.clone() * bytes.clone() - bytes.clone())
            + gamma.clone()
                * derived(UntrustedAdviceReconstructionPublic::EqPlaceWord)
                * bytes.clone()
            + gamma.pow(2)
                * derived(UntrustedAdviceReconstructionPublic::ByteDecode)
                * derived(UntrustedAdviceReconstructionPublic::EqWord)
                * bytes
    }
}

pub fn untrusted_advice_bytes_opening() -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::UntrustedAdviceBytes,
        JoltRelationId::UntrustedAdviceReconstruction,
    )
}

/// The trusted advice byte one-hot opening at `(bound (byte ‖ place) ‖
/// r_word)` — the final claim the packed opening consumes for the slot.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(TrustedAdviceReconstruction)]
pub struct TrustedAdviceReconstructionOutputClaims<C> {
    #[opening(committed = TrustedAdviceBytes)]
    pub bytes: C,
}

/// The consumed word claim: the base advice reduction's trusted terminus.
#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
pub struct TrustedAdviceReconstructionInputClaims<C> {
    #[opening(trusted_advice, from = AdviceClaimReduction)]
    pub word: C,
}

#[derive(Clone)]
pub struct TrustedAdviceReconstruction;

impl SymbolicSumcheck for TrustedAdviceReconstruction {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = crate::protocols::jolt::JoltDerivedId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = ();
    type Challenges<F> = crate::NoChallenges<F>;
    type Inputs<C> = TrustedAdviceReconstructionInputClaims<C>;
    type Outputs<C> = TrustedAdviceReconstructionOutputClaims<C>;

    fn new(_shape: ()) -> Self {
        Self
    }

    fn id() -> JoltRelationId {
        JoltRelationId::TrustedAdviceReconstruction
    }

    fn rounds(&self) -> usize {
        byte_place_vars()
    }

    fn degree(&self) -> usize {
        2
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        opening(final_advice_opening(JoltAdviceKind::Trusted))
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        derived(TrustedAdviceReconstructionPublic::ByteDecode)
            * opening(trusted_advice_bytes_opening())
    }
}

pub fn trusted_advice_bytes_opening() -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::TrustedAdviceBytes,
        JoltRelationId::TrustedAdviceReconstruction,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::jolt::JoltDerivedId;
    use jolt_field::{Fr, FromPrimitiveInt};

    #[test]
    fn untrusted_reconstruction_evaluates_like_core_formula() {
        let relation =
            UntrustedAdviceReconstruction::new(AdviceReconstructionDimensions { word_vars: 4 });

        let bytes = Fr::from_u64(3);
        let gamma = Fr::from_u64(5);
        let eq_byte_place_word = Fr::from_u64(7);
        let eq_place_word = Fr::from_u64(11);
        let byte_decode = Fr::from_u64(13);
        let eq_word = Fr::from_u64(17);
        let word_claim = Fr::from_u64(19);
        let zero = Fr::from_u64(0);

        let input = relation.input_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == final_advice_opening(JoltAdviceKind::Untrusted) => word_claim,
                _ => zero,
            },
            |_| gamma,
            |_| zero,
        );
        assert_eq!(input, gamma + gamma * gamma * word_claim);

        let output = relation.output_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == untrusted_advice_bytes_opening() => bytes,
                _ => zero,
            },
            |_| gamma,
            |id| match *id {
                JoltDerivedId::UntrustedAdviceReconstruction(
                    UntrustedAdviceReconstructionPublic::EqBytePlaceWord,
                ) => eq_byte_place_word,
                JoltDerivedId::UntrustedAdviceReconstruction(
                    UntrustedAdviceReconstructionPublic::EqPlaceWord,
                ) => eq_place_word,
                JoltDerivedId::UntrustedAdviceReconstruction(
                    UntrustedAdviceReconstructionPublic::ByteDecode,
                ) => byte_decode,
                JoltDerivedId::UntrustedAdviceReconstruction(
                    UntrustedAdviceReconstructionPublic::EqWord,
                ) => eq_word,
                _ => zero,
            },
        );
        assert_eq!(
            output,
            eq_byte_place_word * (bytes * bytes - bytes)
                + gamma * eq_place_word * bytes
                + gamma * gamma * byte_decode * eq_word * bytes
        );
    }

    #[test]
    fn untrusted_reconstruction_exposes_expected_dependencies() {
        let relation =
            UntrustedAdviceReconstruction::new(AdviceReconstructionDimensions { word_vars: 4 });

        assert_eq!(
            UntrustedAdviceReconstruction::id(),
            JoltRelationId::UntrustedAdviceReconstruction
        );
        assert_eq!(relation.rounds(), 8 + 3 + 4);
        assert_eq!(relation.degree(), 3);
    }

    #[test]
    fn trusted_reconstruction_evaluates_like_core_formula() {
        let relation = TrustedAdviceReconstruction::new(());

        let bytes = Fr::from_u64(3);
        let byte_decode = Fr::from_u64(7);
        let word_claim = Fr::from_u64(11);
        let zero = Fr::from_u64(0);

        let input = relation.input_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == final_advice_opening(JoltAdviceKind::Trusted) => word_claim,
                _ => zero,
            },
            |_| zero,
            |_| zero,
        );
        assert_eq!(input, word_claim);

        let output = relation.output_expression::<Fr>().evaluate(
            |id| match *id {
                id if id == trusted_advice_bytes_opening() => bytes,
                _ => zero,
            },
            |_| zero,
            |id| match *id {
                JoltDerivedId::TrustedAdviceReconstruction(
                    TrustedAdviceReconstructionPublic::ByteDecode,
                ) => byte_decode,
                _ => zero,
            },
        );
        assert_eq!(output, byte_decode * bytes);
    }

    #[test]
    fn trusted_reconstruction_exposes_expected_dependencies() {
        let relation = TrustedAdviceReconstruction::new(());

        assert_eq!(
            TrustedAdviceReconstruction::id(),
            JoltRelationId::TrustedAdviceReconstruction
        );
        assert_eq!(relation.rounds(), 8 + 3);
        assert_eq!(relation.degree(), 2);
    }
}
