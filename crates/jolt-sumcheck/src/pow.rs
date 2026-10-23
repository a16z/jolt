//! Per-round proof of work.
//!
//! Each round challenge is preceded by a proof-of-work nonce whose difficulty
//! lifts the round to Jolt's Fiat-Shamir target of about `2^128` hash calls.
//! A false claim survives a round only when the challenge is a root of a
//! nonzero polynomial of degree at most `d`, so the round loses a factor `d`
//! (Schwartz-Zippel) and its difficulty is
//! `grind_bits_for_loss(d, T::CHALLENGE_SET, SECURITY_BITS)`, where
//! `T::CHALLENGE_SET` is the [`FieldTranscript`] challenge set for `F`.
//!
//! `d` is always the verifier's degree bound for the claim or statement, never
//! a degree the prover declares, so the difficulty is fixed by the protocol
//! parameters and the proof cannot lower it. The nonce sits in the proof next
//! to the round messages: one per round when the difficulty is positive, none
//! when it is zero. [`Nonced`] serializes the two together so that a proof
//! with no nonces pays no bytes for them.

use std::fmt::{Formatter, Result as FmtResult};
use std::marker::PhantomData;

use jolt_field::Field;
#[cfg(feature = "prover")]
use jolt_transcript::ProverTranscript;
use jolt_transcript::{grind_bits_for_loss, FieldTranscript, SECURITY_BITS};
use serde::de::{Error as _, SeqAccess, Visitor};
use serde::ser::{Error as _, SerializeTuple};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::error::SumcheckError;

/// Proof-of-work difficulty for a round whose polynomial has degree at most
/// `degree`, with challenges drawn by [`FieldTranscript::challenge`].
///
/// # Errors
/// [`SumcheckError::PowDifficulty`] if no supported difficulty reaches the
/// target for this degree and challenge set.
pub fn round_pow_bits<F: Field, T: FieldTranscript<F>>(
    degree: usize,
) -> Result<u32, SumcheckError<F>> {
    let loss = u64::try_from(degree.max(1)).map_err(|_| SumcheckError::PowDifficulty { degree })?;
    grind_bits_for_loss(loss, T::CHALLENGE_SET, SECURITY_BITS)
        .map_err(|_| SumcheckError::PowDifficulty { degree })
}

/// The verifier's view of a proof's round nonces, checked for count up front.
pub(crate) struct RoundNonces<'a> {
    bits: u32,
    nonces: &'a [u32],
}

impl<'a> RoundNonces<'a> {
    /// Accepts `nonces` for `rounds` rounds of degree at most `degree`: one
    /// nonce per round when the difficulty is positive, none otherwise.
    pub(crate) fn new<F: Field, T: FieldTranscript<F>>(
        degree: usize,
        rounds: usize,
        nonces: &'a [u32],
    ) -> Result<Self, SumcheckError<F>> {
        let bits = round_pow_bits::<F, T>(degree)?;
        let expected = if bits == 0 { 0 } else { rounds };
        if nonces.len() != expected {
            return Err(SumcheckError::WrongNumberOfPowNonces {
                expected,
                got: nonces.len(),
            });
        }
        Ok(Self { bits, nonces })
    }

    /// Checks round `round`'s nonce against the transcript, after the round
    /// message is absorbed and before the round challenge is drawn.
    pub(crate) fn check<F: Field, T: FieldTranscript<F>>(
        &self,
        round: usize,
        transcript: &mut T,
    ) -> Result<(), SumcheckError<F>> {
        transcript
            .check_public_grind(self.bits, self.nonces.get(round).copied())
            .map_err(|_| SumcheckError::ProofOfWorkRejected { round })
    }
}

/// Grinds round `round`'s nonce after the round message is absorbed and before
/// the round challenge is drawn, appending it to `nonces` when the difficulty
/// is positive.
#[cfg(feature = "prover")]
pub fn grind_round<F: Field, T: FieldTranscript<F> + ProverTranscript>(
    degree: usize,
    round: usize,
    nonces: &mut Vec<u32>,
    transcript: &mut T,
) -> Result<(), SumcheckError<F>> {
    let bits = round_pow_bits::<F, T>(degree)?;
    let nonce = transcript
        .public_grind(bits)
        .map_err(|_| SumcheckError::ProofOfWorkRejected { round })?;
    nonces.extend(nonce);
    Ok(())
}

/// Prover messages and the proof-of-work nonces ground among them, as
/// deserialized from a proof. [`NoncedRef`] serializes the same layout.
///
/// The wire form is one tuple: the header `2 · messages.len() + has_nonces`,
/// the messages, then, only when `has_nonces` is 1, the nonces as a
/// length-prefixed list, which must not be empty. With no nonces this costs
/// nothing beyond a plain length prefix: under bincode's varint both take one
/// byte for up to 125 messages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Nonced<M> {
    pub messages: Vec<M>,
    pub pow_nonces: Vec<u32>,
}

/// Borrowed [`Nonced`], for serialization.
pub struct NoncedRef<'a, M> {
    pub messages: &'a [M],
    pub pow_nonces: &'a [u32],
}

impl<M: Serialize> Serialize for NoncedRef<'_, M> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let has_nonces = !self.pow_nonces.is_empty();
        let header = u64::try_from(self.messages.len())
            .ok()
            .and_then(|len| len.checked_mul(2))
            .ok_or_else(|| S::Error::custom("too many messages"))?
            | u64::from(has_nonces);
        let len = self
            .messages
            .len()
            .saturating_add(1 + usize::from(has_nonces));
        let mut tuple = serializer.serialize_tuple(len)?;
        tuple.serialize_element(&header)?;
        for message in self.messages {
            tuple.serialize_element(message)?;
        }
        if has_nonces {
            tuple.serialize_element(self.pow_nonces)?;
        }
        tuple.end()
    }
}

impl<'de, M: Deserialize<'de>> Deserialize<'de> for Nonced<M> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct NoncedVisitor<M>(PhantomData<M>);

        impl<'de, M: Deserialize<'de>> Visitor<'de> for NoncedVisitor<M> {
            type Value = Nonced<M>;

            fn expecting(&self, formatter: &mut Formatter<'_>) -> FmtResult {
                formatter.write_str("a header, messages, and proof-of-work nonces")
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let missing = || A::Error::custom("truncated messages with nonces");
                let header: u64 = seq.next_element()?.ok_or_else(missing)?;
                let len = usize::try_from(header >> 1)
                    .map_err(|_| A::Error::custom("message count overflows usize"))?;
                // Grows with the bytes actually read, never with the header.
                let mut messages = Vec::new();
                for _ in 0..len {
                    messages.push(seq.next_element()?.ok_or_else(missing)?);
                }
                let pow_nonces = if header & 1 == 1 {
                    let nonces: Vec<u32> = seq.next_element()?.ok_or_else(missing)?;
                    if nonces.is_empty() {
                        return Err(A::Error::custom("empty proof-of-work nonce list"));
                    }
                    nonces
                } else {
                    Vec::new()
                };
                Ok(Nonced {
                    messages,
                    pow_nonces,
                })
            }
        }

        // The length is read from the header; the deserializer only bounds it.
        deserializer.deserialize_tuple(usize::MAX, NoncedVisitor(PhantomData))
    }
}
