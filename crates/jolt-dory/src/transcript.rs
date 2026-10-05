//! Runs dory-pcs inside the caller's NARG transcript.
//!
//! dory-pcs absorbs every prover message it relies on through
//! [`DoryTranscript::append_serde`], interleaved with its challenges in an
//! order fixed by the proof's round count. The prover adapter sends each such
//! value as a prover message, so the argument string holds exactly Dory's
//! messages in Fiat-Shamir order. dory-pcs verifies from a proof struct it
//! receives up front, so the verifier first rebuilds that struct by reading the
//! messages from the transcript's unread bytes without absorbing them, then runs `dory::verify` with
//! an adapter whose every `append_serde` receives the next message from the
//! live transcript and checks it equals the value dory-pcs absorbs.
//!
//! The Σ₁ responses of a ZK proof are the one part of the proof dory-pcs never
//! absorbs. Both sides place them after the Dory messages, so the argument
//! string carries no byte the verifier reads without absorbing.

use dory::backends::arkworks::{ArkDoryProof, ArkG2, BN254};
use dory::messages::{
    FirstReduceMessage, ScalarProductMessage, ScalarProductProof, SecondReduceMessage, Sigma1Proof,
    Sigma2Proof, VMVMessage,
};
use dory::primitives::arithmetic::Group as DoryGroup;
use dory::primitives::transcript::Transcript as DoryTranscript;
use dory::primitives::{DoryDeserialize, DorySerialize};
use jolt_field::Fr;
use jolt_openings::OpeningsError;
use jolt_transcript::{Channel, ProverTranscript, Sponge, VerifierTranscript};

use crate::scheme::{jolt_fr_to_ark, ArkFr, ArkG1, ArkGT};

/// Prover side: every absorbed value becomes a prover message.
pub(crate) struct DoryProverChannel<'a, H: Sponge> {
    transcript: &'a mut ProverTranscript<H>,
}

impl<'a, H: Sponge> DoryProverChannel<'a, H> {
    pub(crate) fn new(transcript: &'a mut ProverTranscript<H>) -> Self {
        Self { transcript }
    }

    /// Sends the Σ₁ responses dory-pcs does not absorb, after its messages.
    pub(crate) fn send_sigma1_responses(
        &mut self,
        proof: &ArkDoryProof,
    ) -> Result<(), OpeningsError> {
        let sigma1 = proof.sigma1_proof.as_ref().ok_or_else(|| {
            OpeningsError::ProveFailed("ZK proof must contain a Σ₁ proof".to_owned())
        })?;
        for response in [&sigma1.z1, &sigma1.z2, &sigma1.z3] {
            self.transcript.send_bytes(&compressed(response));
        }
        Ok(())
    }
}

impl<H: Sponge> DoryTranscript for DoryProverChannel<'_, H> {
    type Curve = BN254;

    fn append_bytes(&mut self, _label: &[u8], bytes: &[u8]) {
        self.transcript.send_bytes(bytes);
    }

    fn append_field(&mut self, _label: &[u8], x: &ArkFr) {
        self.transcript.send_bytes(&compressed(x));
    }

    fn append_group<G: DoryGroup>(&mut self, _label: &[u8], g: &G) {
        self.transcript.send_bytes(&compressed(g));
    }

    fn append_serde<S: DorySerialize>(&mut self, _label: &[u8], s: &S) {
        self.transcript.send_bytes(&compressed(s));
    }

    fn challenge_scalar(&mut self, _label: &[u8]) -> ArkFr {
        jolt_fr_to_ark(&self.transcript.challenge::<Fr>())
    }

    fn reset(&mut self, _domain_label: &[u8]) {
        unreachable!("reset is not invoked by dory-pcs and is intentionally unsupported")
    }
}

/// Verifier side: every absorbed value must be the next prover message.
pub(crate) struct DoryVerifierChannel<'a, 'p, H: Sponge> {
    transcript: &'a mut VerifierTranscript<'p, H>,
    mismatch: bool,
}

impl<'a, 'p, H: Sponge> DoryVerifierChannel<'a, 'p, H> {
    pub(crate) fn new(transcript: &'a mut VerifierTranscript<'p, H>) -> Self {
        Self {
            transcript,
            mismatch: false,
        }
    }

    /// Receives the Σ₁ responses, which must match the ones `proof` was
    /// rebuilt with, then reports whether every absorbed value matched.
    pub(crate) fn finish(mut self, proof: &ArkDoryProof) -> Result<(), OpeningsError> {
        if let Some(sigma1) = &proof.sigma1_proof {
            for response in [&sigma1.z1, &sigma1.z2, &sigma1.z3] {
                self.expect(&compressed(response));
            }
        }
        if self.mismatch {
            return Err(OpeningsError::VerificationFailed);
        }
        Ok(())
    }

    fn expect(&mut self, bytes: &[u8]) {
        match self.transcript.receive_bytes(bytes.len()) {
            Ok(received) if received == bytes => {}
            Ok(_) | Err(_) => self.mismatch = true,
        }
    }
}

impl<H: Sponge> DoryTranscript for DoryVerifierChannel<'_, '_, H> {
    type Curve = BN254;

    fn append_bytes(&mut self, _label: &[u8], bytes: &[u8]) {
        self.expect(bytes);
    }

    fn append_field(&mut self, _label: &[u8], x: &ArkFr) {
        self.expect(&compressed(x));
    }

    fn append_group<G: DoryGroup>(&mut self, _label: &[u8], g: &G) {
        self.expect(&compressed(g));
    }

    fn append_serde<S: DorySerialize>(&mut self, _label: &[u8], s: &S) {
        self.expect(&compressed(s));
    }

    fn challenge_scalar(&mut self, _label: &[u8]) -> ArkFr {
        jolt_fr_to_ark(&self.transcript.challenge::<Fr>())
    }

    fn reset(&mut self, _domain_label: &[u8]) {
        unreachable!("reset is not invoked by dory-pcs and is intentionally unsupported")
    }
}

#[expect(
    clippy::expect_used,
    reason = "serializing a group or field element into a Vec cannot fail"
)]
fn compressed<S: DorySerialize + ?Sized>(value: &S) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(value.compressed_size());
    value
        .serialize_compressed(&mut bytes)
        .expect("Dory element serialization cannot fail");
    bytes
}

/// Rebuilds the Dory proof for a `num_vars`-variable opening from the next
/// prover messages, reading them from `scout`, a scratch copy of the live
/// verifier transcript. Message order mirrors `dory::verify`.
pub(crate) fn read_proof(
    unread: &[u8],
    num_vars: usize,
    zk: bool,
) -> Result<ArkDoryProof, OpeningsError> {
    let sigma = num_vars.div_ceil(2);
    let nu = num_vars - sigma;
    let reader = &mut { unread };

    let vmv_message = VMVMessage {
        c: read::<ArkGT>(reader)?,
        d2: read(reader)?,
        e1: read::<ArkG1>(reader)?,
    };
    let zk_prefix = if zk {
        let e2 = read::<ArkG2>(reader)?;
        let y_com = read::<ArkG1>(reader)?;
        let sigma1_commitments = (read::<ArkG2>(reader)?, read::<ArkG1>(reader)?);
        let sigma2 = Sigma2Proof {
            a: read::<ArkGT>(reader)?,
            z1: read::<ArkFr>(reader)?,
            z2: read(reader)?,
        };
        Some((e2, y_com, sigma1_commitments, sigma2))
    } else {
        None
    };

    let mut first_messages = Vec::with_capacity(sigma);
    let mut second_messages = Vec::with_capacity(sigma);
    for _ in 0..sigma {
        first_messages.push(FirstReduceMessage {
            d1_left: read::<ArkGT>(reader)?,
            d1_right: read(reader)?,
            d2_left: read(reader)?,
            d2_right: read(reader)?,
            e1_beta: read::<ArkG1>(reader)?,
            e2_beta: read::<ArkG2>(reader)?,
        });
        second_messages.push(SecondReduceMessage {
            c_plus: read::<ArkGT>(reader)?,
            c_minus: read(reader)?,
            e1_plus: read::<ArkG1>(reader)?,
            e1_minus: read(reader)?,
            e2_plus: read::<ArkG2>(reader)?,
            e2_minus: read(reader)?,
        });
    }

    let Some((e2, y_com, (a1, a2), sigma2)) = zk_prefix else {
        let final_message = ScalarProductMessage {
            e1: read::<ArkG1>(reader)?,
            e2: read::<ArkG2>(reader)?,
        };
        return Ok(ArkDoryProof {
            vmv_message,
            first_messages,
            second_messages,
            final_message: Some(final_message),
            nu,
            sigma,
            e2: None,
            y_com: None,
            sigma1_proof: None,
            sigma2_proof: None,
            scalar_product_proof: None,
        });
    };

    let scalar_product = ScalarProductProof {
        p1: read::<ArkGT>(reader)?,
        p2: read(reader)?,
        q: read(reader)?,
        r: read(reader)?,
        e1: read::<ArkG1>(reader)?,
        e2: read::<ArkG2>(reader)?,
        r1: read::<ArkFr>(reader)?,
        r2: read(reader)?,
        r3: read(reader)?,
    };
    let sigma1 = Sigma1Proof {
        a1,
        a2,
        z1: read::<ArkFr>(reader)?,
        z2: read(reader)?,
        z3: read(reader)?,
    };
    Ok(ArkDoryProof {
        vmv_message,
        first_messages,
        second_messages,
        final_message: None,
        nu,
        sigma,
        e2: Some(e2),
        y_com: Some(y_com),
        sigma1_proof: Some(sigma1),
        sigma2_proof: Some(sigma2),
        scalar_product_proof: Some(scalar_product),
    })
}

/// Compressed width of each element type a Dory proof carries.
trait CompressedWidth {
    const BYTES: usize;
}

impl CompressedWidth for ArkFr {
    const BYTES: usize = 32;
}

impl CompressedWidth for ArkG1 {
    const BYTES: usize = 32;
}

impl CompressedWidth for ArkG2 {
    const BYTES: usize = 64;
}

impl CompressedWidth for ArkGT {
    const BYTES: usize = 384;
}

/// Reads one compressed element.
/// Parses the next compressed `T` from the unread proof bytes, advancing
/// `unread` past it.
fn read<T>(unread: &mut &[u8]) -> Result<T, OpeningsError>
where
    T: DoryDeserialize + CompressedWidth,
{
    let (bytes, rest) = unread
        .split_at_checked(T::BYTES)
        .ok_or(OpeningsError::VerificationFailed)?;
    let value = T::deserialize_compressed(bytes).map_err(|_| OpeningsError::VerificationFailed)?;
    *unread = rest;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use dory::primitives::arithmetic::Field as DoryField;

    use super::*;

    #[test]
    fn compressed_widths_match_arkworks() {
        fn width<T: DorySerialize + CompressedWidth>(value: &T) {
            assert_eq!(value.compressed_size(), T::BYTES);
        }
        width(&<ArkFr as DoryField>::one());
        width(&<ArkG1 as DoryGroup>::identity());
        width(&<ArkG2 as DoryGroup>::identity());
        width(&<ArkGT as DoryGroup>::identity());
    }
}
