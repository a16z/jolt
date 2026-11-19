#![cfg(feature = "akita")]

use akita_types::setup_claims::{SetupBatchStatement, SetupQuery};
use common::jolt_device::JoltDevice;
use jolt_akita::deferred::{SetupClaimAccumulator, SetupClaimAdvice, SetupMode};
use jolt_akita::{
    record_checked_hints, verify_with_checked_hints, AkitaError, AkitaField, AkitaScheme,
    AkitaVerificationContext, AkitaVerifierTranscript,
};
use jolt_crypto::NoVectorCommitment;
use jolt_verifier::verifier::verify_with_context;
use jolt_verifier::{JoltProof, JoltVerifierPreprocessing};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::slice::Iter;

pub type Vc = NoVectorCommitment<AkitaField>;
pub type Proof = JoltProof<AkitaScheme, Vc>;
pub type Preprocessing = JoltVerifierPreprocessing<AkitaScheme, Vc>;

/// Trusted bound on recorded challenge advice: the guests' input limit.
pub const CHALLENGE_ADVICE_LIMIT: usize = 1 << 26;

#[derive(Serialize, Deserialize)]
pub struct InnerProof {
    pub proof: Proof,
    pub io: JoltDevice,
}

#[derive(Serialize, Deserialize)]
pub struct ProofPair {
    pub preprocessing: Preprocessing,
    pub inner: [InnerProof; 2],
    pub setup_field_len: usize,
    pub answers: Vec<AkitaField>,
    /// Checked challenge advice for both inner verifications, in order.
    #[serde(with = "serde_bytes")]
    pub challenge_advice: Vec<u8>,
}

pub enum ReplayEvent {
    PrepareStart,
    PrepareEnd,
    InnerStart(usize),
    InnerEnd(usize),
}

impl ProofPair {
    /// Verify both inner proofs with the pair's challenge advice installed.
    pub fn replay(
        &self,
        mode: SetupMode,
        advice: &mut (dyn SetupClaimAdvice<AkitaField> + Send),
        observe: impl FnMut(ReplayEvent) + Send,
    ) -> Result<SetupBatchStatement<AkitaField>, String> {
        verify_with_checked_hints(&self.challenge_advice, CHALLENGE_ADVICE_LIMIT, || {
            self.verify(mode, advice, observe)
        })
        .map_err(|e| e.to_string())
    }

    /// Verify natively and record the challenge advice [`Self::replay`] installs.
    pub fn record_challenge_advice(
        &mut self,
        mode: SetupMode,
        advice: &mut (dyn SetupClaimAdvice<AkitaField> + Send),
    ) -> Result<SetupBatchStatement<AkitaField>, String> {
        let (statement, frame) = record_checked_hints(|| self.verify(mode, advice, |_| {}))
            .map_err(|e| e.to_string())?;
        let statement = statement?;
        self.challenge_advice = frame;
        Ok(statement)
    }

    fn verify(
        &self,
        mode: SetupMode,
        advice: &mut dyn SetupClaimAdvice<AkitaField>,
        mut observe: impl FnMut(ReplayEvent),
    ) -> Result<SetupBatchStatement<AkitaField>, String> {
        observe(ReplayEvent::PrepareStart);
        let polynomial = self
            .preprocessing
            .pcs_setup
            .setup_polynomial(self.setup_field_len)
            .map_err(|e| e.to_string())?;
        let accumulator = SetupClaimAccumulator::new(polynomial.clone());
        let accumulator = if mode == SetupMode::Defer {
            accumulator.with_advice(advice)
        } else {
            accumulator
        };
        let mut context = AkitaVerificationContext {
            mode,
            accumulator: Some(accumulator),
            ..Default::default()
        };
        observe(ReplayEvent::PrepareEnd);
        for (index, inner) in self.inner.iter().enumerate() {
            if inner.io.panic {
                return Err("inner guest panicked".into());
            }
            observe(ReplayEvent::InnerStart(index));
            context.cycle_marker = Some(if index == 0 {
                "inner0_akita"
            } else {
                "inner1_akita"
            });
            let verified =
                verify_with_context::<AkitaField, AkitaScheme, Vc, AkitaVerifierTranscript<'_>>(
                    &self.preprocessing,
                    &inner.io,
                    &inner.proof,
                    None,
                    &mut context,
                );
            observe(ReplayEvent::InnerEnd(index));
            verified.map_err(|e| e.to_string())?;
        }
        SetupBatchStatement::new(
            polynomial,
            2,
            context
                .accumulator
                .map_or_else(Vec::new, SetupClaimAccumulator::into_pending),
        )
        .map_err(|e| e.to_string())
    }
}

pub struct AnswerTape<'a>(pub Iter<'a, AkitaField>);

impl SetupClaimAdvice<AkitaField> for AnswerTape<'_> {
    fn answer(&mut self, _: &SetupQuery<AkitaField>) -> Result<AkitaField, AkitaError> {
        self.0
            .next()
            .copied()
            .ok_or_else(|| AkitaError::InvalidInput("setup advice exhausted".into()))
    }
}

pub fn decode_exact<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, String> {
    let (value, consumed) = bincode::serde::decode_from_slice(bytes, bincode::config::standard())
        .map_err(|e| e.to_string())?;
    if consumed != bytes.len() {
        return Err("trailing transport bytes".into());
    }
    Ok(value)
}
