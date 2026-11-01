#![cfg(feature = "akita")]

use akita_types::setup_claims::{SetupBatchStatement, SetupQuery};
use common::jolt_device::JoltDevice;
use jolt_akita::deferred::{SetupClaimAccumulator, SetupClaimAdvice, SetupMode};
use jolt_akita::{AkitaError, AkitaField, AkitaScheme, AkitaVerificationContext};
use jolt_crypto::NoVectorCommitment;
use jolt_transcript::{Blake2b512, VerifierState};
use jolt_verifier::verifier::verify_with_context;
use jolt_verifier::{JoltProof, JoltVerifierPreprocessing};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::slice::Iter;

pub type Vc = NoVectorCommitment<AkitaField>;
pub type Proof = JoltProof<AkitaScheme, Vc>;
pub type Preprocessing = JoltVerifierPreprocessing<AkitaScheme, Vc>;

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
}

pub enum ReplayEvent {
    PrepareStart,
    PrepareEnd,
    InnerStart(usize),
    InnerEnd(usize),
}

impl ProofPair {
    pub fn replay(
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
        let mut context = AkitaVerificationContext {
            mode,
            accumulator: Some(SetupClaimAccumulator::new(polynomial.clone()).with_advice(advice)),
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
                verify_with_context::<AkitaField, AkitaScheme, Vc, VerifierState<Blake2b512>>(
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
