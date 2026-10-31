#![cfg(feature = "akita")]

use akita_types::setup_claims::{SetupBatchStatement, SetupQuery};
use common::jolt_device::JoltDevice;
use jolt_akita::deferred::{SetupClaimAccumulator, SetupClaimAdvice, SetupMode};
use jolt_akita::{AkitaError, AkitaField, AkitaNativeBatching, AkitaScheme};
use jolt_crypto::NoVectorCommitment;
use jolt_transcript::{Blake2b512, VerifierState};
use jolt_verifier::verifier::verify_with_akita_opening;
use jolt_verifier::{JoltProof, JoltVerifierPreprocessing};
use serde::{Deserialize, Serialize};
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
    AkitaStart(usize),
    AkitaEnd(usize),
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
        let mut accumulator = SetupClaimAccumulator::new(polynomial.clone()).with_advice(advice);
        observe(ReplayEvent::PrepareEnd);
        for (index, inner) in self.inner.iter().enumerate() {
            if inner.io.panic {
                return Err("inner guest panicked".into());
            }
            observe(ReplayEvent::InnerStart(index));
            let verified = verify_with_akita_opening::<
                AkitaField,
                AkitaScheme,
                Vc,
                VerifierState<Blake2b512>,
            >(
                &self.preprocessing,
                &inner.io,
                &inner.proof,
                None,
                |setup, auxiliary, main, proof, transcript| {
                    observe(ReplayEvent::AkitaStart(index));
                    let result = AkitaNativeBatching::verify_trace_batch_with_setup(
                        setup,
                        auxiliary,
                        main,
                        proof,
                        transcript,
                        Some(&mut accumulator),
                        mode,
                    );
                    observe(ReplayEvent::AkitaEnd(index));
                    result
                },
            );
            observe(ReplayEvent::InnerEnd(index));
            verified.map_err(|e| e.to_string())?;
        }
        SetupBatchStatement::new(polynomial, 2, accumulator.into_pending())
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

#[expect(
    clippy::expect_used,
    reason = "benchmark artifacts must decode; invalid guest inputs set the panic flag checked by the host"
)]
pub fn decode_pair(bytes: &[u8]) -> ProofPair {
    let (pair, consumed) =
        bincode::serde::decode_from_slice(bytes, bincode::config::standard()).expect("proof pair");
    assert_eq!(consumed, bytes.len(), "trailing proof pair bytes");
    pair
}
