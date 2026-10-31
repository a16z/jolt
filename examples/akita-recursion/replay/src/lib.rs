#![cfg(feature = "akita")]
//! Verify one saved outer proof, consume its input batch, and export its direct-fold obligations.
use akita_types::setup_claims::{SetupBatchStatement, SetupClaim};
use jolt_akita::deferred::{InheritedSetupClaims, SetupClaimAdvice, SetupMode};
use jolt_akita::{AkitaField, AkitaNativeBatching, AkitaScheme};
use jolt_akita_recursion::{InnerProof, Preprocessing, Vc};
use jolt_transcript::{Blake2b512, VerifierState};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct ExternalInput {
    pub preprocessing: Preprocessing,
    pub outer: InnerProof,
    pub inherited_setup_fields: usize,
    pub output_setup_fields: usize,
    pub answers: Vec<AkitaField>,
}
pub enum Event {
    PrepareStart,
    PrepareEnd,
    JoltStart,
    JoltEnd,
    AkitaStart,
    AkitaEnd,
    RebindStart,
    RebindEnd,
}
impl ExternalInput {
    pub fn inherited(&self) -> Result<SetupBatchStatement<AkitaField>, String> {
        let descriptor: Vec<u8> =
            postcard::from_bytes(&self.outer.io.outputs).map_err(|e| e.to_string())?;
        let polynomial = self
            .preprocessing
            .pcs_setup
            .setup_polynomial(self.inherited_setup_fields)
            .map_err(|e| e.to_string())?;
        SetupBatchStatement::from_descriptor_bytes(&descriptor, &polynomial)
            .map_err(|e| e.to_string())
    }
    pub fn replay(
        &self,
        mode: SetupMode,
        advice: &mut dyn SetupClaimAdvice<AkitaField>,
        mut observe: impl FnMut(Event),
    ) -> Result<SetupBatchStatement<AkitaField>, String> {
        if self.outer.io.panic {
            return Err("saved outer guest panicked".into());
        }
        observe(Event::PrepareStart);
        let statement = self.inherited()?;
        let mut accumulator = jolt_akita::deferred::SetupClaimAccumulator::from_batch(0, statement)
            .with_advice(advice);
        observe(Event::PrepareEnd);
        observe(Event::JoltStart);
        let result = jolt_verifier::verifier::verify_with_akita_opening::<
            AkitaField,
            AkitaScheme,
            Vc,
            VerifierState<Blake2b512>,
        >(
            &self.preprocessing,
            &self.outer.io,
            &self.outer.proof,
            None,
            |setup, aux, main, proof, transcript| {
                observe(Event::AkitaStart);
                let result = AkitaNativeBatching::verify_trace_batch_with_setup(
                    setup,
                    aux,
                    main,
                    proof,
                    transcript,
                    Some(&mut accumulator),
                    mode,
                );
                observe(Event::AkitaEnd);
                result
            },
        );
        observe(Event::JoltEnd);
        result.map_err(|e| e.to_string())?;
        observe(Event::RebindStart);
        let polynomial = self
            .preprocessing
            .pcs_setup
            .setup_polynomial(self.output_setup_fields)
            .map_err(|e| e.to_string())?;
        let claims = accumulator
            .into_pending()
            .into_iter()
            .map(|c| SetupClaim::new(polynomial.clone(), c.query().clone(), c.value()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;
        let output = SetupBatchStatement::new(polynomial, 2, claims).map_err(|e| e.to_string())?;
        observe(Event::RebindEnd);
        Ok(output)
    }
}
pub fn decode(bytes: &[u8]) -> Result<ExternalInput, String> {
    let (value, consumed) = bincode::serde::decode_from_slice(bytes, bincode::config::standard())
        .map_err(|e| e.to_string())?;
    if consumed != bytes.len() {
        return Err("trailing external input bytes".into());
    }
    Ok(value)
}

pub struct Tape<'a>(pub std::slice::Iter<'a, AkitaField>);
impl SetupClaimAdvice<AkitaField> for Tape<'_> {
    fn answer(
        &mut self,
        _: &akita_types::setup_claims::SetupQuery<AkitaField>,
    ) -> Result<AkitaField, jolt_akita::AkitaError> {
        self.0
            .next()
            .copied()
            .ok_or_else(|| jolt_akita::AkitaError::InvalidInput("advice exhausted".into()))
    }
}

pub use jolt_akita_recursion as pair_api;
/// The committed guest supplies the verifier; advice supplies only bundles and answers.
#[derive(Serialize, Deserialize)]
pub struct PairAdvice {
    pub inner: [InnerProof; 2],
    pub output_setup_fields: usize,
    pub answers: Vec<AkitaField>,
}
impl PairAdvice {
    pub fn with_verifier(self, preprocessing: Preprocessing) -> pair_api::ProofPair {
        pair_api::ProofPair {
            preprocessing,
            inner: self.inner,
            setup_field_len: self.output_setup_fields,
            answers: self.answers,
        }
    }
}
/// A is deliberately absent: it is decoded from outer.io.outputs inside the guest.
#[derive(Serialize, Deserialize)]
pub struct OuterAdvice {
    pub outer: InnerProof,
    pub output_setup_fields: usize,
    pub answers: Vec<AkitaField>,
}
impl OuterAdvice {
    pub fn with_verifier(
        self,
        preprocessing: Preprocessing,
        inherited_setup_fields: usize,
    ) -> ExternalInput {
        ExternalInput {
            preprocessing,
            outer: self.outer,
            inherited_setup_fields,
            output_setup_fields: self.output_setup_fields,
            answers: self.answers,
        }
    }
}
pub fn decode_exact<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T, String> {
    let (value, n) = bincode::serde::decode_from_slice(bytes, bincode::config::standard())
        .map_err(|e| e.to_string())?;
    if n != bytes.len() {
        return Err("trailing transport bytes".into());
    }
    Ok(value)
}
pub fn statement_from_output(
    preprocessing: &Preprocessing,
    fields: usize,
    outputs: &[u8],
) -> Result<SetupBatchStatement<AkitaField>, String> {
    let descriptor: Vec<u8> = postcard::from_bytes(outputs).map_err(|e| e.to_string())?;
    let polynomial = preprocessing
        .pcs_setup
        .setup_polynomial(fields)
        .map_err(|e| e.to_string())?;
    let statement = SetupBatchStatement::from_descriptor_bytes(&descriptor, &polynomial)
        .map_err(|e| e.to_string())?;
    InheritedSetupClaims::from_public_output(0, statement.clone(), &descriptor)
        .map_err(|e| e.to_string())?;
    Ok(statement)
}
