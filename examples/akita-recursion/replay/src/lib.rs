#![cfg(feature = "akita")]
//! Verify one saved outer proof, consume its input batch, and export its direct-fold obligations.
use akita_types::setup_claims::{SetupBatchStatement, SetupClaim, SetupQuery};
use jolt_akita::deferred::{SetupClaimAccumulator, SetupClaimAdvice, SetupMode};
use jolt_akita::{AkitaError, AkitaField, AkitaScheme, AkitaVerificationContext};
use jolt_akita_recursion::{InnerProof, Preprocessing, Vc};
use jolt_transcript::{Blake2b512, VerifierState};
use serde::{Deserialize, Serialize};
use std::slice::Iter;

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
        let mut context = AkitaVerificationContext {
            mode,
            accumulator: Some(SetupClaimAccumulator::from_batch(0, statement).with_advice(advice)),
            cycle_marker: Some("outer_akita"),
        };
        observe(Event::PrepareEnd);
        observe(Event::JoltStart);
        let result = jolt_verifier::verifier::verify_with_context::<
            AkitaField,
            AkitaScheme,
            Vc,
            VerifierState<Blake2b512>,
        >(
            &self.preprocessing,
            &self.outer.io,
            &self.outer.proof,
            None,
            &mut context,
        );
        observe(Event::JoltEnd);
        result.map_err(|e| e.to_string())?;
        observe(Event::RebindStart);
        let polynomial = self
            .preprocessing
            .pcs_setup
            .setup_polynomial(self.output_setup_fields)
            .map_err(|e| e.to_string())?;
        let claims = context
            .accumulator
            .map_or_else(Vec::new, SetupClaimAccumulator::into_pending)
            .into_iter()
            .map(|c| SetupClaim::new(polynomial.clone(), c.query().clone(), c.value()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;
        let output = SetupBatchStatement::new(polynomial, 2, claims).map_err(|e| e.to_string())?;
        observe(Event::RebindEnd);
        Ok(output)
    }
}
pub struct Tape<'a>(pub Iter<'a, AkitaField>);
impl SetupClaimAdvice<AkitaField> for Tape<'_> {
    fn answer(&mut self, _: &SetupQuery<AkitaField>) -> Result<AkitaField, AkitaError> {
        self.0
            .next()
            .copied()
            .ok_or_else(|| AkitaError::InvalidInput("advice exhausted".into()))
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
/// A is deliberately absent: it is decoded from outer.io.outputs inside the guest.
#[derive(Serialize, Deserialize)]
pub struct OuterAdvice {
    pub outer: InnerProof,
    pub output_setup_fields: usize,
    pub answers: Vec<AkitaField>,
}
pub use jolt_akita_recursion::decode_exact;
