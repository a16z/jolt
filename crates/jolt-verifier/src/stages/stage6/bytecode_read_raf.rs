//! The stage 6 bytecode read-RAF sumcheck instances.
//!
//! This module currently provides the stage-6a **address phase**. The cycle phase
//! (stage 6b) is deferred to the wiring step: its expected output depends on the
//! bytecode-table public values (`read_raf_public_values`, which needs the
//! preprocessing bytecode rows) and, in committed-program mode, consumes the
//! staged `BytecodeValStage` openings inside its *output* expression — both of
//! which are cleanest to finalize against the live `verify()`/prover wiring.
//!
//! The address phase binds the `log_k` address variables. Its input claim is the
//! gamma-folded bind of the entire prior proof (every stage-1..5 opening plus the
//! two PC claims); that 25-opening formula already lives in the single-sourced
//! [`stage6_bytecode_read_raf_address_input`] helper, so this relation takes the
//! precomputed value and overrides [`SumcheckInstance::input_claim`] rather than
//! restating the bind as a 25-field `InputClaims`. Its output is the staged
//! `BytecodeReadRafAddrClaim` intermediate (consumed by the cycle phase) followed,
//! in committed mode, by the `BytecodeValStage` openings.
//!
//! [`stage6_bytecode_read_raf_address_input`]: super::verify::stage6_bytecode_read_raf_address_input

use core::marker::PhantomData;

use jolt_claims::protocols::jolt::{
    formulas::bytecode::{self, BytecodeReadRafDimensions},
    JoltOpeningId, JoltRelationClaims,
};
use jolt_field::Field;
use jolt_verifier_derive::OutputClaims;
use serde::{Deserialize, Serialize};

use crate::stages::relations::{GetPoint, InputClaims, OpeningClaim, SumcheckInstance};
use crate::VerifierError;

/// The address-phase produced openings: the `BytecodeReadRafAddrClaim`
/// intermediate, plus (committed-program mode only) the staged `BytecodeValStage`
/// openings. In full-program mode `val_stages` is empty.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(BytecodeReadRaf)]
pub struct BytecodeReadRafAddressPhaseOutputClaims<C> {
    #[opening(BytecodeReadRafAddrClaim)]
    pub intermediate: C,
    #[opening(BytecodeValStage)]
    pub val_stages: Vec<C>,
}

/// The address phase's input claim is the gamma-folded prior-proof bind, supplied
/// precomputed (see the module docs), so it consumes no openings through the
/// generic path.
pub struct BytecodeReadRafAddressPhaseInputClaims<C> {
    _cell: PhantomData<C>,
}

impl<C> Default for BytecodeReadRafAddressPhaseInputClaims<C> {
    fn default() -> Self {
        Self { _cell: PhantomData }
    }
}

impl<F: Field> BytecodeReadRafAddressPhaseInputClaims<OpeningClaim<F>> {
    pub fn from_upstream() -> Self {
        Self::default()
    }
}

impl<F: Field> InputClaims<F> for BytecodeReadRafAddressPhaseInputClaims<OpeningClaim<F>> {
    fn resolve_input(&self, _id: &JoltOpeningId) -> Option<F> {
        None
    }
}

pub struct BytecodeReadRafAddressPhase<F: Field> {
    claims: JoltRelationClaims<F>,
    /// The gamma-folded prior-proof bind, precomputed by
    /// `stage6_bytecode_read_raf_address_input`.
    input_claim: F,
    /// `NUM_BYTECODE_VAL_STAGES` in committed-program mode, else 0.
    num_val_stages: usize,
}

impl<F: Field> BytecodeReadRafAddressPhase<F> {
    pub fn new(dimensions: BytecodeReadRafDimensions, input_claim: F, num_val_stages: usize) -> Self {
        Self {
            claims: bytecode::read_raf_address_phase(dimensions),
            input_claim,
            num_val_stages,
        }
    }
}

impl<F: Field> SumcheckInstance<F> for BytecodeReadRafAddressPhase<F> {
    type Inputs<C> = BytecodeReadRafAddressPhaseInputClaims<C>;
    type Outputs<C> = BytecodeReadRafAddressPhaseOutputClaims<C>;

    fn sumcheck_relation(&self) -> &JoltRelationClaims<F> {
        &self.claims
    }

    fn input_claim(
        &self,
        _inputs: &BytecodeReadRafAddressPhaseInputClaims<OpeningClaim<F>>,
    ) -> Result<F, VerifierError> {
        Ok(self.input_claim)
    }

    fn derive_opening_points<C: GetPoint<F>>(
        &self,
        sumcheck_point: &[F],
        _inputs: &BytecodeReadRafAddressPhaseInputClaims<C>,
    ) -> Result<BytecodeReadRafAddressPhaseOutputClaims<Vec<F>>, VerifierError> {
        // `bytecode_r_address` is the reversed address sumcheck point; the
        // intermediate and every staged Val column open there.
        let r_address = sumcheck_point.iter().rev().copied().collect::<Vec<_>>();
        Ok(BytecodeReadRafAddressPhaseOutputClaims {
            intermediate: r_address.clone(),
            val_stages: vec![r_address; self.num_val_stages],
        })
    }
}
