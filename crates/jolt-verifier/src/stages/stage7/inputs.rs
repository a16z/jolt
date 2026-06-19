//! Typed inputs consumed by stage 7.

use jolt_field::Field;
use jolt_transcript::Transcript;
use serde::{Deserialize, Serialize};

use crate::stages::relations::OutputClaims;
use crate::stages::{
    stage4::{Stage4ClearOutput, Stage4Output},
    stage6::{Stage6ClearOutput, Stage6Output, Stage6ZkOutput},
};

use super::advice_address_phase::AdviceAddressPhaseOutputClaims;
use super::committed_reduction_address_phase::{
    BytecodeReductionAddressPhaseOutputClaims, ProgramImageReductionAddressPhaseOutputClaims,
};
use super::hamming_weight_claim_reduction::HammingWeightClaimReductionOutputClaims;

#[derive(Clone, Copy)]
pub enum Deps<'a, F: Field, C> {
    Clear {
        stage4: &'a Stage4ClearOutput<F>,
        stage6: &'a Stage6ClearOutput<F>,
    },
    Zk {
        stage6: &'a Stage6ZkOutput<F, C>,
    },
}

pub fn deps<'a, F: Field, C>(
    stage4: &'a Stage4Output<F, C>,
    stage6: &'a Stage6Output<F, C>,
) -> Result<Deps<'a, F, C>, crate::VerifierError> {
    match (stage4, stage6) {
        (Stage4Output::Clear(stage4), Stage6Output::Clear(stage6)) => {
            Ok(Deps::Clear { stage4, stage6 })
        }
        (Stage4Output::Zk(_), Stage6Output::Zk(stage6)) => Ok(Deps::Zk { stage6 }),
        (Stage4Output::Clear(_), Stage6Output::Zk(_)) => {
            Err(crate::VerifierError::ExpectedClearProof { field: "stage6" })
        }
        (Stage4Output::Zk(_), Stage6Output::Clear(_)) => {
            Err(crate::VerifierError::ExpectedCommittedProof { field: "stage6" })
        }
    }
}

/// The stage 7 produced opening claims, declared in canonical (Fiat-Shamir)
/// order: the hamming-weight reduced RA openings (instruction, bytecode, RAM),
/// the trusted/untrusted advice address-phase openings, the committed bytecode
/// chunk openings, then the program-image opening — each present only when its
/// phase ran. [`opening_values`](Self::opening_values) and
/// [`append_to_transcript`](Self::append_to_transcript) single-source the append
/// order from the per-relation declaration orders. Generic over the cell (`F` on
/// the wire).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
pub struct Stage7OutputClaims<C> {
    pub hamming_weight_claim_reduction: HammingWeightClaimReductionOutputClaims<C>,
    pub advice_address_phase: AdviceAddressPhaseOutputClaims<C>,
    /// Final `BytecodeChunk(i)` claims from the committed-bytecode reduction's
    /// address phase; present only when that phase runs.
    pub bytecode_address_phase: Option<BytecodeReductionAddressPhaseOutputClaims<C>>,
    /// Final `ProgramImageInit` claim from the program-image reduction's address
    /// phase; present only when that phase runs.
    pub program_image_address_phase: Option<ProgramImageReductionAddressPhaseOutputClaims<C>>,
}

impl<F: Field> Stage7OutputClaims<F> {
    /// The produced opening claims in canonical (Fiat-Shamir) order, single-sourced
    /// from the per-relation declaration orders.
    pub fn opening_values(&self) -> Vec<F> {
        let mut values = self.hamming_weight_claim_reduction.opening_values();
        values.extend(self.advice_address_phase.opening_values());
        if let Some(claims) = &self.bytecode_address_phase {
            values.extend(claims.opening_values());
        }
        if let Some(claims) = &self.program_image_address_phase {
            values.extend(claims.opening_values());
        }
        values
    }

    /// Append every produced opening to the transcript in canonical order, each
    /// under the `b"opening_claim"` label, matching the prover's commitment order.
    pub fn append_to_transcript<T: Transcript<Challenge = F>>(&self, transcript: &mut T) {
        for value in self.opening_values() {
            transcript.append_labeled(b"opening_claim", &value);
        }
    }
}
