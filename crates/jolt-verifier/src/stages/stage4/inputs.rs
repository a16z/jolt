//! Typed inputs consumed by stage 4.

use jolt_field::Field;
use jolt_verifier_derive::OutputClaims;
use serde::{Deserialize, Serialize};

use crate::stages::{
    stage2::{Stage2ClearOutput, Stage2Output, Stage2ZkOutput},
    stage3::{Stage3ClearOutput, Stage3Output, Stage3ZkOutput},
};

use super::ram_val_check::{RamValCheckAdviceClaims, RamValCheckOutputClaims};
use super::registers_read_write_checking::RegistersReadWriteOutputClaims;

#[derive(Clone, Copy)]
pub enum Deps<'a, F: Field, C> {
    Clear {
        stage2: &'a Stage2ClearOutput<F>,
        stage3: &'a Stage3ClearOutput<F>,
    },
    Zk {
        stage2: &'a Stage2ZkOutput<F, C>,
        stage3: &'a Stage3ZkOutput<F, C>,
    },
}

pub fn deps<'a, F: Field, C>(
    stage2: &'a Stage2Output<F, C>,
    stage3: &'a Stage3Output<F, C>,
) -> Result<Deps<'a, F, C>, crate::VerifierError> {
    match (stage2, stage3) {
        (Stage2Output::Clear(stage2), Stage3Output::Clear(stage3)) => {
            Ok(Deps::Clear { stage2, stage3 })
        }
        (Stage2Output::Zk(stage2), Stage3Output::Zk(stage3)) => Ok(Deps::Zk { stage2, stage3 }),
        (Stage2Output::Clear(_), Stage3Output::Zk(_)) => {
            Err(crate::VerifierError::ExpectedClearProof { field: "stage3" })
        }
        (Stage2Output::Zk(_), Stage3Output::Clear(_)) => {
            Err(crate::VerifierError::ExpectedCommittedProof { field: "stage3" })
        }
    }
}

/// The stage 4 produced opening claims, declared in canonical (Fiat-Shamir)
/// order: the `Val_init` advice openings, the committed program-image
/// contribution, the register read-write openings, then the RAM value-check
/// openings. The derived `OutputClaims` impl single-sources the transcript append
/// order and the opening count/values from this declaration order, so they cannot
/// drift from one another. Generic over the cell (`F` on the wire).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, OutputClaims)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
#[relation(RamValCheck)]
pub struct Stage4Claims<C> {
    pub advice: RamValCheckAdviceClaims<C>,
    /// Staged `ProgramImageInitContributionRw` scalar; present only in committed
    /// program mode.
    #[opening(ProgramImageInitContributionRw)]
    pub program_image_contribution: Option<C>,
    pub registers_read_write: RegistersReadWriteOutputClaims<C>,
    pub ram_val_check: RamValCheckOutputClaims<C>,
}
