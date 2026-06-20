//! Typed inputs consumed by stage 4.

use jolt_field::Field;
use jolt_transcript::Transcript;
use serde::{Deserialize, Serialize};

use crate::stages::relations::{GetPoint, OpeningClaim, OutputClaims};

use super::ram_val_check::{RamValCheckAdviceClaims, RamValCheckOutputClaims};
use super::registers_read_write_checking::RegistersReadWriteOutputClaims;

/// The stage 4 produced opening claims, declared in canonical (Fiat-Shamir)
/// order: the `Val_init` advice openings, the committed program-image
/// contribution, the register read-write openings, then the RAM value-check
/// openings. [`opening_values`](Self::opening_values) and
/// [`append_to_transcript`](Self::append_to_transcript) single-source the append
/// order from this declaration order. Generic over the cell (`F` on the wire).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
pub struct Stage4OutputClaims<C> {
    pub advice: RamValCheckAdviceClaims<C>,
    /// Staged `ProgramImageInitContributionRw` scalar; present only in committed
    /// program mode.
    pub program_image_contribution: Option<C>,
    pub registers_read_write: RegistersReadWriteOutputClaims<C>,
    pub ram_val_check: RamValCheckOutputClaims<C>,
}

impl<F: Field> Stage4OutputClaims<F> {
    /// The produced opening claims in canonical (Fiat-Shamir) order: the
    /// `Val_init` advice openings, the program-image contribution, the register
    /// read-write openings, then the RAM value-check openings. Single-sources
    /// [`append_to_transcript`](Self::append_to_transcript) and the prover's
    /// output-claim values from the per-relation declaration orders.
    pub fn opening_values(&self) -> Vec<F> {
        self.advice
            .opening_values()
            .into_iter()
            .chain(self.program_image_contribution)
            .chain(self.registers_read_write.opening_values())
            .chain(self.ram_val_check.opening_values())
            .collect()
    }

    /// Append every produced opening to the transcript in canonical order, each
    /// under the `b"opening_claim"` label, matching the prover's commitment order.
    pub fn append_to_transcript<T: Transcript<Challenge = F>>(&self, transcript: &mut T) {
        for value in self.opening_values() {
            transcript.append_labeled(b"opening_claim", &value);
        }
    }
}

/// The shared opening-point accessors, generated for each concrete cell
/// (`OpeningClaim<F>` on the clear path, `Vec<F>` for the ZK point-only form) so
/// both expose the same inherent `*_point()` API. A single `impl<C: GetPoint<F>>`
/// can't express this — `F` would be unconstrained by the self type.
macro_rules! stage4_point_accessors {
    ($cell:ident) => {
        impl<F: Field> Stage4OutputClaims<$cell<F>> {
            /// The register read-write opening point (shared by all five register
            /// openings).
            pub fn registers_read_write_point(&self) -> &[F] {
                self.registers_read_write.registers_val.point()
            }

            /// The RAM value-check opening point (shared by `ram_ra`/`ram_inc`).
            pub fn ram_val_check_point(&self) -> &[F] {
                self.ram_val_check.ram_ra.point()
            }
        }
    };
}

stage4_point_accessors!(OpeningClaim);
stage4_point_accessors!(Vec);
