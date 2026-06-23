//! Typed inputs consumed and outputs produced by stage 4 verification.

use jolt_field::Field;
use jolt_sumcheck::BatchedCommittedSumcheckConsistency;
use jolt_transcript::Transcript;
use serde::{Deserialize, Serialize};

use crate::stages::relations::{GetPoint, OpeningClaim, OutputClaims};
use crate::stages::zk::outputs::CommittedOutputClaimOutput;

use super::ram_val_check::{
    RamValCheckAdviceClaims, RamValCheckInitialEvaluation, RamValCheckOutputClaims,
};
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

/// The Fiat-Shamir challenges the verifier draws during stage 4: the two
/// per-relation batching gammas. (The batch's own sumcheck point and batching
/// coefficients are stage-local verification artifacts and are not propagated to
/// later stages.)
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4Challenges<F: Field> {
    pub registers_gamma: F,
    pub ram_val_check_gamma: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4ClearOutput<F: Field> {
    pub challenges: Stage4Challenges<F>,
    /// The produced stage-4 openings paired with their points (point + value)
    /// via the `OpeningClaim` cell. The opening points are derived from the
    /// batch's sumcheck point; pairing them with the values here lets later stages
    /// consume a ready `OpeningClaim` instead of re-joining a value with a
    /// separately-tracked point.
    pub output_claims: Stage4OutputClaims<OpeningClaim<F>>,
    pub ram_val_check_init: RamValCheckInitialEvaluation<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4ZkOutput<F: Field, C> {
    pub challenges: Stage4Challenges<F>,
    pub batch_consistency: BatchedCommittedSumcheckConsistency<F, C>,
    pub batch_output_claims: CommittedOutputClaimOutput<C>,
    pub ram_val_check_public_eval: F,
    /// The produced opening points (point-only cell), the ZK counterpart of the
    /// clear path's `output_claims`. Read through the same `*_point()` accessors.
    /// The advice / program-image leaves are absent in ZK (BlindFold carries those
    /// openings), so only the register and RAM value-check points are populated.
    pub output_points: Stage4OutputClaims<Vec<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage4Output<F: Field, C> {
    Clear(Stage4ClearOutput<F>),
    Zk(Stage4ZkOutput<F, C>),
}

impl<F: Field, C> Stage4Output<F, C> {
    pub fn clear(&self) -> Result<&Stage4ClearOutput<F>, crate::VerifierError> {
        match self {
            Self::Clear(output) => Ok(output),
            Self::Zk(_) => Err(crate::VerifierError::ExpectedClearProof { field: "stage4" }),
        }
    }

    pub fn zk(&self) -> Result<&Stage4ZkOutput<F, C>, crate::VerifierError> {
        match self {
            Self::Zk(output) => Ok(output),
            Self::Clear(_) => Err(crate::VerifierError::ExpectedCommittedProof { field: "stage4" }),
        }
    }
}
