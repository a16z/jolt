//! Typed inputs consumed by stage 5.

use jolt_field::Field;
use jolt_transcript::Transcript;
use serde::{Deserialize, Serialize};

use crate::stages::relations::{GetPoint, OpeningClaim, OutputClaims};
use crate::stages::{
    stage2::{Stage2Output, Stage2ZkOutput},
    stage4::{Stage4Output, Stage4ZkOutput},
};

use super::instruction_read_raf::{reconstruct_r_address, InstructionReadRafOutputClaims};
use super::ram_ra_claim_reduction::RamRaClaimReductionOutputClaims;
use super::registers_val_evaluation::RegistersValEvaluationOutputClaims;

#[derive(Clone, Copy)]
pub enum Deps<'a, F: Field, C> {
    Clear {
        stage2: &'a crate::stages::stage2::Stage2ClearOutput<F>,
        stage4: &'a crate::stages::stage4::Stage4ClearOutput<F>,
    },
    Zk {
        stage2: &'a Stage2ZkOutput<F, C>,
        stage4: &'a Stage4ZkOutput<F, C>,
    },
}

pub fn deps<'a, F: Field, C>(
    stage2: &'a Stage2Output<F, C>,
    stage4: &'a Stage4Output<F, C>,
) -> Result<Deps<'a, F, C>, crate::VerifierError> {
    match (stage2, stage4) {
        (Stage2Output::Clear(stage2), Stage4Output::Clear(stage4)) => {
            Ok(Deps::Clear { stage2, stage4 })
        }
        (Stage2Output::Zk(stage2), Stage4Output::Zk(stage4)) => Ok(Deps::Zk { stage2, stage4 }),
        (Stage2Output::Clear(_), Stage4Output::Zk(_)) => {
            Err(crate::VerifierError::ExpectedClearProof { field: "stage4" })
        }
        (Stage2Output::Zk(_), Stage4Output::Clear(_)) => {
            Err(crate::VerifierError::ExpectedCommittedProof { field: "stage4" })
        }
    }
}

/// The stage 5 produced opening claims, generic over the cell (`F` on the wire,
/// `Vec<F>` for derived points, `OpeningClaim<F>` (point + value) on the clear path).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
pub struct Stage5OutputClaims<C> {
    pub instruction_read_raf: InstructionReadRafOutputClaims<C>,
    pub ram_ra_claim_reduction: RamRaClaimReductionOutputClaims<C>,
    pub registers_val_evaluation: RegistersValEvaluationOutputClaims<C>,
}

impl<F: Field> Stage5OutputClaims<F> {
    /// The produced opening claims in canonical (Fiat-Shamir) order: the
    /// instruction read-RAF openings, the RAM-RA reduced opening, then the
    /// register value-evaluation openings. Single-sources [`append_to_transcript`]
    /// and the prover's output-claim values from the per-relation declaration
    /// orders.
    ///
    /// [`append_to_transcript`]: Self::append_to_transcript
    pub fn opening_values(&self) -> Vec<F> {
        self.instruction_read_raf
            .opening_values()
            .into_iter()
            .chain(self.ram_ra_claim_reduction.opening_values())
            .chain(self.registers_val_evaluation.opening_values())
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
macro_rules! stage5_point_accessors {
    ($cell:ident) => {
        impl<F: Field> Stage5OutputClaims<$cell<F>> {
            /// The instruction read-RAF cycle point (shared by the lookup-table-flag
            /// and RAF-flag openings).
            pub fn instruction_r_cycle(&self) -> &[F] {
                self.instruction_read_raf.instruction_raf_flag.point()
            }

            /// The contiguous instruction address point, reconstructed from the
            /// virtual-RA opening cells (each is `chunk ++ r_cycle`).
            pub fn instruction_r_address(&self) -> Vec<F> {
                reconstruct_r_address(
                    &self.instruction_read_raf,
                    self.instruction_r_cycle().len(),
                )
            }

            /// The reduced RAM-RA opening point (`address ++ cycle`).
            pub fn ram_reduced_opening_point(&self) -> &[F] {
                self.ram_ra_claim_reduction.ram_ra.point()
            }

            /// The register value-evaluation opening point (shared by `rd_inc`/`rd_wa`).
            pub fn registers_opening_point(&self) -> &[F] {
                self.registers_val_evaluation.rd_inc.point()
            }
        }
    };
}

stage5_point_accessors!(OpeningClaim);
stage5_point_accessors!(Vec);
