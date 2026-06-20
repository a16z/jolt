//! Typed inputs consumed and outputs produced by stage 5 verification.

use jolt_field::Field;
use jolt_sumcheck::BatchedCommittedSumcheckConsistency;
use jolt_transcript::Transcript;
use serde::{Deserialize, Serialize};

use crate::stages::relations::{GetPoint, OpeningClaim, OutputClaims};
use crate::stages::zk::outputs::CommittedOutputClaimOutput;

use super::instruction_read_raf::{reconstruct_r_address, InstructionReadRafOutputClaims};
use super::ram_ra_claim_reduction::RamRaClaimReductionOutputClaims;
use super::registers_val_evaluation::RegistersValEvaluationOutputClaims;

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
                reconstruct_r_address(&self.instruction_read_raf, self.instruction_r_cycle().len())
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

/// The Fiat-Shamir challenges the verifier draws during stage 5: the instruction
/// and RAM batching gammas. (The batch's own sumcheck point and batching
/// coefficients are stage-local verification artifacts and are not propagated to
/// later stages.)
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5Challenges<F: Field> {
    pub instruction_gamma: F,
    pub ram_gamma: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5ClearOutput<F: Field> {
    pub challenges: Stage5Challenges<F>,
    /// The produced stage-5 openings paired with their points (point + value) via
    /// the `OpeningClaim` cell. Later stages read each opening's value and point
    /// directly off these opening claims.
    pub output_claims: Stage5OutputClaims<OpeningClaim<F>>,
    /// The instruction read-RAF address point, materialized contiguously from the
    /// virtual-RA opening cells (which tile it as `chunk ++ r_cycle`). Stored
    /// because stage 6 re-chunks it by the committed-chunk width — a different
    /// split than the virtual-RA cells carry — so it needs a contiguous copy that
    /// downstream code can borrow.
    pub instruction_r_address: Vec<F>,
}

impl<F: Field> Stage5ClearOutput<F> {
    /// The instruction read-RAF cycle point, shared by the lookup-table-flag and
    /// RAF-flag openings.
    pub fn instruction_r_cycle(&self) -> &[F] {
        self.output_claims.instruction_r_cycle()
    }

    /// The reduced RAM-RA opening point (`address ++ cycle`, `log_k + log_t` vars).
    pub fn ram_reduced_opening_point(&self) -> &[F] {
        self.output_claims.ram_reduced_opening_point()
    }

    /// The register value-evaluation opening point (`REGISTER_ADDRESS_BITS + log_t`
    /// vars), shared by the `rd_inc` and `rd_wa` openings.
    pub fn registers_opening_point(&self) -> &[F] {
        self.output_claims.registers_opening_point()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5ZkOutput<F: Field, C> {
    pub challenges: Stage5Challenges<F>,
    pub batch_consistency: BatchedCommittedSumcheckConsistency<F, C>,
    pub batch_output_claims: CommittedOutputClaimOutput<C>,
    /// The produced opening points (point-only cell), the ZK counterpart of the
    /// clear path's `output_claims`. Read through the same `*_point()` accessors.
    pub output_points: Stage5OutputClaims<Vec<F>>,
    /// The contiguous instruction address point, stored (rather than reconstructed
    /// from `output_points` on demand) so stage 6 can borrow it — the per-chunk
    /// virtual-RA cells don't hold it contiguously. Mirrors `Stage5ClearOutput`.
    pub instruction_r_address: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage5Output<F: Field, C> {
    Clear(Stage5ClearOutput<F>),
    Zk(Stage5ZkOutput<F, C>),
}

impl<F: Field, C> Stage5Output<F, C> {
    pub fn clear(&self) -> Result<&Stage5ClearOutput<F>, crate::VerifierError> {
        match self {
            Self::Clear(output) => Ok(output),
            Self::Zk(_) => Err(crate::VerifierError::ExpectedClearProof { field: "stage5" }),
        }
    }

    pub fn zk(&self) -> Result<&Stage5ZkOutput<F, C>, crate::VerifierError> {
        match self {
            Self::Zk(output) => Ok(output),
            Self::Clear(_) => Err(crate::VerifierError::ExpectedCommittedProof { field: "stage5" }),
        }
    }
}
