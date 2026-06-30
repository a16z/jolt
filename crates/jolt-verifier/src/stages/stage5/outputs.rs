//! Typed inputs consumed and outputs produced by stage 5 verification.

use jolt_field::Field;
use jolt_sumcheck::BatchedCommittedSumcheckConsistency;

use crate::stages::relations::{GetPoint, OpeningClaim, SumcheckBatch};
use crate::stages::zk::outputs::CommittedOutputClaimOutput;

use super::instruction_read_raf::{reconstruct_r_address, InstructionReadRaf};
use super::ram_ra_claim_reduction::RamRaClaimReduction;
use super::registers_val_evaluation::RegistersValEvaluation;

/// Source-of-truth for stage 5's sumcheck batch: the three instances in
/// Fiat-Shamir batch order (instruction read-RAF, RAM-RA reduction, register
/// value-evaluation). `#[derive(SumcheckBatch)]` generates the
/// `Stage5InputClaims<F, C>`, `Stage5OutputClaims<F, C>`, and `Stage5Challenges<F>`
/// aggregates — one field per instance, in this declaration order — plus the
/// `Stage5OutputClaims` Fiat-Shamir opening plumbing (`opening_values` /
/// `append_to_transcript`). The field order is load-bearing: it fixes the canonical
/// opening order absorbed into the transcript, which must match the prover's
/// commitment order.
#[derive(SumcheckBatch)]
pub struct Stage5Sumchecks<F: Field> {
    pub instruction_read_raf: InstructionReadRaf<F>,
    pub ram_ra_claim_reduction: RamRaClaimReduction<F>,
    pub registers_val_evaluation: RegistersValEvaluation<F>,
}

/// The shared opening-point accessors, generated for each concrete cell
/// (`OpeningClaim<F>` on the clear path, `Vec<F>` for the ZK point-only form) so
/// both expose the same inherent `*_point()` API. A single `impl<C: GetPoint<F>>`
/// can't express this — `F` would be unconstrained by the self type.
macro_rules! stage5_point_accessors {
    ($cell:ident) => {
        impl<F: Field> Stage5OutputClaims<F, $cell<F>> {
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage5ClearOutput<F: Field> {
    pub challenges: Stage5Challenges<F>,
    /// The produced stage-5 openings paired with their points (point + value) via
    /// the `OpeningClaim` cell. Later stages read each opening's value and point
    /// directly off these opening claims.
    pub output_claims: Stage5OutputClaims<F, OpeningClaim<F>>,
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
    pub output_points: Stage5OutputClaims<F, Vec<F>>,
    /// The contiguous instruction address point, stored (rather than reconstructed
    /// from `output_points` on demand) so stage 6 can borrow it — the per-chunk
    /// virtual-RA cells don't hold it contiguously. Mirrors `Stage5ClearOutput`.
    pub instruction_r_address: Vec<F>,
}

// The clear variant carries the located opening claims (point + value) that
// later stages read on the hot path; the ZK variant carries the committed
// consistency and output-claim commitments. Boxing the common clear variant to
// shrink the rarer ZK one would add indirection to every clear-path access.
#[expect(
    clippy::large_enum_variant,
    reason = "clear variant holds the located opening claims read on the hot path; boxing it would penalize the common case"
)]
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

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_claims::protocols::jolt::relations::instruction::InstructionReadRafOutputClaims;
    use jolt_claims::protocols::jolt::relations::ram::RamRaClaimReductionOutputClaims;
    use jolt_claims::protocols::jolt::relations::registers::RegistersValEvaluationOutputClaims;
    use jolt_field::{Fr, FromPrimitiveInt};

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    /// Locks the stage-5 Fiat-Shamir append order against silent drift: the
    /// instruction read-RAF openings, then the RAM-RA reduced opening, then the
    /// register value-evaluation openings, each member single-sourcing its own
    /// per-field order from its `OutputClaims` derive. A wrong batch order here
    /// silently breaks soundness, so it is pinned with distinct sentinels.
    #[test]
    fn opening_values_follow_canonical_order() {
        let claims = Stage5OutputClaims::<Fr, Fr> {
            instruction_read_raf: InstructionReadRafOutputClaims {
                lookup_table_flags: vec![fr(1), fr(2)],
                instruction_ra: vec![fr(3), fr(4)],
                instruction_raf_flag: fr(5),
            },
            ram_ra_claim_reduction: RamRaClaimReductionOutputClaims { ram_ra: fr(6) },
            registers_val_evaluation: RegistersValEvaluationOutputClaims {
                rd_inc: fr(7),
                rd_wa: fr(8),
            },
        };

        assert_eq!(claims.opening_values(), (1..=8).map(fr).collect::<Vec<_>>());
    }
}
