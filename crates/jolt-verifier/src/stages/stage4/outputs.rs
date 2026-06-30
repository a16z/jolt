//! Typed inputs consumed and outputs produced by stage 4 verification.

use jolt_field::Field;
use jolt_sumcheck::BatchedCommittedSumcheckConsistency;
use jolt_transcript::Transcript;

use crate::stages::relations::{GetPoint, OpeningClaim, OutputClaims, SumcheckBatch};
use crate::stages::zk::outputs::CommittedOutputClaimOutput;

use super::ram_val_check::{RamValCheck, RamValCheckInitialEvaluation};
use super::registers_read_write_checking::RegistersReadWriteChecking;

/// Source-of-truth for stage 4's sumcheck batch: the two instances in
/// Fiat-Shamir batch order (registers read-write, then RAM value-check).
/// `#[derive(SumcheckBatch)]` generates the `Stage4InputClaims<F, C>`,
/// `Stage4OutputClaims<F, C>`, and `Stage4Challenges<F>` aggregates — one field
/// per instance, in this declaration order.
///
/// The RAM value-check instance produces *more* openings than the register one:
/// besides its main `ram_ra`/`ram_inc`, it also stages the `Val_init` advice
/// contributions and (in committed program mode) the program-image contribution.
/// Those staged openings are folded into `RamValCheckOutputClaims`, so the
/// aggregate is genuinely one-field-per-instance. But the stage-4 Fiat-Shamir
/// append order interleaves them around the register openings — advice +
/// program-image come *before* the register openings, then `ram_ra`/`ram_inc`
/// come *after* — which a plain per-instance concatenation cannot express. The
/// stage therefore opts out of the generated `opening_values` /
/// `append_to_transcript` via `#[sumcheck_batch(custom_opening_values)]` and
/// supplies the exact interleaved order below.
#[derive(SumcheckBatch)]
#[sumcheck_batch(custom_opening_values)]
pub struct Stage4Sumchecks<F: Field> {
    pub registers_read_write: RegistersReadWriteChecking<F>,
    pub ram_val_check: RamValCheck<F>,
}

#[expect(
    clippy::mismatching_type_param_order,
    reason = "the cell param C is pinned to F for the value-only wire form; the second `F` is the cell, not a reordered field param"
)]
impl<F: Field> Stage4OutputClaims<F, F> {
    /// The produced opening claims in canonical (Fiat-Shamir) order, matching the
    /// prover's commitment (flush) order exactly: the `Val_init` advice openings,
    /// the committed program-image contribution, the register read-write openings,
    /// then the RAM value-check `ram_ra`/`ram_inc` openings. The advice and
    /// program-image openings are produced by the RAM value-check instance but are
    /// *appended first* (before the registers), so this is hand-written rather than
    /// a per-instance concatenation — see [`Stage4Sumchecks`].
    pub fn opening_values(&self) -> Vec<F> {
        let ram = &self.ram_val_check;
        ram.untrusted_advice
            .into_iter()
            .chain(ram.trusted_advice)
            .chain(ram.program_image)
            .chain(self.registers_read_write.opening_values())
            .chain([ram.ram_ra, ram.ram_inc])
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
        impl<F: Field> Stage4OutputClaims<F, $cell<F>> {
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage4ClearOutput<F: Field> {
    pub challenges: Stage4Challenges<F>,
    /// The produced stage-4 openings paired with their points (point + value)
    /// via the `OpeningClaim` cell. The opening points are derived from the
    /// batch's sumcheck point; pairing them with the values here lets later stages
    /// consume a ready `OpeningClaim` instead of re-joining a value with a
    /// separately-tracked point.
    pub output_claims: Stage4OutputClaims<F, OpeningClaim<F>>,
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
    pub output_points: Stage4OutputClaims<F, Vec<F>>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_claims::protocols::jolt::relations::ram::RamValCheckOutputClaims;
    use jolt_claims::protocols::jolt::relations::registers::RegistersReadWriteOutputClaims;
    use jolt_field::{Fr, FromPrimitiveInt};

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    fn registers_claims() -> RegistersReadWriteOutputClaims<Fr> {
        RegistersReadWriteOutputClaims {
            registers_val: fr(3),
            rs1_ra: fr(4),
            rs2_ra: fr(5),
            rd_wa: fr(6),
            rd_inc: fr(7),
        }
    }

    /// Locks the stage-4 Fiat-Shamir append order against silent drift: with no
    /// staged advice / program-image openings, the order is the five register
    /// openings then the two RAM value-check openings. A wrong order here silently
    /// breaks soundness, so it is pinned with distinct sentinels.
    #[test]
    fn opening_values_follow_canonical_order_without_advice() {
        let claims = Stage4OutputClaims::<Fr, Fr> {
            registers_read_write: registers_claims(),
            ram_val_check: RamValCheckOutputClaims {
                untrusted_advice: None,
                trusted_advice: None,
                program_image: None,
                ram_ra: fr(8),
                ram_inc: fr(9),
            },
        };

        assert_eq!(claims.opening_values(), (3..=9).map(fr).collect::<Vec<_>>());
    }

    /// The full interleaved order: advice (untrusted, trusted) and the
    /// program-image contribution come *first*, then the five register openings,
    /// then `ram_ra`/`ram_inc` last — exactly matching the prover's stage-4
    /// `pending_claims` flush order.
    #[test]
    fn opening_values_interleave_advice_then_registers_then_ram() {
        let claims = Stage4OutputClaims::<Fr, Fr> {
            registers_read_write: registers_claims(),
            ram_val_check: RamValCheckOutputClaims {
                untrusted_advice: Some(fr(1)),
                trusted_advice: Some(fr(2)),
                program_image: Some(fr(10)),
                ram_ra: fr(8),
                ram_inc: fr(9),
            },
        };

        assert_eq!(
            claims.opening_values(),
            vec![
                fr(1),  // untrusted advice
                fr(2),  // trusted advice
                fr(10), // program-image contribution
                fr(3),  // registers_val
                fr(4),  // rs1_ra
                fr(5),  // rs2_ra
                fr(6),  // rd_wa
                fr(7),  // rd_inc
                fr(8),  // ram_ra
                fr(9),  // ram_inc
            ]
        );
    }
}
