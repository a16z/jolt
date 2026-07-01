//! Typed inputs consumed and outputs produced by stage 2 verification.

use jolt_claims::protocols::jolt::{formulas::instruction, JoltRelationId};
use jolt_field::Field;
use jolt_poly::{Point, HIGH_TO_LOW};
use jolt_sumcheck::{BatchedCommittedSumcheckConsistency, CommittedSumcheckConsistency};
use jolt_transcript::FsAbsorb;
use serde::{Deserialize, Serialize};

use crate::stages::relations::{GetPoint, OpeningClaim, OutputClaims};
use crate::stages::stage1::Stage1ClearOutput;
use crate::stages::zk::outputs::CommittedOutputClaimOutput;
use crate::VerifierError;

pub use super::instruction_claim_reduction::InstructionClaimReductionOutputClaims;
pub use super::product_remainder::ProductRemainderOutputClaims;
pub use super::ram_output_check::RamOutputCheckOutputClaims;
pub use super::ram_raf_evaluation::RamRafEvaluationOutputClaims;
pub use super::ram_read_write_checking::RamReadWriteOutputClaims;

/// Stage 1 outputs that feed the stage 2 product uni-skip input claim. Extracted
/// into a typed value so the prover and verifier derive the same input claim from
/// the shared [`product_uniskip_input_claim`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage2ProductUniSkipInputValues<F: Field> {
    pub product: F,
    pub should_branch: F,
    pub should_jump: F,
}

impl<F: Field> Stage2ProductUniSkipInputValues<F> {
    pub fn from_stage1(stage1: &Stage1ClearOutput<F>) -> Self {
        Self {
            product: stage1.outer.product,
            should_branch: stage1.outer.should_branch,
            should_jump: stage1.outer.should_jump,
        }
    }
}

/// Combines the stage 1 product values against the uni-skip Lagrange `weights`
/// (derived from `tau_high`) into the stage 2 product uni-skip input claim.
pub fn product_uniskip_input_claim<F: Field>(
    values: Stage2ProductUniSkipInputValues<F>,
    weights: &[F],
) -> Result<F, VerifierError> {
    let [product, should_branch, should_jump, rest @ ..] = weights else {
        return Err(stage2_product_public_input_failed(format!(
            "Stage 2 product uni-skip expected at least 3 weights, got {}",
            weights.len()
        )));
    };
    let claim = *product * values.product
        + *should_branch * values.should_branch
        + *should_jump * values.should_jump;

    if !rest.is_empty() {
        return Err(stage2_product_public_input_failed(format!(
            "Stage 2 product uni-skip expected 3 weights, got {}",
            weights.len()
        )));
    }
    Ok(claim)
}

fn stage2_product_public_input_failed(reason: String) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::SpartanProductVirtualization,
        reason,
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Stage2OutputClaims<F: Field> {
    pub product_uniskip_output_claim: F,
    pub batch_outputs: Stage2BatchOutputClaims<F>,
}

/// The produced stage 2 batch openings, one per-relation `OutputClaims` struct.
/// Generic over the cell: `F` is the serialized wire form (value only), and
/// `OpeningClaim<F>` is the clear opening-claim form (point + value) propagated to
/// later stages — mirroring stage 3/4's `StageNOutputClaims<OpeningClaim<F>>`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
pub struct Stage2BatchOutputClaims<C> {
    pub ram_read_write: RamReadWriteOutputClaims<C>,
    pub product_remainder: ProductRemainderOutputClaims<C>,
    pub instruction_claim_reduction: InstructionClaimReductionOutputClaims<C>,
    pub ram_raf_evaluation: RamRafEvaluationOutputClaims<C>,
    pub ram_output_check: RamOutputCheckOutputClaims<C>,
}

impl<F: Field> Stage2BatchOutputClaims<F> {
    /// The stage 2 batch produced opening claims in canonical (Fiat-Shamir) order:
    /// the RAM read-write openings, the eight product-remainder openings, the two
    /// reduced instruction lookup operands (the other reduced openings alias the
    /// product-remainder ones and are not re-absorbed), then the RAM RAF and output
    /// openings. Single-sources [`append_to_transcript`](Self::append_to_transcript)
    /// and the prover's batch output-claim values.
    pub fn opening_values(&self) -> Vec<F> {
        // Full relations delegate to their derived `opening_values()` so the
        // per-field order is single-sourced from the `OutputClaims` derive. Only
        // the two reduced instruction lookup operands are listed explicitly: the
        // reduction's other openings alias the product-remainder ones and are not
        // re-absorbed (see `validate`).
        self.ram_read_write
            .opening_values()
            .into_iter()
            .chain(self.product_remainder.opening_values())
            .chain([
                self.instruction_claim_reduction.left_lookup_operand,
                self.instruction_claim_reduction.right_lookup_operand,
            ])
            .chain(self.ram_raf_evaluation.opening_values())
            .chain(self.ram_output_check.opening_values())
            .collect()
    }

    /// Append every batch opening to the transcript in canonical order, matching
    /// the prover's commitment order.
    pub fn append_to_transcript<T: FsAbsorb>(&self, transcript: &mut T) {
        for value in self.opening_values() {
            transcript.absorb_field(&value);
        }
    }
}

/// The shared per-relation opening-point accessors, generated for each concrete
/// cell (`OpeningClaim<F>` on the clear path, `Vec<F>` for the ZK point-only form)
/// so both expose the same inherent `*_point()` API. A single `impl<C: GetPoint<F>>`
/// can't express this — `F` would be unconstrained by the self type.
macro_rules! stage2_batch_point_accessors {
    ($cell:ident) => {
        impl<F: Field> Stage2BatchOutputClaims<$cell<F>> {
            /// The RAM read-write opening point (shared by `val`/`ra`/`inc`).
            pub fn ram_read_write_point(&self) -> &[F] {
                self.ram_read_write.val.point()
            }

            /// The product-remainder opening point (shared by all eight openings).
            pub fn product_remainder_point(&self) -> &[F] {
                self.product_remainder.left_instruction_input.point()
            }

            /// The reduced instruction-claim opening point (shared by all five
            /// openings).
            pub fn instruction_claim_reduction_point(&self) -> &[F] {
                self.instruction_claim_reduction.left_lookup_operand.point()
            }

            /// The RAM RAF opening point (`[r_address ‖ tau_low]`).
            pub fn ram_raf_evaluation_point(&self) -> &[F] {
                self.ram_raf_evaluation.ram_ra.point()
            }

            /// The RAM output-check opening point (`r_address`).
            pub fn ram_output_check_point(&self) -> &[F] {
                self.ram_output_check.val_final.point()
            }
        }
    };
}

stage2_batch_point_accessors!(OpeningClaim);
stage2_batch_point_accessors!(Vec);

impl<F: Field> Stage2BatchOutputClaims<OpeningClaim<F>> {
    /// Enforce the cross-relation aliases between the product-remainder openings and
    /// the reduced instruction-claim openings: when a reduced opening is present it
    /// must share the product-remainder opening's point and value. Downstream
    /// consumers rely on these aliases when they fall back to the product remainder —
    /// the stage-5 instruction read-RAF wiring (`lookup_output`) and the stage-3
    /// instruction-input virtualization (`left`/`right_instruction_input`) — so the
    /// stage-2 verifier checks them here (mirroring
    /// [`Stage3OutputClaims::validate`](crate::stages::stage3::Stage3OutputClaims::validate))
    /// rather than each consumer re-checking. Errors preserve the opening and
    /// relation ids those consumers reported.
    pub fn validate(&self) -> Result<(), VerifierError> {
        let [(lookup_output_reduced, lookup_output_product)] =
            instruction::read_raf_consistency_openings();
        let [(left_reduced, left_product), (right_reduced, right_product)] =
            instruction::input_virtualization_consistency_openings();

        // Every reduced instruction opening shares the product-remainder opening
        // point; if the points disagree the reduced openings cannot alias the
        // product ones.
        if self.product_remainder_point() != self.instruction_claim_reduction_point() {
            return Err(VerifierError::StageClaimOpeningMismatch {
                stage: JoltRelationId::InstructionReadRaf,
                left: lookup_output_reduced,
                right: lookup_output_product,
            });
        }

        // `lookup_output`: stage-5 instruction read-RAF fallback to the product remainder.
        let product_lookup_output = self.product_remainder.lookup_output.value;
        let reduced_lookup_output = self
            .instruction_claim_reduction
            .lookup_output
            .as_ref()
            .map_or(product_lookup_output, |claim| claim.value);
        if reduced_lookup_output != product_lookup_output {
            return Err(VerifierError::StageClaimOpeningMismatch {
                stage: JoltRelationId::InstructionReadRaf,
                left: lookup_output_reduced,
                right: lookup_output_product,
            });
        }

        // `left`/`right_instruction_input`: stage-3 instruction-input virtualization
        // fallback to the product remainder.
        let product_left = self.product_remainder.left_instruction_input.value;
        let product_right = self.product_remainder.right_instruction_input.value;
        let reduced_left = self
            .instruction_claim_reduction
            .left_instruction_input
            .as_ref()
            .map_or(product_left, |claim| claim.value);
        let reduced_right = self
            .instruction_claim_reduction
            .right_instruction_input
            .as_ref()
            .map_or(product_right, |claim| claim.value);
        if reduced_left != product_left {
            return Err(VerifierError::StageClaimOpeningMismatch {
                stage: JoltRelationId::InstructionInputVirtualization,
                left: left_reduced,
                right: left_product,
            });
        }
        if reduced_right != product_right {
            return Err(VerifierError::StageClaimOpeningMismatch {
                stage: JoltRelationId::InstructionInputVirtualization,
                left: right_reduced,
                right: right_product,
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2PublicOutput<F: Field> {
    pub product_uniskip_challenge: F,
    pub product_tau_low: Vec<F>,
    pub product_tau_high: F,
    pub ram_read_write_gamma: F,
    pub instruction_gamma: F,
    pub output_address_challenges: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2ClearOutput<F: Field> {
    pub public: Stage2PublicOutput<F>,
    /// The produced batch openings paired with their points (point + value) via the
    /// `OpeningClaim` cell. The opening points are derived from each relation's
    /// sumcheck point; later stages read them through the
    /// `*_point()` accessors and read values through `.value`, instead of joining a
    /// separately-tracked `VerifiedStage2Batch` with the wire values.
    pub output_claims: Stage2BatchOutputClaims<OpeningClaim<F>>,
    pub product_uniskip: VerifiedProductUniSkip<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2ZkOutput<F: Field, C> {
    pub public: Stage2PublicOutput<F>,
    pub product_uniskip_consistency: CommittedSumcheckConsistency<F, C>,
    pub product_uniskip_output_claims: CommittedOutputClaimOutput<C>,
    pub batch_consistency: BatchedCommittedSumcheckConsistency<F, C>,
    pub batch_output_claims: CommittedOutputClaimOutput<C>,
    /// The produced batch opening points (point-only cell), the ZK counterpart of
    /// the clear path's `output_claims`. Later stages read them through the same
    /// `*_point()` accessors via [`GetPoint`](crate::stages::relations::GetPoint).
    pub output_points: Stage2BatchOutputClaims<Vec<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage2Output<F: Field, C> {
    Clear(Stage2ClearOutput<F>),
    Zk(Stage2ZkOutput<F, C>),
}

impl<F: Field, C> Stage2Output<F, C> {
    pub fn clear(&self) -> Result<&Stage2ClearOutput<F>, crate::VerifierError> {
        match self {
            Self::Clear(output) => Ok(output),
            Self::Zk(_) => Err(crate::VerifierError::ExpectedClearProof { field: "stage2" }),
        }
    }

    pub fn zk(&self) -> Result<&Stage2ZkOutput<F, C>, crate::VerifierError> {
        match self {
            Self::Zk(output) => Ok(output),
            Self::Clear(_) => Err(crate::VerifierError::ExpectedCommittedProof { field: "stage2" }),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedProductUniSkip<F: Field> {
    pub tau_low: Vec<F>,
    pub tau_high: F,
    pub sumcheck_point: Point<HIGH_TO_LOW, F>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    /// Locks the stage-2 batch Fiat-Shamir append order against silent drift. The
    /// full relations single-source their order via the `OutputClaims` derive; only
    /// the two reduced instruction lookup operands are curated (the reduction's
    /// other openings alias the product-remainder ones and must NOT be
    /// re-absorbed). Those aliased `Option` openings carry distinct sentinels here
    /// to prove they are skipped.
    #[test]
    fn opening_values_follow_canonical_order() {
        let claims = Stage2BatchOutputClaims {
            ram_read_write: RamReadWriteOutputClaims {
                val: fr(1),
                ra: fr(2),
                inc: fr(3),
            },
            product_remainder: ProductRemainderOutputClaims {
                left_instruction_input: fr(4),
                right_instruction_input: fr(5),
                jump_flag: fr(6),
                write_lookup_output_to_rd: fr(7),
                lookup_output: fr(8),
                branch_flag: fr(9),
                next_is_noop: fr(10),
                virtual_instruction: fr(11),
            },
            instruction_claim_reduction: InstructionClaimReductionOutputClaims {
                lookup_output: Some(fr(101)),
                left_lookup_operand: fr(12),
                right_lookup_operand: fr(13),
                left_instruction_input: Some(fr(102)),
                right_instruction_input: Some(fr(103)),
            },
            ram_raf_evaluation: RamRafEvaluationOutputClaims { ram_ra: fr(14) },
            ram_output_check: RamOutputCheckOutputClaims { val_final: fr(15) },
        };

        assert_eq!(
            claims.opening_values(),
            (1..=15).map(fr).collect::<Vec<_>>()
        );
    }
}
