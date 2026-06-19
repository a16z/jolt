//! Typed inputs consumed by stage 2.

use jolt_field::Field;
use jolt_transcript::Transcript;
use serde::{Deserialize, Serialize};

use jolt_claims::protocols::jolt::JoltRelationId;

use crate::stages::relations::OpeningClaim;
use crate::stages::stage1::{Stage1ClearOutput, Stage1Output, Stage1ZkOutput};
use crate::VerifierError;

pub use super::instruction_claim_reduction::InstructionClaimReductionOutputClaims;
pub use super::product_remainder::ProductRemainderOutputClaims;
pub use super::ram_output_check::RamOutputCheckOutputClaims;
pub use super::ram_raf_evaluation::RamRafEvaluationOutputClaims;
pub use super::ram_read_write_checking::RamReadWriteOutputClaims;

#[derive(Clone, Copy)]
pub enum Deps<'a, F: Field, C> {
    Clear { stage1: &'a Stage1ClearOutput<F> },
    Zk { stage1: &'a Stage1ZkOutput<F, C> },
}

pub fn deps<F: Field, C>(stage1: &Stage1Output<F, C>) -> Deps<'_, F, C> {
    match stage1 {
        Stage1Output::Clear(stage1) => Deps::Clear { stage1 },
        Stage1Output::Zk(stage1) => Deps::Zk { stage1 },
    }
}

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
        vec![
            self.ram_read_write.val,
            self.ram_read_write.ra,
            self.ram_read_write.inc,
            self.product_remainder.left_instruction_input,
            self.product_remainder.right_instruction_input,
            self.product_remainder.jump_flag,
            self.product_remainder.write_lookup_output_to_rd,
            self.product_remainder.lookup_output,
            self.product_remainder.branch_flag,
            self.product_remainder.next_is_noop,
            self.product_remainder.virtual_instruction,
            self.instruction_claim_reduction.left_lookup_operand,
            self.instruction_claim_reduction.right_lookup_operand,
            self.ram_raf_evaluation.ram_ra,
            self.ram_output_check.val_final,
        ]
    }

    /// Append every batch opening to the transcript in canonical order, each under
    /// the `b"opening_claim"` label, matching the prover's commitment order.
    pub fn append_to_transcript<T: Transcript<Challenge = F>>(&self, transcript: &mut T) {
        for value in self.opening_values() {
            transcript.append_labeled(b"opening_claim", &value);
        }
    }
}

impl<F: Field> Stage2BatchOutputClaims<OpeningClaim<F>> {
    /// The RAM read-write opening point (shared by `val`/`ra`/`inc`).
    pub fn ram_read_write_point(&self) -> &[F] {
        &self.ram_read_write.val.point
    }

    /// The product-remainder opening point (shared by all eight openings).
    pub fn product_remainder_point(&self) -> &[F] {
        &self.product_remainder.left_instruction_input.point
    }

    /// The reduced instruction-claim opening point (shared by all five openings).
    pub fn instruction_claim_reduction_point(&self) -> &[F] {
        &self.instruction_claim_reduction.left_lookup_operand.point
    }

    /// The RAM RAF opening point (`[r_address ‖ tau_low]`).
    pub fn ram_raf_evaluation_point(&self) -> &[F] {
        &self.ram_raf_evaluation.ram_ra.point
    }

    /// The RAM output-check opening point (`r_address`).
    pub fn ram_output_check_point(&self) -> &[F] {
        &self.ram_output_check.val_final.point
    }
}
