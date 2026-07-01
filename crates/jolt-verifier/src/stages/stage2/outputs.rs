//! Typed inputs consumed and outputs produced by stage 2 verification.

use jolt_claims::protocols::jolt::{geometry::instruction, JoltRelationId};
use jolt_field::Field;
use jolt_poly::{Point, HIGH_TO_LOW};
use jolt_sumcheck::{BatchedCommittedSumcheckConsistency, CommittedSumcheckConsistency};
use jolt_transcript::Transcript;
use serde::{Deserialize, Serialize};

use crate::stages::relations::{OutputClaims, SumcheckBatch};
use crate::stages::zk::outputs::CommittedOutputClaimOutput;
use crate::VerifierError;

pub use super::instruction_claim_reduction::{
    InstructionClaimReduction, InstructionClaimReductionOutputClaims,
};
pub use super::product_remainder::{ProductRemainder, ProductRemainderOutputClaims};
pub use super::ram_output_check::{RamOutputCheck, RamOutputCheckOutputClaims};
pub use super::ram_raf_evaluation::{RamRafEvaluation, RamRafEvaluationOutputClaims};
pub use super::ram_read_write_checking::{RamReadWriteChecking, RamReadWriteOutputClaims};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: for<'a> Deserialize<'a>"))]
pub struct Stage2OutputClaims<F: Field> {
    pub product_uniskip_output_claim: F,
    pub batch_outputs: Stage2BatchOutputClaims<F>,
}

/// Source-of-truth for stage 2's five-instance sumcheck batch, in Fiat-Shamir
/// batch order (RAM read-write, product remainder, instruction claim-reduction,
/// RAM RAF evaluation, RAM output check). `#[derive(SumcheckBatch)]` generates the
/// `Stage2Batch{Input,Output}{Claims,Points}<F>` and `Stage2BatchChallenges<F>`
/// aggregates — one field per instance, in this
/// declaration order. The product uni-skip is a separate sub-sumcheck, not part of
/// this batch.
///
/// The opt-out `#[sumcheck_batch(custom_opening_values)]` suppresses the generated
/// `opening_values` / `append_to_transcript`: the instruction claim-reduction has
/// three cross-relation aliased openings (`lookup_output` /
/// `left`/`right_instruction_input`) that alias the product-remainder openings and
/// must NOT be re-absorbed, so the canonical order is curated by hand below (and
/// enforced by `validate`).
#[derive(SumcheckBatch)]
#[sumcheck_batch(custom_opening_values)]
pub struct Stage2BatchSumchecks<F: Field> {
    pub ram_read_write: RamReadWriteChecking<F>,
    pub product_remainder: ProductRemainder<F>,
    pub instruction_claim_reduction: InstructionClaimReduction<F>,
    pub ram_raf_evaluation: RamRafEvaluation<F>,
    pub ram_output_check: RamOutputCheck<F>,
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

    /// Append every batch opening to the transcript in canonical order, each under
    /// the `b"opening_claim"` label, matching the prover's commitment order.
    pub fn append_to_transcript<T: Transcript<Challenge = F>>(&self, transcript: &mut T) {
        for value in self.opening_values() {
            transcript.append_labeled(b"opening_claim", &value);
        }
    }
}

/// The shared per-relation opening-point accessors over the point-only stage-2
/// batch aggregate.
impl<F: Field> Stage2BatchOutputPoints<F> {
    /// The RAM read-write opening point (shared by `val`/`ra`/`inc`).
    pub fn ram_read_write_point(&self) -> &[F] {
        self.ram_read_write.val()
    }

    /// The product-remainder opening point (shared by all eight openings).
    pub fn product_remainder_point(&self) -> &[F] {
        self.product_remainder.left_instruction_input()
    }

    /// The reduced instruction-claim opening point (shared by all five openings).
    pub fn instruction_claim_reduction_point(&self) -> &[F] {
        self.instruction_claim_reduction.left_lookup_operand()
    }

    /// The RAM RAF opening point (`[r_address ‖ tau_low]`).
    pub fn ram_raf_evaluation_point(&self) -> &[F] {
        self.ram_raf_evaluation.ram_ra()
    }

    /// The RAM output-check opening point (`r_address`).
    pub fn ram_output_check_point(&self) -> &[F] {
        self.ram_output_check.val_final()
    }
}

impl<F: Field> Stage2BatchOutputClaims<F> {
    /// Enforce the cross-relation aliases between the product-remainder openings and
    /// the reduced instruction-claim openings: when a reduced opening is present it
    /// must share the product-remainder opening's point and value. Downstream
    /// consumers rely on these aliases when they fall back to the product remainder —
    /// the stage-5 instruction read-RAF wiring (`lookup_output`) and the stage-3
    /// instruction-input virtualization (`left`/`right_instruction_input`) — so the
    /// stage-2 verifier checks them here rather than each consumer re-checking. Errors
    /// preserve the opening and relation ids those consumers reported. The value
    /// fallbacks read off `self` (the values); the point-agreement test reads off the
    /// paired `points` struct.
    pub fn validate(&self, points: &Stage2BatchOutputPoints<F>) -> Result<(), VerifierError> {
        let [(lookup_output_reduced, lookup_output_product)] =
            instruction::read_raf_consistency_openings();
        let [(left_reduced, left_product), (right_reduced, right_product)] =
            instruction::input_virtualization_consistency_openings();

        // Every reduced instruction opening shares the product-remainder opening
        // point; if the points disagree the reduced openings cannot alias the
        // product ones.
        if points.product_remainder_point() != points.instruction_claim_reduction_point() {
            return Err(VerifierError::StageClaimOpeningMismatch {
                stage: JoltRelationId::InstructionReadRaf,
                left: lookup_output_reduced,
                right: lookup_output_product,
            });
        }

        // `lookup_output`: stage-5 instruction read-RAF fallback to the product remainder.
        let product_lookup_output = self.product_remainder.lookup_output;
        let reduced_lookup_output = self
            .instruction_claim_reduction
            .lookup_output
            .unwrap_or(product_lookup_output);
        if reduced_lookup_output != product_lookup_output {
            return Err(VerifierError::StageClaimOpeningMismatch {
                stage: JoltRelationId::InstructionReadRaf,
                left: lookup_output_reduced,
                right: lookup_output_product,
            });
        }

        // `left`/`right_instruction_input`: stage-3 instruction-input virtualization
        // fallback to the product remainder.
        let product_left = self.product_remainder.left_instruction_input;
        let product_right = self.product_remainder.right_instruction_input;
        let reduced_left = self
            .instruction_claim_reduction
            .left_instruction_input
            .unwrap_or(product_left);
        let reduced_right = self
            .instruction_claim_reduction
            .right_instruction_input
            .unwrap_or(product_right);
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
pub struct Stage2ClearOutput<F: Field> {
    /// The produced batch opening *values* (wire form); later stages read each
    /// opening's value directly off these fields.
    pub output_values: Stage2BatchOutputClaims<F>,
    /// The produced batch opening *points*, paired field-for-field with
    /// `output_values`. Later stages read the points through the `*_point()`
    /// accessors.
    pub output_points: Stage2BatchOutputPoints<F>,
    pub product_uniskip: VerifiedProductUniSkip<F>,
}

/// Stage 2's ZK output, carrying the Fiat-Shamir values BlindFold sources via
/// `input.stage2.<field>`. The two batch gammas are the generated
/// [`Stage2BatchChallenges`] member structs (`challenges.ram_read_write.gamma`,
/// `challenges.instruction_claim_reduction.gamma`; the other three batch relations
/// draw nothing — `NoChallenges`). The remaining three are non-batch draws — the
/// product uni-skip reduction challenge and its freshly-drawn `product_tau_high`
/// scalar (a separate sub-sumcheck), and the RAM output-check address reference
/// point (folded in like stage 6's booleanity reference points) — so they are not
/// part of the per-instance aggregate. `product_tau_low` is opening-derived (stage
/// 1's remainder sumcheck point low half), stored so downstream stage-3 relation
/// construction can read it mode-agnostically via
/// [`Stage2Output::product_tau_low`]; BlindFold independently recomputes it from
/// `stage1.remainder_consistency`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2ZkOutput<F: Field, C> {
    pub challenges: Stage2BatchChallenges<F>,
    pub product_uniskip_challenge: F,
    pub product_tau_low: Vec<F>,
    pub product_tau_high: F,
    pub output_address_challenges: Vec<F>,
    pub product_uniskip_consistency: CommittedSumcheckConsistency<F, C>,
    pub product_uniskip_output_claims: CommittedOutputClaimOutput<C>,
    pub batch_consistency: BatchedCommittedSumcheckConsistency<F, C>,
    pub batch_output_claims: CommittedOutputClaimOutput<C>,
    /// The produced batch opening points, the ZK counterpart of the clear path's
    /// `output_points`. Later stages read them through the same `*_point()` accessors.
    pub output_points: Stage2BatchOutputPoints<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage2Output<F: Field, C> {
    Clear(Stage2ClearOutput<F>),
    Zk(Stage2ZkOutput<F, C>),
}

impl<F: Field, C> Stage2Output<F, C> {
    /// The product uni-skip `tau_low` (stage 1's remainder point low half,
    /// reversed), available regardless of proving mode. Stage 3's relation
    /// construction evaluates its `EqPlusOne`/`EqSpartan` publics against it.
    pub fn product_tau_low(&self) -> &[F] {
        match self {
            Self::Clear(output) => &output.product_uniskip.tau_low,
            Self::Zk(output) => &output.product_tau_low,
        }
    }

    /// The produced batch opening points, available regardless of proving mode.
    pub fn batch_output_points(&self) -> &Stage2BatchOutputPoints<F> {
        match self {
            Self::Clear(output) => &output.output_points,
            Self::Zk(output) => &output.output_points,
        }
    }

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
        let claims = Stage2BatchOutputClaims::<Fr> {
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

    /// A stage-2 batch output where the reduced instruction openings alias the
    /// product-remainder ones (equal values): `lookup_output`,
    /// `left`/`right_instruction_input`. The paired points agree on the shared
    /// point. `validate` accepts it; the tests below perturb one alias value (or the
    /// shared point) each to assert rejection.
    fn consistent_values() -> Stage2BatchOutputClaims<Fr> {
        Stage2BatchOutputClaims::<Fr> {
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
                lookup_output: Some(fr(8)),
                left_lookup_operand: fr(12),
                right_lookup_operand: fr(13),
                left_instruction_input: Some(fr(4)),
                right_instruction_input: Some(fr(5)),
            },
            ram_raf_evaluation: RamRafEvaluationOutputClaims { ram_ra: fr(14) },
            ram_output_check: RamOutputCheckOutputClaims { val_final: fr(15) },
        }
    }

    /// The paired points for [`consistent_values`]: the product-remainder and
    /// reduced instruction-claim openings share the SAME point (so the alias holds).
    fn consistent_points(shared_point: Vec<Fr>) -> Stage2BatchOutputPoints<Fr> {
        let p = || shared_point.clone();
        Stage2BatchOutputPoints::<Fr> {
            ram_read_write: RamReadWriteOutputClaims {
                val: p(),
                ra: p(),
                inc: p(),
            },
            product_remainder: ProductRemainderOutputClaims {
                left_instruction_input: p(),
                right_instruction_input: p(),
                jump_flag: p(),
                write_lookup_output_to_rd: p(),
                lookup_output: p(),
                branch_flag: p(),
                next_is_noop: p(),
                virtual_instruction: p(),
            },
            instruction_claim_reduction: InstructionClaimReductionOutputClaims {
                lookup_output: Some(p()),
                left_lookup_operand: p(),
                right_lookup_operand: p(),
                left_instruction_input: Some(p()),
                right_instruction_input: Some(p()),
            },
            ram_raf_evaluation: RamRafEvaluationOutputClaims { ram_ra: p() },
            ram_output_check: RamOutputCheckOutputClaims { val_final: p() },
        }
    }

    #[test]
    fn validate_accepts_consistent_reduction() {
        assert!(consistent_values()
            .validate(&consistent_points(vec![fr(7)]))
            .is_ok());
    }

    #[test]
    fn validate_rejects_lookup_output_mismatch() {
        let mut values = consistent_values();
        values.instruction_claim_reduction.lookup_output = Some(fr(99));
        assert!(values.validate(&consistent_points(vec![fr(7)])).is_err());
    }

    #[test]
    fn validate_rejects_left_instruction_input_mismatch() {
        let mut values = consistent_values();
        values.instruction_claim_reduction.left_instruction_input = Some(fr(99));
        assert!(values.validate(&consistent_points(vec![fr(7)])).is_err());
    }

    #[test]
    fn validate_rejects_right_instruction_input_mismatch() {
        let mut values = consistent_values();
        values.instruction_claim_reduction.right_instruction_input = Some(fr(99));
        assert!(values.validate(&consistent_points(vec![fr(7)])).is_err());
    }

    #[test]
    fn validate_rejects_reduction_point_mismatch() {
        let values = consistent_values();
        // Shift the product-remainder opening point away from the reduction's, so
        // the reduced openings cannot alias the product ones.
        let mut points = consistent_points(vec![fr(7)]);
        points.product_remainder.left_instruction_input = vec![fr(1)];
        assert!(values.validate(&points).is_err());
    }
}
