use jolt_field::JoltField;
use jolt_sumcheck::{BatchedCommittedSumcheckConsistency, CommittedSumcheckConsistency};
use serde::{Deserialize, Serialize};

use crate::stages::relations::SumcheckBatch;
use crate::stages::zk::outputs::CommittedOutputClaimOutput;

#[cfg(feature = "field-inline")]
pub use super::field_registers_claim_reduction::{
    FieldRegistersClaimReduction, FieldRegistersClaimReductionOutputClaims,
};
pub use super::instruction_claim_reduction::{
    InstructionClaimReduction, InstructionClaimReductionOutputClaims,
};
pub use super::product_remainder::{ProductRemainder, ProductRemainderOutputClaims};
pub use super::ram_output_check::{RamOutputCheck, RamOutputCheckOutputClaims};
pub use super::ram_raf_evaluation::{RamRafEvaluation, RamRafEvaluationOutputClaims};
pub use super::ram_read_write_checking::{RamReadWriteChecking, RamReadWriteOutputClaims};

#[cfg(feature = "field-inline")]
pub use jolt_claims::protocols::field_inline::relations::product::FieldRegistersProductOutputClaims;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: for<'a> Deserialize<'a>"))]
pub struct Stage2OutputClaims<F: JoltField> {
    pub product_uniskip_output_claim: F,
    #[cfg_attr(feature = "field-inline", serde(with = "canonical_batch"))]
    pub batch_outputs: Stage2BatchOutputClaims<F>,
}

impl<F: JoltField> Stage2OutputClaims<F> {
    pub fn new(product_uniskip_output_claim: F, batch_outputs: Stage2BatchOutputClaims<F>) -> Self {
        Self {
            product_uniskip_output_claim,
            batch_outputs,
        }
    }
}

impl<F: JoltField> Stage2BatchOutputClaims<F> {
    /// Construct the ordinary stage-2 batch claims. Producers without field-inline semantics
    /// use this regardless of the build's feature set — the field-inline claim-reduction slot
    /// defaults to all-zero claims, inert because such producers' proofs never declare the
    /// field-inline axis.
    #[cfg_attr(
        not(feature = "field-inline"),
        expect(
            clippy::useless_conversion,
            reason = "field-inline selects a composed claim or opening id"
        )
    )]
    pub fn new(
        ram_read_write: RamReadWriteOutputClaims<F>,
        product_remainder: ProductRemainderOutputClaims<F>,
        instruction_claim_reduction: InstructionClaimReductionOutputClaims<F>,
        ram_raf_evaluation: RamRafEvaluationOutputClaims<F>,
        ram_output_check: RamOutputCheckOutputClaims<F>,
    ) -> Self {
        Self {
            ram_read_write,
            product_remainder: product_remainder.into(),
            instruction_claim_reduction,
            #[cfg(feature = "field-inline")]
            field_registers_claim_reduction: Default::default(),
            ram_raf_evaluation,
            ram_output_check,
        }
    }
}

#[derive(SumcheckBatch)]
#[sumcheck_batch(crate = "crate")]
pub struct Stage2BatchSumchecks<F: JoltField> {
    pub ram_read_write: RamReadWriteChecking<F>,
    pub product_remainder: ProductRemainder<F>,
    pub instruction_claim_reduction: InstructionClaimReduction<F>,
    #[cfg(feature = "field-inline")]
    pub field_registers_claim_reduction: FieldRegistersClaimReduction<F>,
    pub ram_raf_evaluation: RamRafEvaluation<F>,
    pub ram_output_check: RamOutputCheck<F>,
}

impl<F: JoltField> Stage2BatchOutputPoints<F> {
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

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "allocative", derive(::allocative::Allocative))]
pub struct Stage2ClearOutput<F: JoltField> {
    pub output_values: Stage2BatchOutputClaims<F>,
    pub output_points: Stage2BatchOutputPoints<F>,
    pub product_tau_low: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2ZkOutput<F: JoltField, C> {
    pub challenges: Stage2BatchChallenges<F>,
    pub product_uniskip_challenge: F,
    pub product_tau_low: Vec<F>,
    pub product_tau_high: F,
    pub product_uniskip_consistency: CommittedSumcheckConsistency<F, C>,
    pub product_uniskip_output_claims: CommittedOutputClaimOutput<C>,
    pub batch_consistency: BatchedCommittedSumcheckConsistency<F, C>,
    pub batch_output_claims: CommittedOutputClaimOutput<C>,
    pub output_points: Stage2BatchOutputPoints<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage2Output<F: JoltField, C> {
    Clear(Stage2ClearOutput<F>),
    Zk(Stage2ZkOutput<F, C>),
}

impl<F: JoltField, C> Stage2Output<F, C> {
    /// The product uni-skip `tau_low` (stage 1's remainder point low half,
    /// reversed), available regardless of proving mode. Stage 3's relation
    /// construction evaluates its `EqPlusOne`/`EqSpartan` publics against it.
    pub fn product_tau_low(&self) -> &[F] {
        match self {
            Self::Clear(output) => &output.product_tau_low,
            Self::Zk(output) => &output.product_tau_low,
        }
    }

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

#[cfg(test)]
#[expect(clippy::unwrap_used)]
#[expect(
    clippy::as_conversions,
    reason = "tests use plain arithmetic on fixture data"
)]
mod tests {
    use super::*;
    use crate::stages::relations::draw_recording::{record, DrawEvent};
    use crate::stages::relations::ConcreteSumcheck;
    use common::jolt_device::{JoltDevice, MemoryConfig};
    #[cfg(feature = "field-inline")]
    use jolt_claims::protocols::field_inline::FieldRegistersTraceDimensions;
    use jolt_claims::protocols::jolt::geometry::{
        dimensions::{ReadWriteDimensions, TraceDimensions},
        ram::RamRafEvaluationDimensions,
        spartan::SpartanProductDimensions,
    };
    use jolt_field::{Fr, Ring};
    use jolt_program::preprocess::PublicIoMemory;
    use jolt_transcript::Transcript;

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    fn sumchecks() -> Stage2BatchSumchecks<Fr> {
        let log_t = 4usize;
        let log_k = 3usize;
        let dimensions = ReadWriteDimensions::new(log_t, log_k, 2, 1);
        let raf_dimensions = RamRafEvaluationDimensions::try_from(dimensions).unwrap();
        let public_memory = PublicIoMemory::new(&JoltDevice::new(&MemoryConfig {
            program_size: Some(1024),
            ..Default::default()
        }))
        .unwrap();
        Stage2BatchSumchecks::<Fr> {
            ram_read_write: RamReadWriteChecking::new(dimensions, log_k, Vec::new()),
            product_remainder: ProductRemainder::new(
                SpartanProductDimensions::new(log_t),
                fr(1),
                fr(2),
                Vec::new(),
            ),
            instruction_claim_reduction: InstructionClaimReduction::new(
                TraceDimensions::new(log_t),
                Vec::new(),
            ),
            #[cfg(feature = "field-inline")]
            field_registers_claim_reduction: FieldRegistersClaimReduction::new(
                FieldRegistersTraceDimensions::new(log_t),
                Vec::new(),
            ),
            ram_raf_evaluation: RamRafEvaluation::new(
                dimensions,
                raf_dimensions,
                log_k,
                0,
                Vec::new(),
            ),
            ram_output_check: RamOutputCheck::new(dimensions, public_memory),
        }
    }

    #[test]
    fn draw_challenges_matches_inline_draw_sequence() {
        let sumchecks = sumchecks();
        let log_k = sumchecks.ram_output_check.read_write_dimensions().log_k();
        #[cfg(not(feature = "field-inline"))]
        let gamma_draws = 2usize;
        #[cfg(feature = "field-inline")]
        let gamma_draws = 3usize;
        let (inline_events, (inline_gammas, inline_output_address)) = record(|t| {
            (
                (0..gamma_draws)
                    .map(|_| t.challenge_scalar())
                    .collect::<Vec<Fr>>(),
                (0..log_k).map(|_| t.challenge()).collect::<Vec<Fr>>(),
            )
        });
        let (draw_events, challenges) = record(|t| sumchecks.draw_challenges(t).unwrap());

        assert_eq!(draw_events, inline_events);
        assert_eq!(
            draw_events,
            (1..=(gamma_draws + log_k) as u64)
                .map(DrawEvent::Squeeze)
                .collect::<Vec<_>>()
        );
        #[cfg(not(feature = "field-inline"))]
        let drawn_gammas = vec![
            challenges.ram_read_write.gamma,
            challenges.instruction_claim_reduction.gamma,
        ];
        #[cfg(feature = "field-inline")]
        let drawn_gammas = vec![
            challenges.ram_read_write.gamma,
            challenges.instruction_claim_reduction.gamma,
            challenges.field_registers_claim_reduction.gamma,
        ];
        assert_eq!(drawn_gammas, inline_gammas);
        assert_eq!(
            challenges.ram_output_check.output_address,
            inline_output_address
        );
    }

    #[cfg_attr(
        not(feature = "field-inline"),
        expect(
            clippy::useless_conversion,
            reason = "field-inline selects composed claim and opening types"
        )
    )]
    fn consistent_values() -> Stage2BatchOutputClaims<Fr> {
        #[cfg_attr(not(feature = "field-inline"), expect(unused_mut))]
        let mut claims = Stage2BatchOutputClaims::<Fr> {
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
            }
            .into(),
            instruction_claim_reduction: InstructionClaimReductionOutputClaims {
                lookup_output: fr(8),
                left_lookup_operand: fr(12),
                right_lookup_operand: fr(13),
                left_instruction_input: fr(4),
                right_instruction_input: fr(5),
            },
            #[cfg(feature = "field-inline")]
            field_registers_claim_reduction: FieldRegistersClaimReductionOutputClaims {
                rd_value: fr(16),
                rs1_value: fr(17),
                rs2_value: fr(18),
            },
            ram_raf_evaluation: RamRafEvaluationOutputClaims { ram_ra: fr(14) },
            ram_output_check: RamOutputCheckOutputClaims { val_final: fr(15) },
        };
        #[cfg(feature = "field-inline")]
        {
            claims.product_remainder.field_inline = FieldRegistersProductOutputClaims {
                rs1_value: fr(17),
                rs2_value: fr(18),
                rd_value: fr(16),
            };
        }
        claims
    }

    #[cfg(not(feature = "field-inline"))]
    #[test]
    fn opening_values_follow_canonical_order() {
        let mut claims = consistent_values();
        claims.instruction_claim_reduction.lookup_output = fr(101);
        claims.instruction_claim_reduction.left_instruction_input = fr(102);
        claims.instruction_claim_reduction.right_instruction_input = fr(103);

        assert_eq!(
            sumchecks().opening_values(&claims),
            (1..=15).map(fr).collect::<Vec<_>>()
        );
    }

    #[cfg(feature = "field-inline")]
    #[test]
    fn opening_values_follow_canonical_field_inline_order() {
        let mut claims = consistent_values();
        claims.instruction_claim_reduction.lookup_output = fr(101);
        claims.instruction_claim_reduction.left_instruction_input = fr(102);
        claims.instruction_claim_reduction.right_instruction_input = fr(103);
        claims.product_remainder.field_inline = FieldRegistersProductOutputClaims {
            rs1_value: fr(201),
            rs2_value: fr(202),
            rd_value: fr(203),
        };

        let expected = (1..=11)
            .map(fr)
            .chain([fr(201), fr(202), fr(203)])
            .chain([fr(12), fr(13)])
            .chain([fr(14), fr(15)])
            .collect::<Vec<_>>();
        assert_eq!(sumchecks().opening_values(&claims), expected);
    }

    #[test]
    fn output_claim_count_matches_absorbed_openings() {
        let sumchecks = sumchecks();
        assert_eq!(
            sumchecks.opening_values(&consistent_values()).len(),
            sumchecks.output_claim_count()
        );
        assert_eq!(
            sumchecks.output_claim_count(),
            if cfg!(feature = "field-inline") {
                18
            } else {
                15
            }
        );
    }

    #[test]
    #[cfg_attr(
        not(feature = "field-inline"),
        expect(
            clippy::useless_conversion,
            reason = "field-inline selects composed claim and opening types"
        )
    )]
    fn alias_declarations_are_valid() {
        use jolt_claims::SymbolicSumcheck as _;
        use std::collections::BTreeSet;

        let sumchecks = sumchecks();
        let expression_openings = sumchecks
            .instruction_claim_reduction
            .symbolic()
            .expected_output_openings::<Fr>();
        let source_wire_openings = sumchecks.product_remainder.wire_output_openings();

        let pairs = InstructionClaimReduction::<Fr>::aliased_output_openings();
        assert_eq!(pairs.len(), 3);
        let mut seen = BTreeSet::new();
        for (aliased, source) in pairs {
            assert!(
                seen.insert(aliased),
                "duplicate aliased opening {aliased:?}"
            );
            assert!(
                expression_openings.contains(&aliased),
                "aliased opening {aliased:?} is not referenced by the reduction's output Expr",
            );
            assert!(
                source_wire_openings.contains(&source.into()),
                "source {source:?} is not absorbed by the product remainder",
            );
        }
    }

    #[cfg(feature = "field-inline")]
    #[test]
    fn wire_claims_reconstruct_reduction_aliases_from_the_product() {
        let claims = Stage2OutputClaims::new(fr(19), consistent_values());
        let bytes = postcard::to_stdvec(&claims).unwrap();
        let decoded: Stage2OutputClaims<Fr> = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(decoded, claims);
        let mut inconsistent = claims;
        inconsistent
            .batch_outputs
            .field_registers_claim_reduction
            .rs1_value += fr(1);
        assert!(sumchecks()
            .validate_aliases(&inconsistent.batch_outputs)
            .is_err());
        assert_eq!(postcard::to_stdvec(&inconsistent).unwrap(), bytes);
    }

    #[test]
    fn validate_aliases_accepts_consistent_reduction() {
        assert!(sumchecks().validate_aliases(&consistent_values()).is_ok());
    }

    #[test]
    fn validate_aliases_rejects_lookup_output_mismatch() {
        let mut values = consistent_values();
        values.instruction_claim_reduction.lookup_output = fr(99);
        assert!(sumchecks().validate_aliases(&values).is_err());
    }

    #[test]
    fn validate_aliases_rejects_left_instruction_input_mismatch() {
        let mut values = consistent_values();
        values.instruction_claim_reduction.left_instruction_input = fr(99);
        assert!(sumchecks().validate_aliases(&values).is_err());
    }

    #[test]
    fn validate_aliases_rejects_right_instruction_input_mismatch() {
        let mut values = consistent_values();
        values.instruction_claim_reduction.right_instruction_input = fr(99);
        assert!(sumchecks().validate_aliases(&values).is_err());
    }

    #[test]
    fn aliased_members_derive_identical_opening_points() {
        let sumchecks = sumchecks();
        let product = &sumchecks.product_remainder;
        let reduction = &sumchecks.instruction_claim_reduction;
        assert_eq!(product.rounds(), reduction.rounds());
        let batch_num_vars = product.rounds() + 2;
        assert_eq!(
            product.instance_point_offset(batch_num_vars).unwrap(),
            reduction.instance_point_offset(batch_num_vars).unwrap(),
        );

        let point: Vec<Fr> = (0..product.rounds() as u64).map(|i| fr(20 + i)).collect();
        let input_points = sumchecks.empty_input_points();
        let product_points = product
            .derive_opening_points(&point, &input_points.product_remainder)
            .unwrap();
        let reduction_points = reduction
            .derive_opening_points(&point, &input_points.instruction_claim_reduction)
            .unwrap();
        assert_eq!(product_points.lookup_output, reduction_points.lookup_output);
        assert_eq!(
            product_points.left_instruction_input,
            reduction_points.left_instruction_input,
        );
        assert_eq!(
            product_points.right_instruction_input,
            reduction_points.right_instruction_input,
        );
    }

    #[cfg(feature = "field-inline")]
    #[test]
    fn field_registers_claim_reduction_shares_the_product_remainder_point() {
        let sumchecks = sumchecks();
        let product = &sumchecks.product_remainder;
        let reduction = &sumchecks.field_registers_claim_reduction;
        assert_eq!(product.rounds(), reduction.rounds());
        let batch_num_vars = product.rounds() + 2;
        assert_eq!(
            product.instance_point_offset(batch_num_vars).unwrap(),
            reduction.instance_point_offset(batch_num_vars).unwrap(),
        );

        let point: Vec<Fr> = (0..product.rounds() as u64).map(|i| fr(40 + i)).collect();
        let input_points = sumchecks.empty_input_points();
        let product_points = product
            .derive_opening_points(&point, &input_points.product_remainder)
            .unwrap();
        let reduction_points = reduction
            .derive_opening_points(&point, &input_points.field_registers_claim_reduction)
            .unwrap();
        assert_eq!(
            product_points.left_instruction_input,
            reduction_points.rd_value,
        );
        assert_eq!(reduction_points.rd_value, reduction_points.rs1_value);
        assert_eq!(reduction_points.rd_value, reduction_points.rs2_value);
    }
}

#[cfg(feature = "field-inline")]
mod canonical_batch {
    use super::*;
    use jolt_claims::protocols::composed::ProductOutputs;
    use serde::ser::SerializeStruct;
    use serde::{Deserializer, Serializer};

    pub fn serialize<F: JoltField, S: Serializer>(
        claims: &Stage2BatchOutputClaims<F>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        let mut wire = serializer.serialize_struct("Stage2BatchOutputClaims", 5)?;
        wire.serialize_field("ram_read_write", &claims.ram_read_write)?;
        wire.serialize_field("product_remainder", &claims.product_remainder)?;
        wire.serialize_field(
            "instruction_claim_reduction",
            &claims.instruction_claim_reduction,
        )?;
        wire.serialize_field("ram_raf_evaluation", &claims.ram_raf_evaluation)?;
        wire.serialize_field("ram_output_check", &claims.ram_output_check)?;
        wire.end()
    }

    #[derive(Deserialize)]
    #[serde(bound = "F: JoltField")]
    struct CanonicalBatch<F: JoltField> {
        ram_read_write: RamReadWriteOutputClaims<F>,
        product_remainder: ProductOutputs<F>,
        instruction_claim_reduction: InstructionClaimReductionOutputClaims<F>,
        ram_raf_evaluation: RamRafEvaluationOutputClaims<F>,
        ram_output_check: RamOutputCheckOutputClaims<F>,
    }

    pub fn deserialize<'de, F: JoltField, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Stage2BatchOutputClaims<F>, D::Error> {
        let CanonicalBatch {
            ram_read_write,
            product_remainder,
            instruction_claim_reduction,
            ram_raf_evaluation,
            ram_output_check,
        } = CanonicalBatch::<F>::deserialize(deserializer)?;
        let field_registers_claim_reduction = FieldRegistersClaimReductionOutputClaims {
            rs1_value: product_remainder.field_inline.rs1_value,
            rs2_value: product_remainder.field_inline.rs2_value,
            rd_value: product_remainder.field_inline.rd_value,
        };
        Ok(Stage2BatchOutputClaims {
            ram_read_write,
            product_remainder,
            instruction_claim_reduction,
            field_registers_claim_reduction,
            ram_raf_evaluation,
            ram_output_check,
        })
    }
}
