//! The stage 2 regular-batch relation bundle.
//!
//! [`Stage2BatchRelations`] holds the five batch sumcheck relation objects and
//! their consumed-claim wiring, and is driven identically by the prover (while
//! producing the batch proof) and the verifier (after checking it). Both sides
//! build it once, take its input claims (to seed the batched sumcheck), and feed
//! it the produced opening points to evaluate every expected output — so the
//! input/output claim algebra is single-sourced through the relation objects'
//! `jolt-claims` formulas (and stays in lockstep with the BlindFold constraints).
//!
//! The product uni-skip first round is a univariate skip rather than a
//! [`SumcheckInstance`], so it stays hand-coded on each side; this bundle covers
//! only the five regular-batch relations.

use jolt_claims::protocols::jolt::{
    formulas::{
        dimensions::TraceDimensions, ram::RamRafEvaluationDimensions,
        spartan::SpartanProductDimensions,
    },
    JoltReadWriteConfig, JoltRelationId,
};
use jolt_field::Field;
use jolt_program::preprocess::PublicIoMemory;

use crate::stages::relations::{OpeningClaim, SumcheckInstance};
use crate::stages::stage1::Stage1ClearOutput;
use crate::verifier::CheckedInputs;
use crate::VerifierError;

use super::inputs::Stage2BatchOutputClaims;
use super::{
    InstructionClaimReduction, InstructionClaimReductionInputClaims,
    InstructionClaimReductionOutputClaims, ProductRemainder, ProductRemainderInputClaims,
    ProductRemainderOutputClaims, RamOutputCheck, RamOutputCheckInputClaims,
    RamOutputCheckOutputClaims, RamRafEvaluation, RamRafEvaluationInputClaims,
    RamRafEvaluationOutputClaims, RamReadWriteChecking, RamReadWriteInputClaims,
    RamReadWriteOutputClaims, Stage2BatchExpectedOutputClaims, Stage2BatchInputClaims,
};

/// Parameters needed to build the five stage 2 batch relations: the trace/RAM
/// dimensions, the public IO (for the RAM output check), the upstream stage 1
/// openings and product uni-skip reduced claim (consumed as inputs), the product
/// uni-skip challenge / `tau` points, and the batch's Fiat-Shamir gammas and
/// output-address challenges.
pub struct Stage2BatchRelationsRequest<'a, F: Field> {
    pub log_t: usize,
    pub log_k: usize,
    pub rw_config: JoltReadWriteConfig,
    pub checked: &'a CheckedInputs,
    pub stage1: &'a Stage1ClearOutput<F>,
    pub product_uniskip_output_claim: F,
    pub product_tau_low: Vec<F>,
    pub product_tau_high: F,
    pub product_uniskip_challenge: F,
    pub ram_read_write_gamma: F,
    pub instruction_gamma: F,
    pub output_address_challenges: Vec<F>,
}

/// The produced opening points (one per relation) the batch reduced to, borrowed
/// for [`Stage2BatchRelations::expected_outputs`]. The verifier derives these from
/// the batch reduction; the prover derives them from the flat sumcheck challenges.
/// They must agree (the e2e suite enforces it).
pub struct Stage2BatchOpeningPointRefs<'a, F> {
    pub ram_read_write: &'a [F],
    pub product_remainder: &'a [F],
    pub instruction_reduction: &'a [F],
    pub ram_raf: &'a [F],
    pub ram_output: &'a [F],
}

/// The five stage 2 regular-batch relations plus their consumed-claim wiring.
pub struct Stage2BatchRelations<F: Field> {
    pub ram_read_write: RamReadWriteChecking<F>,
    pub ram_read_write_inputs: RamReadWriteInputClaims<OpeningClaim<F>>,
    pub product_remainder: ProductRemainder<F>,
    pub product_remainder_inputs: ProductRemainderInputClaims<OpeningClaim<F>>,
    pub instruction_reduction: InstructionClaimReduction<F>,
    pub instruction_reduction_inputs: InstructionClaimReductionInputClaims<OpeningClaim<F>>,
    pub ram_raf: RamRafEvaluation<F>,
    pub ram_raf_inputs: RamRafEvaluationInputClaims<OpeningClaim<F>>,
    pub ram_output: RamOutputCheck<F>,
    pub ram_output_inputs: RamOutputCheckInputClaims<OpeningClaim<F>>,
}

impl<F: Field> Stage2BatchRelations<F> {
    pub fn new(request: Stage2BatchRelationsRequest<'_, F>) -> Result<Self, VerifierError> {
        let read_write_dimensions = request
            .rw_config
            .ram_dimensions(request.log_t, request.log_k);
        let raf_dimensions = RamRafEvaluationDimensions::try_from(read_write_dimensions).map_err(
            |error| VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RamRafEvaluation,
                reason: error.to_string(),
            },
        )?;
        let trace_dimensions = TraceDimensions::new(request.log_t);
        let product_dimensions = SpartanProductDimensions::new(request.log_t);
        let lowest_address = request.checked.public_io.memory_layout.get_lowest_address();
        let public_memory = PublicIoMemory::new(&request.checked.public_io).map_err(|error| {
            VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RamOutputCheck,
                reason: error.to_string(),
            }
        })?;

        Ok(Self {
            ram_read_write: RamReadWriteChecking::new(
                read_write_dimensions,
                request.log_k,
                request.ram_read_write_gamma,
                request.product_tau_low.clone(),
            ),
            ram_read_write_inputs: RamReadWriteInputClaims::from_upstream(request.stage1),
            product_remainder: ProductRemainder::new(
                product_dimensions,
                request.product_uniskip_challenge,
                request.product_tau_high,
                request.product_tau_low.clone(),
            ),
            product_remainder_inputs: ProductRemainderInputClaims::from_uniskip_output(
                request.product_uniskip_output_claim,
            ),
            instruction_reduction: InstructionClaimReduction::new(
                trace_dimensions,
                request.instruction_gamma,
                request.product_tau_low.clone(),
            ),
            instruction_reduction_inputs: InstructionClaimReductionInputClaims::from_upstream(
                request.stage1,
            ),
            ram_raf: RamRafEvaluation::new(
                read_write_dimensions,
                raf_dimensions,
                request.log_k,
                lowest_address,
                request.product_tau_low,
            ),
            ram_raf_inputs: RamRafEvaluationInputClaims::from_upstream(request.stage1),
            ram_output: RamOutputCheck::new(
                read_write_dimensions,
                request.output_address_challenges,
                public_memory,
            ),
            ram_output_inputs: RamOutputCheckInputClaims::from_upstream(),
        })
    }

    /// Evaluate every relation's input claim (the claimed sum that seeds its slot
    /// in the batched sumcheck).
    pub fn input_claims(&self) -> Result<Stage2BatchInputClaims<F>, VerifierError> {
        Ok(Stage2BatchInputClaims {
            ram_read_write: self.ram_read_write.input_claim(&self.ram_read_write_inputs)?,
            product_remainder: self
                .product_remainder
                .input_claim(&self.product_remainder_inputs)?,
            instruction_claim_reduction: self
                .instruction_reduction
                .input_claim(&self.instruction_reduction_inputs)?,
            ram_raf_evaluation: self.ram_raf.input_claim(&self.ram_raf_inputs)?,
            ram_output_check: self.ram_output.input_claim(&self.ram_output_inputs)?,
        })
    }

    /// Reconstruct every relation's expected output claim from the produced opening
    /// `points` and committed opening `values`. The three aliased
    /// instruction-claim-reduction openings, absent on the wire, reuse the
    /// product-remainder openings (or zero when the points disagree — a defensive
    /// fallback that mirrors the legacy reconstruction).
    pub fn expected_outputs(
        &self,
        points: Stage2BatchOpeningPointRefs<'_, F>,
        values: &Stage2BatchOutputClaims<F>,
    ) -> Result<Stage2BatchExpectedOutputClaims<F>, VerifierError> {
        let opening = |point: &[F], value: F| OpeningClaim {
            point: point.to_vec(),
            value,
        };

        let ram_read_write_openings = RamReadWriteOutputClaims {
            val: opening(points.ram_read_write, values.ram_read_write.val),
            ra: opening(points.ram_read_write, values.ram_read_write.ra),
            inc: opening(points.ram_read_write, values.ram_read_write.inc),
        };
        let product = &values.product_remainder;
        let product_remainder_openings = ProductRemainderOutputClaims {
            left_instruction_input: opening(points.product_remainder, product.left_instruction_input),
            right_instruction_input: opening(
                points.product_remainder,
                product.right_instruction_input,
            ),
            jump_flag: opening(points.product_remainder, product.jump_flag),
            write_lookup_output_to_rd: opening(
                points.product_remainder,
                product.write_lookup_output_to_rd,
            ),
            lookup_output: opening(points.product_remainder, product.lookup_output),
            branch_flag: opening(points.product_remainder, product.branch_flag),
            next_is_noop: opening(points.product_remainder, product.next_is_noop),
            virtual_instruction: opening(points.product_remainder, product.virtual_instruction),
        };
        let reduction = &values.instruction_claim_reduction;
        let points_match = points.product_remainder == points.instruction_reduction;
        let aliased = |value: Option<F>, product_value: F| {
            let resolved = value.unwrap_or(if points_match {
                product_value
            } else {
                F::from_u64(0)
            });
            Some(opening(points.instruction_reduction, resolved))
        };
        let instruction_reduction_openings = InstructionClaimReductionOutputClaims {
            lookup_output: aliased(reduction.lookup_output, product.lookup_output),
            left_lookup_operand: opening(points.instruction_reduction, reduction.left_lookup_operand),
            right_lookup_operand: opening(
                points.instruction_reduction,
                reduction.right_lookup_operand,
            ),
            left_instruction_input: aliased(
                reduction.left_instruction_input,
                product.left_instruction_input,
            ),
            right_instruction_input: aliased(
                reduction.right_instruction_input,
                product.right_instruction_input,
            ),
        };
        let ram_raf_openings = RamRafEvaluationOutputClaims {
            ram_ra: opening(points.ram_raf, values.ram_raf_evaluation),
        };
        let ram_output_openings = RamOutputCheckOutputClaims {
            val_final: opening(points.ram_output, values.ram_output_check),
        };

        Ok(Stage2BatchExpectedOutputClaims {
            ram_read_write: self
                .ram_read_write
                .expected_output(&self.ram_read_write_inputs, &ram_read_write_openings)?,
            product_remainder: self
                .product_remainder
                .expected_output(&self.product_remainder_inputs, &product_remainder_openings)?,
            instruction_claim_reduction: self.instruction_reduction.expected_output(
                &self.instruction_reduction_inputs,
                &instruction_reduction_openings,
            )?,
            ram_raf_evaluation: self
                .ram_raf
                .expected_output(&self.ram_raf_inputs, &ram_raf_openings)?,
            ram_output_check: self
                .ram_output
                .expected_output(&self.ram_output_inputs, &ram_output_openings)?,
        })
    }
}
