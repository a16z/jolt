//! The stage 6 lattice `IncVirtualization` cycle-phase sumcheck instance.
//!
//! Consumes the same four reduced `Inc` claims as [`IncClaimReduction`]
//! (RAM read-write / val-check, register read-write / val-evaluation) but
//! reduces them to the packed `FusedInc` column opening plus the
//! `OpFlags(Store)` destination selector, instead of the per-polynomial
//! `RamInc` / `RdInc` openings. Its publics are the same per-source `Eq`
//! coefficients.
//!
//! [`IncClaimReduction`]: super::inc_claim_reduction::IncClaimReduction

use jolt_claims::protocols::jolt::lattice::relations::inc_virtualization::{
    IncVirtualization as IncVirtualizationSymbolic, IncVirtualizationChallenges,
    IncVirtualizationOutputClaims,
};
use jolt_claims::protocols::jolt::relations::claim_reductions::increments::IncClaimReductionInputClaims;
use jolt_claims::protocols::jolt::{
    geometry::dimensions::TraceDimensions, IncVirtualizationPublic, JoltDerivedId, JoltRelationId,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_poly::try_eq_mle;

use crate::stages::relations::{ConcreteSumcheck, GetPoint, OpeningClaim};
use crate::stages::{
    stage2::Stage2ClearOutput, stage4::Stage4ClearOutput, stage5::Stage5ClearOutput,
};
use crate::VerifierError;
use jolt_claims::protocols::jolt::LatticeJolt;

use super::inc_claim_reduction::inc_claim_reduction_inputs_from_upstream;

pub struct IncVirtualization<F: Field> {
    symbolic: IncVirtualizationSymbolic,
    ram_read_write_cycle: Vec<F>,
    ram_val_check_cycle: Vec<F>,
    registers_read_write_cycle: Vec<F>,
    registers_val_evaluation_cycle: Vec<F>,
}

impl<F: Field> IncVirtualization<F> {
    pub fn new(
        trace_dimensions: TraceDimensions,
        ram_read_write_cycle: Vec<F>,
        ram_val_check_cycle: Vec<F>,
        registers_read_write_cycle: Vec<F>,
        registers_val_evaluation_cycle: Vec<F>,
    ) -> Self {
        Self {
            symbolic: IncVirtualizationSymbolic::new(trace_dimensions),
            ram_read_write_cycle,
            ram_val_check_cycle,
            registers_read_write_cycle,
            registers_val_evaluation_cycle,
        }
    }
}

fn public_input_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::IncVirtualization,
        reason: reason.to_string(),
    }
}

impl<F: Field> ConcreteSumcheck<F> for IncVirtualization<F> {
    type Symbolic = IncVirtualizationSymbolic;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points<C: GetPoint<F>>(
        &self,
        sumcheck_point: &[F],
        _inputs: &IncClaimReductionInputClaims<C>,
    ) -> Result<IncVirtualizationOutputClaims<Vec<F>>, VerifierError> {
        // The fused column and its store selector share the cycle opening
        // point (the reversed sumcheck point).
        let opening_point = sumcheck_point.iter().rev().copied().collect::<Vec<_>>();
        Ok(IncVirtualizationOutputClaims {
            fused_inc: opening_point.clone(),
            store: opening_point,
        })
    }

    fn derive_output_term<C: GetPoint<F>>(
        &self,
        id: &JoltDerivedId,
        _inputs: &IncClaimReductionInputClaims<C>,
        outputs: &IncVirtualizationOutputClaims<OpeningClaim<F>>,
        _challenges: &IncVirtualizationChallenges<F>,
    ) -> Result<F, VerifierError> {
        let JoltDerivedId::IncVirtualization(public) = id else {
            return Err(VerifierError::MissingStageClaimDerived { id: *id });
        };
        let opening_point = outputs.fused_inc.point();
        let cycle = match public {
            IncVirtualizationPublic::EqRamReadWrite => &self.ram_read_write_cycle,
            IncVirtualizationPublic::EqRamValCheck => &self.ram_val_check_cycle,
            IncVirtualizationPublic::EqRegistersReadWrite => &self.registers_read_write_cycle,
            IncVirtualizationPublic::EqRegistersValEvaluation => {
                &self.registers_val_evaluation_cycle
            }
        };
        try_eq_mle(opening_point, cycle).map_err(public_input_failed)
    }
}

/// Drives the lattice-only `IncVirtualization` phase: a single-instance
/// batched sumcheck strictly between stage 5 and the stage-6 address phase,
/// so its `OpFlags(Store)` output exists when the six-stage read-raf address
/// fold is constructed.
pub fn verify_inc_virtualization_phase<F, PCS, VC, T, ZkProof, JOP, Cmts>(
    proof: &crate::proof::JoltProof<PCS, VC, ZkProof, JOP, Cmts, LatticeJolt>,
    trace_dimensions: TraceDimensions,
    inc_cycle_points: [Vec<F>; 4],
    transcript: &mut T,
    stage2: &Stage2ClearOutput<F>,
    stage4: &Stage4ClearOutput<F>,
    stage5: &Stage5ClearOutput<F>,
) -> Result<IncVirtualizationOutputClaims<OpeningClaim<F>>, VerifierError>
where
    F: jolt_field::Field,
    PCS: jolt_openings::CommitmentScheme<Field = F>,
    VC: jolt_crypto::VectorCommitment<Field = F>,
    T: jolt_transcript::Transcript<Challenge = F>,
{
    use crate::stages::relations::{zip_openings, OutputClaims as _};
    use jolt_sumcheck::{BatchedSumcheckVerifier, SumcheckClaim};

    let sumcheck_failed = |reason: String| VerifierError::StageClaimSumcheckFailed {
        stage: JoltRelationId::IncVirtualization,
        reason,
    };

    let [ram_read_write, ram_val_check, registers_read_write, registers_val_evaluation] =
        inc_cycle_points;
    let relation = IncVirtualization::new(
        trace_dimensions,
        ram_read_write,
        ram_val_check,
        registers_read_write,
        registers_val_evaluation,
    );
    let inputs = inc_claim_reduction_inputs_from_upstream(stage2, stage4, stage5);
    let challenges = relation.draw_challenges(transcript)?;
    let input_claim = relation.input_claim(&inputs, &challenges)?;

    let sumcheck_claims = vec![SumcheckClaim::new(
        relation.rounds(),
        relation.degree(),
        input_claim,
    )];
    let batch = BatchedSumcheckVerifier::verify_compressed_boolean(
        &sumcheck_claims,
        &proof.stages.inc_virtualization_proof,
        transcript,
    )
    .map_err(|error| sumcheck_failed(error.to_string()))?;

    let point = batch
        .try_instance_point(relation.rounds())
        .map_err(|error| sumcheck_failed(error.to_string()))?;
    let points = relation.derive_opening_points(point, &inputs)?;
    let wire = &proof.clear_claims()?.stage6.inc_claim_reduction;
    let outputs: IncVirtualizationOutputClaims<OpeningClaim<F>> = zip_openings(wire, &points);

    let expected = relation.expected_output(&inputs, &outputs, &challenges)?;
    if batch.batching_coefficients.len() != 1 {
        return Err(sumcheck_failed(format!(
            "single-instance phase returned {} batching coefficients",
            batch.batching_coefficients.len()
        )));
    }
    if batch.reduction.value != batch.batching_coefficients[0] * expected {
        return Err(VerifierError::StageClaimOutputMismatch {
            stage: JoltRelationId::IncVirtualization,
        });
    }

    for value in wire.opening_values() {
        transcript.append_labeled(b"opening_claim", &value);
    }
    Ok(outputs)
}
