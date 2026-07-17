//! The inc-virtualization stage (packed path): its own single-instance sumcheck
//! stage between stage 5 and stage 6a, that virtualizes the four reduced `Inc`
//! claims into the committed `FusedInc` stream and its `OpFlags(Store)`
//! destination selector. `RdInc`/`RamInc` are never PCS-opened on the packed
//! path; this stage is where their claims leave the base PIOP. The store
//! selector claim feeds stage 6a's read-raf sixth val stage, and its cycle point
//! (this stage's bound point) anchors the stage-7 chunk reconstruction's decode
//! legs.

use jolt_claims::protocols::jolt::geometry::dimensions::TraceDimensions;
use jolt_claims::protocols::jolt::geometry::dimensions::REGISTER_ADDRESS_BITS;
use jolt_claims::protocols::jolt::lattice::relations::inc_virtualization::{
    IncVirtualization, IncVirtualizationChallenges, IncVirtualizationOutputClaims,
};
use jolt_claims::protocols::jolt::relations::claim_reductions::increments::IncClaimReductionInputClaims;
use jolt_claims::protocols::jolt::{IncVirtualizationPublic, JoltDerivedId, JoltRelationId};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_poly::try_eq_mle;
use jolt_sumcheck::SumcheckProof;
use jolt_transcript::Transcript;

use crate::stages::relations::{ConcreteSumcheck, OutputAppend, SumcheckBatch};
use crate::stages::stage2::outputs::Stage2ClearOutput;
use crate::stages::stage4::outputs::Stage4ClearOutput;
use crate::stages::stage5::outputs::Stage5ClearOutput;
use crate::stages::stage6b::inc_claim_reduction::{
    inc_claim_reduction_input_points_from_upstream, inc_claim_reduction_input_values_from_upstream,
};
use crate::verifier::CheckedInputs;
use crate::VerifierError;

fn public_input_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::IncVirtualization,
        reason: reason.to_string(),
    }
}

/// The phase's one sumcheck instance: the lattice `IncVirtualization`
/// relation, carrying the four consuming relations' cycle points for its
/// per-source `Eq` publics.
pub struct IncVirtualizationInstance<F: Field> {
    symbolic: IncVirtualization,
    ram_read_write_cycle: Vec<F>,
    ram_val_check_cycle: Vec<F>,
    registers_read_write_cycle: Vec<F>,
    registers_val_evaluation_cycle: Vec<F>,
}

impl<F: Field> ConcreteSumcheck<F> for IncVirtualizationInstance<F> {
    type Symbolic = IncVirtualization;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        _input_points: &IncClaimReductionInputClaims<Vec<F>>,
    ) -> Result<IncVirtualizationOutputClaims<Vec<F>>, VerifierError> {
        // The fused stream and its selector share the cycle opening point (the
        // reversed sumcheck point).
        let opening_point = sumcheck_point.iter().rev().copied().collect::<Vec<_>>();
        Ok(IncVirtualizationOutputClaims {
            fused_inc: opening_point.clone(),
            store: opening_point,
        })
    }

    fn derive_output_term(
        &self,
        id: &JoltDerivedId,
        _input_points: &IncClaimReductionInputClaims<Vec<F>>,
        output_points: &IncVirtualizationOutputClaims<Vec<F>>,
        _challenges: &IncVirtualizationChallenges<F>,
    ) -> Result<F, VerifierError> {
        let JoltDerivedId::IncVirtualization(public) = id else {
            return Err(VerifierError::MissingStageClaimDerived { id: *id });
        };
        let opening_point = output_points.fused_inc();
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

/// Source-of-truth for the phase's single-instance batch.
#[derive(SumcheckBatch)]
pub struct IncVirtualizationPhaseSumchecks<F: Field> {
    pub inc_virtualization: IncVirtualizationInstance<F>,
}

/// The stage's verified openings: the `FusedInc` and store-selector claims at
/// the stage's bound cycle point, consumed by stage 6a's read-raf (store
/// claim), the stage-6b store cycle binding, and the stage-7 chunk
/// reconstruction (fused claim + this point).
pub struct IncVirtualizationOutput<F: Field> {
    pub output_values: IncVirtualizationOutputClaims<F>,
    pub output_points: IncVirtualizationOutputClaims<Vec<F>>,
}

pub fn verify<F, C, T>(
    checked: &CheckedInputs,
    sumcheck_proof: &SumcheckProof<F, C>,
    claims: &IncVirtualizationPhaseOutputClaims<F>,
    transcript: &mut T,
    stage2: &Stage2ClearOutput<F>,
    stage4: &Stage4ClearOutput<F>,
    stage5: &Stage5ClearOutput<F>,
) -> Result<IncVirtualizationOutput<F>, VerifierError>
where
    F: Field,
    C: Clone + jolt_transcript::AppendToTranscript,
    T: Transcript<Challenge = F>,
{
    let log_t = checked.trace_length.ilog2() as usize;
    let trace_dimensions = TraceDimensions::new(log_t);

    // The consumed claims are exactly the base inc claim reduction's (this
    // phase replaces it on the packed path), so its upstream wiring is reused.
    let input_values = inc_claim_reduction_input_values_from_upstream(
        &stage2.output_values,
        &stage4.output_values,
        &stage5.output_values,
    );
    let input_points = inc_claim_reduction_input_points_from_upstream(
        &stage2.output_points,
        &stage4.output_points,
        &stage5.output_points,
    );

    // The consuming relations bind `(address ‖ cycle)`; the eq kernels of
    // this phase run over the cycle suffix only (the `Inc` streams are
    // cycle-indexed), so each upstream point is split at its address width.
    let log_k = checked.ram_K.ilog2() as usize;
    let cycle_suffix = |label: &'static str, point: &[F], address_vars: usize| {
        if point.len() < address_vars {
            return Err(VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::IncVirtualization,
                reason: format!(
                    "{label} has {} variables, expected at least {address_vars}",
                    point.len()
                ),
            });
        }
        Ok(point[address_vars..].to_vec())
    };
    let sumchecks = IncVirtualizationPhaseSumchecks {
        inc_virtualization: IncVirtualizationInstance {
            symbolic: IncVirtualization::new(trace_dimensions),
            ram_read_write_cycle: cycle_suffix(
                "IncVirtualization RAM read-write opening",
                input_points.ram_inc_read_write(),
                log_k,
            )?,
            ram_val_check_cycle: cycle_suffix(
                "IncVirtualization RAM value-check opening",
                input_points.ram_inc_val_check(),
                log_k,
            )?,
            registers_read_write_cycle: cycle_suffix(
                "IncVirtualization register read-write opening",
                input_points.rd_inc_read_write(),
                REGISTER_ADDRESS_BITS,
            )?,
            registers_val_evaluation_cycle: cycle_suffix(
                "IncVirtualization register value-evaluation opening",
                input_points.rd_inc_val_evaluation(),
                REGISTER_ADDRESS_BITS,
            )?,
        },
    };

    let challenges = sumchecks.draw_challenges(transcript)?;
    sumchecks.validate_output_claims(claims)?;

    let phase_inputs = IncVirtualizationPhaseInputClaims {
        inc_virtualization: input_values,
    };
    let phase_input_points = IncVirtualizationPhaseInputPoints {
        inc_virtualization: input_points,
    };

    let output_points = sumchecks.verify_clear(
        &phase_inputs,
        &phase_input_points,
        &challenges,
        claims,
        sumcheck_proof,
        transcript,
        6,
    )?;
    claims.inc_virtualization.append_openings(transcript);

    Ok(IncVirtualizationOutput {
        output_values: claims.inc_virtualization.clone(),
        output_points: output_points.inc_virtualization,
    })
}
