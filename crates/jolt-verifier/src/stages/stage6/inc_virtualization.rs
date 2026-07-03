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
use crate::VerifierError;

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
