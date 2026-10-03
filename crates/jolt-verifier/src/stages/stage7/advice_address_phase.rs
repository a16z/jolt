#[cfg(not(feature = "akita"))]
use jolt_claims::protocols::jolt::geometry::claim_reductions::advice::cycle_phase_advice_opening;
use jolt_claims::protocols::jolt::relations;
pub use jolt_claims::protocols::jolt::relations::claim_reductions::advice::{
    TrustedAdviceAddressPhaseInputClaims, TrustedAdviceAddressPhaseOutputClaims,
    UntrustedAdviceAddressPhaseInputClaims, UntrustedAdviceAddressPhaseOutputClaims,
};
use jolt_claims::protocols::jolt::{
    AdviceClaimReductionLayout, AdviceClaimReductionPublic, JoltAdviceKind, JoltDerivedId,
    JoltRelationId, PrecommittedReductionLayout,
};
use jolt_claims::{NoChallenges, SymbolicSumcheck};
use jolt_field::JoltField;

use crate::stages::relations::ConcreteSumcheck;
#[cfg(not(feature = "akita"))]
use crate::stages::stage6b::outputs::Stage6bOutputClaims;
use crate::VerifierError;

#[cfg(not(feature = "akita"))]
pub fn trusted_advice_input_values_from_upstream<F: JoltField>(
    cycle_phase: &Stage6bOutputClaims<F>,
) -> Result<TrustedAdviceAddressPhaseInputClaims<F>, VerifierError> {
    let trusted = cycle_phase
        .advice_cycle_phase_claim(JoltAdviceKind::Trusted)
        .ok_or(VerifierError::MissingOpeningClaim {
            id: cycle_phase_advice_opening(JoltAdviceKind::Trusted).into(),
        })?;
    Ok(TrustedAdviceAddressPhaseInputClaims { trusted })
}

#[cfg(not(feature = "akita"))]
pub fn untrusted_advice_input_values_from_upstream<F: JoltField>(
    cycle_phase: &Stage6bOutputClaims<F>,
) -> Result<UntrustedAdviceAddressPhaseInputClaims<F>, VerifierError> {
    let untrusted = cycle_phase
        .advice_cycle_phase_claim(JoltAdviceKind::Untrusted)
        .ok_or(VerifierError::MissingOpeningClaim {
            id: cycle_phase_advice_opening(JoltAdviceKind::Untrusted).into(),
        })?;
    Ok(UntrustedAdviceAddressPhaseInputClaims { untrusted })
}

fn advice_public_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::AdviceClaimReduction,
        reason: reason.to_string(),
    }
}

#[derive(Clone)]
pub struct TrustedAdviceAddressPhase<F: JoltField> {
    symbolic: relations::claim_reductions::advice::TrustedAddressPhase,
    layout: AdviceClaimReductionLayout,
    cycle_phase_variables: Vec<F>,
    reference_opening_point: Option<Vec<F>>,
}

impl<F: JoltField> TrustedAdviceAddressPhase<F> {
    pub fn new(
        layout: &AdviceClaimReductionLayout,
        reference_opening_point: Option<Vec<F>>,
        cycle_phase_variables: Vec<F>,
    ) -> Self {
        Self {
            symbolic: relations::claim_reductions::advice::TrustedAddressPhase::new(
                layout.dimensions(),
            ),
            layout: layout.clone(),
            cycle_phase_variables,
            reference_opening_point,
        }
    }
}

impl<F: JoltField> ConcreteSumcheck<F> for TrustedAdviceAddressPhase<F> {
    type Symbolic = relations::claim_reductions::advice::TrustedAddressPhase;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    /// The advice address phase is bound on the offset-0 prefix of the batch
    /// challenge vector (two-phase reductions front-load the address rounds), not
    /// the front-loaded suffix.
    fn instance_point_offset(&self, _batch_num_vars: usize) -> Result<usize, VerifierError> {
        Ok(0)
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        _input_points: &TrustedAdviceAddressPhaseInputClaims<Vec<F>>,
    ) -> Result<TrustedAdviceAddressPhaseOutputClaims<Vec<F>>, VerifierError> {
        let opening_point = self
            .layout
            .address_phase_opening_point(&self.cycle_phase_variables, sumcheck_point)
            .map_err(advice_public_failed)?;
        Ok(TrustedAdviceAddressPhaseOutputClaims {
            trusted: opening_point,
        })
    }

    fn derive_output_term(
        &self,
        id: &JoltDerivedId,
        _input_points: &TrustedAdviceAddressPhaseInputClaims<Vec<F>>,
        output_points: &TrustedAdviceAddressPhaseOutputClaims<Vec<F>>,
        _challenges: &NoChallenges<F>,
    ) -> Result<F, VerifierError> {
        let JoltDerivedId::AdviceClaimReduction(AdviceClaimReductionPublic::FinalScale(
            JoltAdviceKind::Trusted,
        )) = id
        else {
            return Err(VerifierError::MissingStageClaimDerived { id: (*id).into() });
        };
        let reference_opening_point = self.reference_opening_point.as_ref().ok_or_else(|| {
            advice_public_failed(
                "advice address phase has no reference opening point (ZK-only construction)",
            )
        })?;
        self.layout
            .address_phase_scale_at_opening_point(reference_opening_point, output_points.trusted())
            .map_err(advice_public_failed)
    }
}

#[derive(Clone)]
pub struct UntrustedAdviceAddressPhase<F: JoltField> {
    symbolic: relations::claim_reductions::advice::UntrustedAddressPhase,
    layout: AdviceClaimReductionLayout,
    cycle_phase_variables: Vec<F>,
    reference_opening_point: Option<Vec<F>>,
}

impl<F: JoltField> UntrustedAdviceAddressPhase<F> {
    pub fn new(
        layout: &AdviceClaimReductionLayout,
        reference_opening_point: Option<Vec<F>>,
        cycle_phase_variables: Vec<F>,
    ) -> Self {
        Self {
            symbolic: relations::claim_reductions::advice::UntrustedAddressPhase::new(
                layout.dimensions(),
            ),
            layout: layout.clone(),
            cycle_phase_variables,
            reference_opening_point,
        }
    }
}

impl<F: JoltField> ConcreteSumcheck<F> for UntrustedAdviceAddressPhase<F> {
    type Symbolic = relations::claim_reductions::advice::UntrustedAddressPhase;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn instance_point_offset(&self, _batch_num_vars: usize) -> Result<usize, VerifierError> {
        Ok(0)
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        _input_points: &UntrustedAdviceAddressPhaseInputClaims<Vec<F>>,
    ) -> Result<UntrustedAdviceAddressPhaseOutputClaims<Vec<F>>, VerifierError> {
        let opening_point = self
            .layout
            .address_phase_opening_point(&self.cycle_phase_variables, sumcheck_point)
            .map_err(advice_public_failed)?;
        Ok(UntrustedAdviceAddressPhaseOutputClaims {
            untrusted: opening_point,
        })
    }

    fn derive_output_term(
        &self,
        id: &JoltDerivedId,
        _input_points: &UntrustedAdviceAddressPhaseInputClaims<Vec<F>>,
        output_points: &UntrustedAdviceAddressPhaseOutputClaims<Vec<F>>,
        _challenges: &NoChallenges<F>,
    ) -> Result<F, VerifierError> {
        let JoltDerivedId::AdviceClaimReduction(AdviceClaimReductionPublic::FinalScale(
            JoltAdviceKind::Untrusted,
        )) = id
        else {
            return Err(VerifierError::MissingStageClaimDerived { id: (*id).into() });
        };
        let reference_opening_point = self.reference_opening_point.as_ref().ok_or_else(|| {
            advice_public_failed(
                "advice address phase has no reference opening point (ZK-only construction)",
            )
        })?;
        self.layout
            .address_phase_scale_at_opening_point(
                reference_opening_point,
                output_points.untrusted(),
            )
            .map_err(advice_public_failed)
    }
}
