//! The stage 7 advice claim-reduction address phase.
//!
//! The trusted/untrusted advice two-phase reduction begins in stage 6b (cycle
//! phase) and, when active address-phase rounds remain, finishes here. The two
//! instances differ only by a runtime [`JoltAdviceKind`], so each is a separate
//! batch member built from a per-kind relation that carries the kind; the produced
//! and consumed openings are keyed positionally by kind (trusted/untrusted
//! `Option<C>` fields, exactly as stage 4's `RamValCheckAdviceClaims`), so the
//! claim structs stay fully derive-driven.
//!
//! As with the committed-program address phases, the `FinalScale` public is a
//! function of the reduction's final opening point, which `derive_output_term`
//! recovers from the output claims.

use jolt_claims::protocols::jolt::relations;
pub use jolt_claims::protocols::jolt::relations::claim_reductions::advice::{
    AdviceAddressPhaseInputClaims, AdviceAddressPhaseOutputClaims,
};
use jolt_claims::protocols::jolt::{
    AdviceClaimReductionLayout, AdviceClaimReductionPublic, JoltAdviceKind, JoltDerivedId,
    JoltRelationId, PrecommittedReductionLayout,
};
use jolt_claims::{NoChallenges, SymbolicSumcheck};
use jolt_field::Field;

use crate::stages::relations::{ConcreteSumcheck, GetPoint, OpeningClaim};
use crate::VerifierError;

pub struct AdviceAddressPhase<F: Field> {
    symbolic: relations::claim_reductions::advice::AddressPhase,
    kind: JoltAdviceKind,
    layout: AdviceClaimReductionLayout,
    cycle_phase_variables: Vec<F>,
    reference_opening_point: Vec<F>,
}

impl<F: Field> AdviceAddressPhase<F> {
    /// `reference_opening_point` is the RAM address point of the staged advice
    /// opening from RAM value-check (stage 4). It and the cycle-phase variables are
    /// known before the stage-7 sumcheck.
    pub fn new(
        kind: JoltAdviceKind,
        layout: &AdviceClaimReductionLayout,
        reference_opening_point: Vec<F>,
        cycle_phase_variables: Vec<F>,
    ) -> Self {
        Self {
            symbolic: relations::claim_reductions::advice::AddressPhase::new((
                kind,
                layout.dimensions(),
            )),
            kind,
            layout: layout.clone(),
            cycle_phase_variables,
            reference_opening_point,
        }
    }

    pub fn kind(&self) -> JoltAdviceKind {
        self.kind
    }

    /// This kind's produced opening point, recovered from the output claims.
    fn output_point<'a>(
        &self,
        outputs: &'a AdviceAddressPhaseOutputClaims<OpeningClaim<F>>,
    ) -> Result<&'a [F], VerifierError> {
        let claim = match self.kind {
            JoltAdviceKind::Trusted => outputs.trusted.as_ref(),
            JoltAdviceKind::Untrusted => outputs.untrusted.as_ref(),
        };
        claim
            .map(|opening| opening.point.as_slice())
            .ok_or_else(|| advice_public_failed("advice address phase produced no opening"))
    }
}

fn advice_public_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::AdviceClaimReduction,
        reason: reason.to_string(),
    }
}

impl<F: Field> ConcreteSumcheck<F> for AdviceAddressPhase<F> {
    type Symbolic = relations::claim_reductions::advice::AddressPhase;
    type Inputs<C> = AdviceAddressPhaseInputClaims<C>;
    type Outputs<C> = AdviceAddressPhaseOutputClaims<C>;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points<C: GetPoint<F>>(
        &self,
        sumcheck_point: &[F],
        _inputs: &AdviceAddressPhaseInputClaims<C>,
    ) -> Result<AdviceAddressPhaseOutputClaims<Vec<F>>, VerifierError> {
        let opening_point = self
            .layout
            .address_phase_opening_point(&self.cycle_phase_variables, sumcheck_point)
            .map_err(advice_public_failed)?;
        Ok(match self.kind {
            JoltAdviceKind::Trusted => AdviceAddressPhaseOutputClaims {
                trusted: Some(opening_point),
                untrusted: None,
            },
            JoltAdviceKind::Untrusted => AdviceAddressPhaseOutputClaims {
                trusted: None,
                untrusted: Some(opening_point),
            },
        })
    }

    fn derive_output_term<C: GetPoint<F>>(
        &self,
        id: &JoltDerivedId,
        _inputs: &AdviceAddressPhaseInputClaims<C>,
        outputs: &AdviceAddressPhaseOutputClaims<OpeningClaim<F>>,
        _challenges: &NoChallenges<F>,
    ) -> Result<F, VerifierError> {
        let JoltDerivedId::AdviceClaimReduction(AdviceClaimReductionPublic::FinalScale(kind)) = id
        else {
            return Err(VerifierError::MissingStageClaimDerived { id: *id });
        };
        if *kind != self.kind {
            return Err(VerifierError::MissingStageClaimDerived { id: *id });
        }
        self.layout
            .address_phase_scale_at_opening_point(
                &self.reference_opening_point,
                self.output_point(outputs)?,
            )
            .map_err(advice_public_failed)
    }
}
