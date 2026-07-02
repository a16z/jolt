//! The stage 7 advice claim-reduction address phase.
//!
//! The trusted/untrusted advice two-phase reduction begins in stage 6b (cycle
//! phase) and, when active address-phase rounds remain, finishes here. The two
//! instances differ only by a runtime [`JoltAdviceKind`], so each is a separate
//! batch member built from a per-kind relation that carries the kind; the produced
//! and consumed openings are keyed positionally by kind (trusted/untrusted
//! `Option<C>` fields, exactly as the advice leaves folded into stage 4's
//! `RamValCheckOutputClaims`), so the claim structs stay fully derive-driven.
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

use crate::stages::relations::ConcreteSumcheck;
use crate::stages::stage6b::outputs::Stage6bOutputClaims;
use crate::VerifierError;

/// The consumed cycle-phase advice opening *value* for `kind`, in the shared
/// `AdviceAddressPhaseInputClaims` with only that kind's slot filled (the relation
/// reads only its own kind's field).
pub fn advice_input_values_from_upstream<F: Field>(
    cycle_phase: &Stage6bOutputClaims<F>,
    kind: JoltAdviceKind,
) -> AdviceAddressPhaseInputClaims<F> {
    let claim = cycle_phase.advice_cycle_phase_claim(kind);
    match kind {
        JoltAdviceKind::Trusted => AdviceAddressPhaseInputClaims {
            trusted: claim,
            untrusted: None,
        },
        JoltAdviceKind::Untrusted => AdviceAddressPhaseInputClaims {
            trusted: None,
            untrusted: claim,
        },
    }
}

pub struct AdviceAddressPhase<F: Field> {
    symbolic: relations::claim_reductions::advice::AddressPhase,
    kind: JoltAdviceKind,
    layout: AdviceClaimReductionLayout,
    cycle_phase_variables: Vec<F>,
    /// The RAM address point of the staged advice opening from RAM value-check
    /// (stage 4). Consumed only by the clear-only `derive_output_term` (`FinalScale`),
    /// so it is `None` in ZK — where BlindFold recomputes the scale independently and
    /// this relation's `derive_output_term` never runs.
    reference_opening_point: Option<Vec<F>>,
}

impl<F: Field> AdviceAddressPhase<F> {
    /// `reference_opening_point` is the RAM address point of the staged advice
    /// opening from RAM value-check (stage 4), `None` in ZK (clear-only aux). It and
    /// the cycle-phase variables are known before the stage-7 sumcheck.
    pub fn new(
        kind: JoltAdviceKind,
        layout: &AdviceClaimReductionLayout,
        reference_opening_point: Option<Vec<F>>,
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

    /// This kind's produced opening point, recovered from the output points.
    fn output_point<'a>(
        &self,
        output_points: &'a AdviceAddressPhaseOutputClaims<Vec<F>>,
    ) -> Result<&'a [F], VerifierError> {
        let point = match self.kind {
            JoltAdviceKind::Trusted => output_points.trusted(),
            JoltAdviceKind::Untrusted => output_points.untrusted(),
        };
        point.ok_or_else(|| advice_public_failed("advice address phase produced no opening"))
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
        _input_points: &AdviceAddressPhaseInputClaims<Vec<F>>,
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

    fn derive_output_term(
        &self,
        id: &JoltDerivedId,
        _input_points: &AdviceAddressPhaseInputClaims<Vec<F>>,
        output_points: &AdviceAddressPhaseOutputClaims<Vec<F>>,
        _challenges: &NoChallenges<F>,
    ) -> Result<F, VerifierError> {
        let JoltDerivedId::AdviceClaimReduction(AdviceClaimReductionPublic::FinalScale(kind)) = id
        else {
            return Err(VerifierError::MissingStageClaimDerived { id: *id });
        };
        if *kind != self.kind {
            return Err(VerifierError::MissingStageClaimDerived { id: *id });
        }
        let reference_opening_point = self.reference_opening_point.as_ref().ok_or_else(|| {
            advice_public_failed(
                "advice address phase has no reference opening point (ZK-only construction)",
            )
        })?;
        self.layout
            .address_phase_scale_at_opening_point(
                reference_opening_point,
                self.output_point(output_points)?,
            )
            .map_err(advice_public_failed)
    }
}
