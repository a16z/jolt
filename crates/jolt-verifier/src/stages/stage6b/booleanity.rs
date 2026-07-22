//! The stage 6b booleanity cycle-phase sumcheck instance.
//!
//! Booleanity proves every one-hot `Ra` chunk (instruction, bytecode, RAM) is
//! boolean. It runs in two phases: the stage-6a address phase binds the
//! `log_k_chunk` address variables and stages the `BooleanityAddrClaim`
//! intermediate; this stage-6b cycle phase binds the `log_t` cycle variables and
//! opens the committed per-family `Ra` claims. The cycle phase's single public,
//! `EqAddressCycle`, ties the full two-phase sumcheck point to the reference
//! address/cycle drawn from the stage-5 instruction opening.

use jolt_claims::protocols::jolt::relations;
pub use jolt_claims::protocols::jolt::relations::booleanity::{
    BooleanityCyclePhaseChallenges, BooleanityInputClaims, BooleanityOutputClaims,
};
use jolt_claims::protocols::jolt::{
    geometry::booleanity::BooleanityDimensions, BooleanityPublic, JoltDerivedId, JoltRelationId,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_poly::try_eq_mle;

use crate::stages::relations::ConcreteSumcheck;
use crate::VerifierError;

#[derive(Clone)]
pub struct Booleanity<F: Field> {
    symbolic: relations::booleanity::BooleanityCyclePhase,
    dimensions: BooleanityDimensions,
    /// The address opening prefix from the stage-6a phase.
    r_address: Vec<F>,
    /// The reference address/cycle the `EqAddressCycle` public compares against.
    reference_address: Vec<F>,
    reference_cycle: Vec<F>,
}

impl<F: Field> Booleanity<F> {
    pub fn new(
        dimensions: BooleanityDimensions,
        r_address: Vec<F>,
        reference_address: Vec<F>,
        reference_cycle: Vec<F>,
    ) -> Self {
        Self {
            symbolic: relations::booleanity::BooleanityCyclePhase::new(dimensions),
            dimensions,
            r_address,
            reference_address,
            reference_cycle,
        }
    }
}

fn public_input_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::Booleanity,
        reason: reason.to_string(),
    }
}

impl<F: Field> Booleanity<F> {
    pub fn dimensions(&self) -> BooleanityDimensions {
        self.dimensions
    }

    pub fn r_address(&self) -> &[F] {
        &self.r_address
    }

    pub fn reference_address(&self) -> &[F] {
        &self.reference_address
    }

    pub fn reference_cycle(&self) -> &[F] {
        &self.reference_cycle
    }
}

impl<F: Field> ConcreteSumcheck<F> for Booleanity<F> {
    type Symbolic = relations::booleanity::BooleanityCyclePhase;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        _input_points: &BooleanityInputClaims<Vec<F>>,
    ) -> Result<BooleanityOutputClaims<Vec<F>>, VerifierError> {
        let r_cycle = sumcheck_point.iter().rev().copied().collect::<Vec<_>>();
        let opening_point = [self.r_address.as_slice(), r_cycle.as_slice()].concat();
        let layout = self.dimensions.layout;
        Ok(BooleanityOutputClaims {
            instruction_ra: vec![opening_point.clone(); layout.instruction()],
            bytecode_ra: vec![opening_point.clone(); layout.bytecode()],
            ram_ra: vec![opening_point; layout.ram()],
        })
    }

    fn derive_output_term(
        &self,
        id: &JoltDerivedId,
        _input_points: &BooleanityInputClaims<Vec<F>>,
        output_points: &BooleanityOutputClaims<Vec<F>>,
        _challenges: &BooleanityCyclePhaseChallenges<F>,
    ) -> Result<F, VerifierError> {
        let JoltDerivedId::Booleanity(BooleanityPublic::EqAddressCycle) = id else {
            return Err(VerifierError::MissingStageClaimDerived { id: *id });
        };
        // Recover the raw two-phase sumcheck point from a produced opening point
        // (`r_address ++ r_cycle`): each half is the reverse of its phase's
        // sumcheck sub-point, and `EqAddressCycle` compares `[6a ++ 6b]` against
        // `reversed(reference_address) ++ reversed(reference_cycle)`.
        let opening_point = output_points
            .instruction_ra()
            .first()
            .or_else(|| output_points.bytecode_ra().first())
            .or_else(|| output_points.ram_ra().first())
            .ok_or_else(|| public_input_failed("booleanity produced no openings"))?;
        let log_k_chunk = self.dimensions.log_k_chunk;
        let (r_address, r_cycle) = opening_point.split_at(log_k_chunk);
        let full_sumcheck_point = r_address
            .iter()
            .rev()
            .chain(r_cycle.iter().rev())
            .copied()
            .collect::<Vec<_>>();
        let reference_eq_point = self
            .reference_address
            .iter()
            .rev()
            .chain(self.reference_cycle.iter().rev())
            .copied()
            .collect::<Vec<_>>();
        try_eq_mle(&full_sumcheck_point, &reference_eq_point).map_err(public_input_failed)
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::stages::relations::draw_recording::{record, DrawEvent};
    use jolt_claims::protocols::jolt::geometry::ra::JoltRaPolynomialLayout;
    use jolt_field::Fr;
    use jolt_transcript::Transcript;

    // Booleanity inherits the default `draw_challenges` (one `challenge_scalar`): the
    // inline draw is a single `challenge()`. The historical zero-gamma re-roll was
    // dropped — a real Fiat-Shamir transcript never yields zero, and nothing else
    // checks for it.
    #[test]
    fn default_draw_challenges_matches_inline_booleanity_gamma() {
        let layout = JoltRaPolynomialLayout::new(1, 1, 1).unwrap();
        let relation = Booleanity::<Fr>::new(
            BooleanityDimensions::new(layout, 3, 2),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        );

        let (inline_events, inline_gamma) = record(|t| t.challenge());
        let (draw_events, challenges) = record(|t| relation.draw_challenges(t).unwrap());

        assert_eq!(draw_events, inline_events);
        assert_eq!(draw_events, vec![DrawEvent::Squeeze(1)]);
        assert_eq!(challenges.gamma, inline_gamma);
    }
}
