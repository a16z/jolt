//! The stage 6b booleanity cycle-phase sumcheck instance.
//!
//! Booleanity proves every one-hot `Ra` chunk (instruction, bytecode, RAM) is
//! boolean. It runs in two phases: the stage-6a address phase binds the
//! `log_k_chunk` address variables and stages the `BooleanityAddrClaim`
//! intermediate; this stage-6b cycle phase binds the `log_t` cycle variables and
//! opens the committed per-family `Ra` claims. The cycle phase's single public,
//! `EqAddressCycle`, ties the full two-phase sumcheck point to the reference
//! address/cycle drawn from the stage-5 instruction opening.
//!
//! Under the `akita` feature the symbolic swaps to the lattice cycle phase,
//! which extends the same boolean fold over the unsigned-inc chunk and MSB
//! one-hot columns, all opened at the shared `(r_address ‖ r_cycle)` point.

#[cfg(feature = "akita")]
use jolt_claims::protocols::jolt::lattice::relations::booleanity as lattice_booleanity;
#[cfg(not(feature = "akita"))]
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

use crate::stages::relations::{ConcreteSumcheck, SumcheckInputPoints, SumcheckOutputPoints};
use crate::VerifierError;

#[cfg(not(feature = "akita"))]
type CyclePhaseSymbolic = relations::booleanity::BooleanityCyclePhase;
#[cfg(feature = "akita")]
type CyclePhaseSymbolic = lattice_booleanity::LatticeBooleanityCyclePhase;

/// The cycle phase's shape: the base dimensions, plus (akita) the inc
/// chunking they imply. The driver constructs it, keeping this member's
/// constructor infallible.
#[cfg(not(feature = "akita"))]
pub type BooleanityCycleDimensions = BooleanityDimensions;
#[cfg(feature = "akita")]
pub type BooleanityCycleDimensions = lattice_booleanity::LatticeBooleanityDimensions;

#[derive(Clone)]
pub struct Booleanity<F: Field> {
    symbolic: CyclePhaseSymbolic,
    dimensions: BooleanityCycleDimensions,
    /// The address opening prefix from the stage-6a phase.
    r_address: Vec<F>,
    /// The reference address/cycle the `EqAddressCycle` public compares against.
    reference_address: Vec<F>,
    reference_cycle: Vec<F>,
}

impl<F: Field> Booleanity<F> {
    pub fn new(
        dimensions: BooleanityCycleDimensions,
        r_address: Vec<F>,
        reference_address: Vec<F>,
        reference_cycle: Vec<F>,
    ) -> Self {
        Self {
            symbolic: CyclePhaseSymbolic::new(dimensions),
            dimensions,
            r_address,
            reference_address,
            reference_cycle,
        }
    }

    fn base_dimensions(&self) -> &BooleanityDimensions {
        #[cfg(not(feature = "akita"))]
        {
            &self.dimensions
        }
        #[cfg(feature = "akita")]
        {
            &self.dimensions.base
        }
    }

    pub fn dimensions(&self) -> BooleanityDimensions {
        *self.base_dimensions()
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

fn public_input_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::Booleanity,
        reason: reason.to_string(),
    }
}

impl<F: Field> ConcreteSumcheck<F> for Booleanity<F> {
    type Symbolic = CyclePhaseSymbolic;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        _input_points: &SumcheckInputPoints<F, Self>,
    ) -> Result<SumcheckOutputPoints<F, Self>, VerifierError> {
        let r_cycle = sumcheck_point.iter().rev().copied().collect::<Vec<_>>();
        let opening_point = [self.r_address.as_slice(), r_cycle.as_slice()].concat();
        let layout = self.base_dimensions().layout;
        #[cfg(not(feature = "akita"))]
        {
            let _ = &r_cycle;
            Ok(BooleanityOutputClaims {
                instruction_ra: vec![opening_point.clone(); layout.instruction()],
                bytecode_ra: vec![opening_point.clone(); layout.bytecode()],
                ram_ra: vec![opening_point; layout.ram()],
            })
        }
        #[cfg(feature = "akita")]
        Ok(lattice_booleanity::LatticeBooleanityOutputClaims {
            instruction_ra: vec![opening_point.clone(); layout.instruction()],
            bytecode_ra: vec![opening_point.clone(); layout.bytecode()],
            ram_ra: vec![opening_point.clone(); layout.ram()],
            unsigned_inc_chunks: vec![
                opening_point.clone();
                self.dimensions.chunking().chunk_count()
            ],
            unsigned_inc_msb: opening_point,
        })
    }

    fn derive_output_term(
        &self,
        id: &JoltDerivedId,
        _input_points: &SumcheckInputPoints<F, Self>,
        output_points: &SumcheckOutputPoints<F, Self>,
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
        let log_k_chunk = self.base_dimensions().log_k_chunk;
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
        let dimensions = BooleanityDimensions::new(layout, 3, 2);
        #[cfg(feature = "akita")]
        let dimensions = lattice_booleanity::LatticeBooleanityDimensions::new(dimensions).unwrap();
        let relation = Booleanity::<Fr>::new(dimensions, Vec::new(), Vec::new(), Vec::new());

        let (inline_events, inline_gamma) = record(|t| t.challenge());
        let (draw_events, challenges) = record(|t| relation.draw_challenges(t).unwrap());

        assert_eq!(draw_events, inline_events);
        assert_eq!(draw_events, vec![DrawEvent::Squeeze(1)]);
        assert_eq!(challenges.gamma, inline_gamma);
    }
}
