//! The stage 2 `RamReadWriteChecking` sumcheck instance.
//!
//! A self-contained relation object driven identically by the prover (while
//! producing the stage 2 batch proof) and the verifier (after checking it). It
//! owns the RAM read-write opening-point derivation and the `EqCycle` public-value
//! computation, so the input/output claim algebra lives here once instead of being
//! hand-coded on each side (and stays in lockstep with the BlindFold constraint,
//! which evaluates the same `ram::read_write_checking` formula).

use jolt_claims::protocols::jolt::relations;
pub use jolt_claims::protocols::jolt::relations::ram::{
    RamReadWriteInputClaims, RamReadWriteOutputClaims,
};
use jolt_claims::protocols::jolt::{
    geometry::dimensions::ReadWriteDimensions, JoltChallengeId, JoltDerivedId, JoltRelationId,
    RamReadWriteChallenge, RamReadWritePublic,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_poly::try_eq_mle;

use crate::stages::relations::{ConcreteSumcheck, GetPoint, OpeningClaim};
use crate::stages::stage1::Stage1ClearOutput;
use crate::VerifierError;

/// Wire the consumed RAM read/write value openings from stage 1's outer sumcheck.
/// (Verifier-side constructor for the moved [`RamReadWriteInputClaims`].)
pub fn ram_read_write_inputs_from_upstream<F: Field>(
    stage1: &Stage1ClearOutput<F>,
) -> RamReadWriteInputClaims<OpeningClaim<F>> {
    let value = |value: F| OpeningClaim {
        point: Vec::new(),
        value,
    };
    RamReadWriteInputClaims {
        ram_read_value: value(stage1.outer.ram_read_value),
        ram_write_value: value(stage1.outer.ram_write_value),
    }
}

pub struct RamReadWriteChecking<F: Field> {
    symbolic: relations::ram::ReadWriteChecking,
    dimensions: ReadWriteDimensions,
    ram_log_k: usize,
    gamma: F,
    product_tau_low: Vec<F>,
}

impl<F: Field> RamReadWriteChecking<F> {
    pub fn new(
        dimensions: ReadWriteDimensions,
        ram_log_k: usize,
        gamma: F,
        product_tau_low: Vec<F>,
    ) -> Self {
        Self {
            symbolic: relations::ram::ReadWriteChecking::new(dimensions),
            dimensions,
            ram_log_k,
            gamma,
            product_tau_low,
        }
    }
}

fn public_input_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::RamReadWriteChecking,
        reason: reason.to_string(),
    }
}

impl<F: Field> ConcreteSumcheck<F> for RamReadWriteChecking<F> {
    type Symbolic = relations::ram::ReadWriteChecking;
    type Inputs<C> = RamReadWriteInputClaims<C>;
    type Outputs<C> = RamReadWriteOutputClaims<C>;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points<C: GetPoint<F>>(
        &self,
        sumcheck_point: &[F],
        _inputs: &RamReadWriteInputClaims<C>,
    ) -> Result<RamReadWriteOutputClaims<Vec<F>>, VerifierError> {
        let opening_point = self
            .dimensions
            .read_write_opening_point(sumcheck_point)
            .map_err(public_input_failed)?
            .opening_point;
        Ok(RamReadWriteOutputClaims {
            val: opening_point.clone(),
            ra: opening_point.clone(),
            inc: opening_point,
        })
    }

    fn resolve_challenge(&self, id: &JoltChallengeId) -> Result<F, VerifierError> {
        match id {
            JoltChallengeId::RamReadWrite(RamReadWriteChallenge::Gamma) => Ok(self.gamma),
            _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        }
    }

    fn resolve_public<C: GetPoint<F>>(
        &self,
        id: &JoltDerivedId,
        _inputs: &RamReadWriteInputClaims<C>,
        outputs: Option<&RamReadWriteOutputClaims<OpeningClaim<F>>>,
    ) -> Result<F, VerifierError> {
        let outputs = outputs.ok_or(VerifierError::MissingStageClaimDerived { id: *id })?;
        let JoltDerivedId::RamReadWrite(public_id) = id else {
            return Err(VerifierError::MissingStageClaimDerived { id: *id });
        };
        match public_id {
            // The opening point is `[r_address(log_k) || r_cycle(log_t)]`, so the
            // cycle sub-point is the suffix past the address bits.
            RamReadWritePublic::EqCycle => {
                let r_cycle = &outputs.val.point()[self.ram_log_k..];
                try_eq_mle(&self.product_tau_low, r_cycle).map_err(public_input_failed)
            }
        }
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::stages::relations::draw_recording::{record, DrawEvent};
    use jolt_field::Fr;
    use jolt_transcript::Transcript;

    // Representative of the 14 single-`challenge_scalar` relations that inherit the
    // default `draw_challenges`: the inline `ram_read_write_gamma = challenge_scalar()`
    // is one squeeze, and the default's one-`challenge_scalar`-per-field draw stores
    // exactly that scalar.
    #[test]
    fn default_draw_challenges_matches_inline_ram_read_write_gamma() {
        let relation = RamReadWriteChecking::<Fr>::new(
            ReadWriteDimensions::new(4, 3, 2, 1),
            3,
            Fr::from(0u64),
            Vec::new(),
        );

        let (inline_events, inline_gamma) = record(|t| t.challenge_scalar());
        let (draw_events, challenges) = record(|t| relation.draw_challenges(t).unwrap());

        assert_eq!(draw_events, inline_events);
        assert_eq!(draw_events, vec![DrawEvent::Squeeze(1)]);
        assert_eq!(challenges.gamma, inline_gamma);
    }
}
