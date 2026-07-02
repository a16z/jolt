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
    RamReadWriteChallenges, RamReadWriteInputClaims, RamReadWriteOutputClaims,
};
use jolt_claims::protocols::jolt::{
    geometry::dimensions::ReadWriteDimensions, JoltDerivedId, JoltRelationId, RamReadWritePublic,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_poly::try_eq_mle;

use crate::stages::relations::ConcreteSumcheck;
use crate::stages::stage1::Stage1ClearOutput;
use crate::VerifierError;

/// Wire the consumed RAM read/write value opening *values* from stage 1's outer
/// sumcheck. (Verifier-side constructor for the moved [`RamReadWriteInputClaims`].)
pub fn ram_read_write_input_values_from_upstream<F: Field>(
    stage1: &Stage1ClearOutput<F>,
) -> RamReadWriteInputClaims<F> {
    let outer = &stage1.output_values.outer_remainder;
    RamReadWriteInputClaims {
        ram_read_value: outer.ram_read_value,
        ram_write_value: outer.ram_write_value,
    }
}

/// Wire the consumed RAM read/write value opening *points* (both empty — these
/// openings carry no point at this stage, so no upstream data is needed and the
/// same wiring serves the clear and ZK paths).
pub fn ram_read_write_input_points_from_upstream<F: Field>() -> RamReadWriteInputClaims<Vec<F>> {
    RamReadWriteInputClaims {
        ram_read_value: Vec::new(),
        ram_write_value: Vec::new(),
    }
}

pub struct RamReadWriteChecking<F: Field> {
    symbolic: relations::ram::ReadWriteChecking,
    dimensions: ReadWriteDimensions,
    ram_log_k: usize,
    product_tau_low: Vec<F>,
}

impl<F: Field> RamReadWriteChecking<F> {
    pub fn new(dimensions: ReadWriteDimensions, ram_log_k: usize, product_tau_low: Vec<F>) -> Self {
        Self {
            symbolic: relations::ram::ReadWriteChecking::new(dimensions),
            dimensions,
            ram_log_k,
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

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        _input_points: &RamReadWriteInputClaims<Vec<F>>,
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

    fn derive_output_term(
        &self,
        id: &JoltDerivedId,
        _input_points: &RamReadWriteInputClaims<Vec<F>>,
        output_points: &RamReadWriteOutputClaims<Vec<F>>,
        _challenges: &RamReadWriteChallenges<F>,
    ) -> Result<F, VerifierError> {
        let JoltDerivedId::RamReadWrite(public_id) = id else {
            return Err(VerifierError::MissingStageClaimDerived { id: *id });
        };
        match public_id {
            // The opening point is `[r_address(log_k) || r_cycle(log_t)]`, so the
            // cycle sub-point is the suffix past the address bits.
            RamReadWritePublic::EqCycle => {
                let r_cycle = &output_points.val()[self.ram_log_k..];
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
        let relation =
            RamReadWriteChecking::<Fr>::new(ReadWriteDimensions::new(4, 3, 2, 1), 3, Vec::new());

        let (inline_events, inline_gamma) = record(|t| t.challenge_scalar());
        let (draw_events, challenges) = record(|t| relation.draw_challenges(t).unwrap());

        assert_eq!(draw_events, inline_events);
        assert_eq!(draw_events, vec![DrawEvent::Squeeze(1)]);
        assert_eq!(challenges.gamma, inline_gamma);
    }
}
