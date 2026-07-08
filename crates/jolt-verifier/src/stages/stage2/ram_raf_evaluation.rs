//! The stage 2 `RamRafEvaluation` sumcheck instance.
//!
//! Owns the RAM RAF address opening-point derivation and the `UnmapAddress`
//! public-value computation, in lockstep with the BlindFold constraint's
//! `ram::raf_evaluation` formula. The phase-3 cycle scaling on the input is baked
//! into that formula's constant coefficient.
//!
//! The produced `ram_ra` opening point is `[r_address(log_k) ‖ tau_low(log_t)]`;
//! `UnmapAddress` reads only the address prefix.

use jolt_claims::protocols::jolt::relations;
pub use jolt_claims::protocols::jolt::relations::ram::{
    RamRafEvaluationInputClaims, RamRafEvaluationOutputClaims,
};
use jolt_claims::protocols::jolt::{
    geometry::{dimensions::ReadWriteDimensions, ram::RamRafEvaluationDimensions},
    JoltDerivedId, JoltRelationId, RamRafEvaluationPublic,
};
use jolt_claims::{NoChallenges, SymbolicSumcheck};
use jolt_field::Field;
use jolt_poly::{IdentityPolynomial, MultilinearEvaluation};

use crate::stages::relations::ConcreteSumcheck;
use crate::stages::stage1::Stage1ClearOutput;
use crate::VerifierError;

/// Wire the consumed RAM address opening *value* from stage 1's outer sumcheck.
/// (Verifier-side constructor for the moved [`RamRafEvaluationInputClaims`].)
pub fn ram_raf_evaluation_input_values_from_upstream<F: Field>(
    stage1: &Stage1ClearOutput<F>,
) -> RamRafEvaluationInputClaims<F> {
    RamRafEvaluationInputClaims {
        ram_address: stage1.output_values.outer_remainder.ram_address,
    }
}

pub struct RamRafEvaluation<F: Field> {
    symbolic: relations::ram::RafEvaluation,
    read_write_dimensions: ReadWriteDimensions,
    ram_log_k: usize,
    lowest_address: u64,
    tau_low: Vec<F>,
}

impl<F: Field> RamRafEvaluation<F> {
    pub fn new(
        read_write_dimensions: ReadWriteDimensions,
        raf_dimensions: RamRafEvaluationDimensions,
        ram_log_k: usize,
        lowest_address: u64,
        tau_low: Vec<F>,
    ) -> Self {
        Self {
            symbolic: relations::ram::RafEvaluation::new(raf_dimensions),
            read_write_dimensions,
            ram_log_k,
            lowest_address,
            tau_low,
        }
    }
}

fn public_input_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::RamRafEvaluation,
        reason: reason.to_string(),
    }
}

impl<F: Field> ConcreteSumcheck<F> for RamRafEvaluation<F> {
    type Symbolic = relations::ram::RafEvaluation;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    /// Delegates to [`super::phase1_instance_point_offset`] (the phase-1 sub-point
    /// slicing shared with `RamOutputCheck`).
    fn instance_point_offset(&self, batch_num_vars: usize) -> Result<usize, VerifierError> {
        super::phase1_instance_point_offset(self.read_write_dimensions, self.id(), batch_num_vars)
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        _input_points: &RamRafEvaluationInputClaims<Vec<F>>,
    ) -> Result<RamRafEvaluationOutputClaims<Vec<F>>, VerifierError> {
        let address = self
            .read_write_dimensions
            .address_opening_point(sumcheck_point)
            .map_err(public_input_failed)?;
        if address.len() != self.ram_log_k {
            return Err(public_input_failed(format!(
                "RAM RAF address point length mismatch: expected {}, got {}",
                self.ram_log_k,
                address.len()
            )));
        }
        let opening_point = [address.as_slice(), self.tau_low.as_slice()].concat();
        Ok(RamRafEvaluationOutputClaims {
            ram_ra: opening_point,
        })
    }

    fn derive_output_term(
        &self,
        id: &JoltDerivedId,
        _input_points: &RamRafEvaluationInputClaims<Vec<F>>,
        output_points: &RamRafEvaluationOutputClaims<Vec<F>>,
        _challenges: &NoChallenges<F>,
    ) -> Result<F, VerifierError> {
        let JoltDerivedId::RamRafEvaluation(public_id) = id else {
            return Err(VerifierError::MissingStageClaimDerived { id: *id });
        };
        match public_id {
            // The produced opening point is `[r_address(log_k) ‖ tau_low]`; the
            // unmap reads only the address prefix and lifts it back to a byte
            // address (`identity(r_address) * 8 + lowest_address`).
            RamRafEvaluationPublic::UnmapAddress => {
                let point = output_points.ram_ra();
                if point.len() < self.ram_log_k {
                    return Err(public_input_failed(format!(
                        "RAM RAF opening point is too short: expected at least {}, got {}",
                        self.ram_log_k,
                        point.len()
                    )));
                }
                let address = &point[..self.ram_log_k];
                Ok(
                    IdentityPolynomial::new(self.ram_log_k).evaluate(address) * F::from_u64(8)
                        + F::from_u64(self.lowest_address),
                )
            }
        }
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use jolt_field::Fr;

    /// The `instance_point_offset` override must reproduce the legacy phase-1
    /// slicing `(batch_num_vars - (log_t + log_k)) + phase1_num_rounds` the
    /// pre-port verifier computed via `try_round_offset(log_t + log_k)`.
    #[test]
    fn instance_point_offset_matches_legacy_phase1_formula() {
        for (log_t, log_k, phase1, phase2) in [(4usize, 3usize, 2usize, 1usize), (6, 5, 3, 2)] {
            let dimensions = ReadWriteDimensions::new(log_t, log_k, phase1, phase2);
            let raf_dimensions = RamRafEvaluationDimensions::try_from(dimensions).unwrap();
            let relation =
                RamRafEvaluation::<Fr>::new(dimensions, raf_dimensions, log_k, 0, Vec::new());
            // The real batch has `log_t + log_k` variables (the RAM read-write
            // leader); also probe a padded vector.
            for batch_num_vars in [log_t + log_k, log_t + log_k + 5] {
                let legacy = (batch_num_vars - (log_t + log_k)) + phase1;
                let offset = relation.instance_point_offset(batch_num_vars).unwrap();
                assert_eq!(offset, legacy);
                assert_eq!(offset + relation.rounds(), batch_num_vars);
            }
            assert!(relation.instance_point_offset(log_t + log_k - 1).is_err());
        }
    }
}
