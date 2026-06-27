//! The stage 2 `RamRafEvaluation` sumcheck instance.
//!
//! A self-contained relation object driven identically by the prover (while
//! producing the stage 2 batch proof) and the verifier (after checking it). It
//! owns the RAM RAF address opening-point derivation and the `UnmapAddress`
//! public-value computation, so the input/output claim algebra lives here once (and
//! stays in lockstep with the BlindFold constraint, which evaluates the same
//! `ram::raf_evaluation` formula). The phase-3 cycle scaling on the input is baked
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

use crate::stages::relations::{ConcreteSumcheck, GetPoint, OpeningClaim};
use crate::stages::stage1::Stage1ClearOutput;
use crate::VerifierError;

/// Wire the consumed RAM address opening from stage 1's outer sumcheck.
/// (Verifier-side constructor for the moved [`RamRafEvaluationInputClaims`].)
pub fn ram_raf_evaluation_inputs_from_upstream<F: Field>(
    stage1: &Stage1ClearOutput<F>,
) -> RamRafEvaluationInputClaims<OpeningClaim<F>> {
    RamRafEvaluationInputClaims {
        ram_address: OpeningClaim {
            point: Vec::new(),
            value: stage1.outer.ram_address,
        },
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
    type Inputs<C> = RamRafEvaluationInputClaims<C>;
    type Outputs<C> = RamRafEvaluationOutputClaims<C>;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points<C: GetPoint<F>>(
        &self,
        sumcheck_point: &[F],
        _inputs: &RamRafEvaluationInputClaims<C>,
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

    fn derive_output_term<C: GetPoint<F>>(
        &self,
        id: &JoltDerivedId,
        _inputs: &RamRafEvaluationInputClaims<C>,
        outputs: &RamRafEvaluationOutputClaims<OpeningClaim<F>>,
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
                let point = outputs.ram_ra.point();
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
