//! The stage 4 `RegistersReadWriteChecking` sumcheck instance.
//!
//! A self-contained relation object driven identically by the prover (while
//! producing the stage 4 batch proof) and the verifier (after checking it). It
//! owns the register read-write point derivation and the `EqCycle` public-value
//! computation, so the input/output claim algebra lives here once instead of
//! being hand-coded on each side.

use jolt_claims::protocols::jolt::relations;
pub use jolt_claims::protocols::jolt::relations::registers::{
    RegistersReadWriteChallenges, RegistersReadWriteInputClaims, RegistersReadWriteOutputClaims,
};
use jolt_claims::protocols::jolt::{
    geometry::dimensions::{ReadWriteDimensions, REGISTER_ADDRESS_BITS},
    JoltDerivedId, JoltRelationId, RegistersReadWritePublic,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_poly::try_eq_mle;

use crate::stages::relations::{ConcreteSumcheck, GetPoint, OpeningClaim};
use crate::stages::stage3::outputs::Stage3ClearOutput;
use crate::VerifierError;

/// Wire the consumed openings from stage 3's registers claim-reduction output,
/// all sharing that relation's opening point. (Verifier-side constructor for the
/// moved [`RegistersReadWriteInputClaims`].)
pub fn registers_read_write_inputs_from_upstream<F: Field>(
    stage3: &Stage3ClearOutput<F>,
) -> RegistersReadWriteInputClaims<OpeningClaim<F>> {
    // The stage-3 register-reduction openings already carry their shared
    // opening point (point + value), so the consumed claims are just clones.
    let reduction = &stage3.output_claims.registers_claim_reduction;
    RegistersReadWriteInputClaims {
        rd_write_value: reduction.rd_write_value.clone(),
        rs1_value: reduction.rs1_value.clone(),
        rs2_value: reduction.rs2_value.clone(),
    }
}

pub struct RegistersReadWriteChecking<F: Field> {
    symbolic: relations::registers::ReadWriteChecking,
    register_dimensions: ReadWriteDimensions,
    _field: core::marker::PhantomData<F>,
}

impl<F: Field> RegistersReadWriteChecking<F> {
    pub fn new(register_dimensions: ReadWriteDimensions) -> Self {
        Self {
            symbolic: relations::registers::ReadWriteChecking::new(register_dimensions),
            register_dimensions,
            _field: core::marker::PhantomData,
        }
    }
}

fn public_input_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::RegistersReadWriteChecking,
        reason: reason.to_string(),
    }
}

impl<F: Field> ConcreteSumcheck<F> for RegistersReadWriteChecking<F> {
    type Symbolic = relations::registers::ReadWriteChecking;
    type Inputs<C> = RegistersReadWriteInputClaims<C>;
    type Outputs<C> = RegistersReadWriteOutputClaims<C>;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points<C: GetPoint<F>>(
        &self,
        sumcheck_point: &[F],
        _inputs: &RegistersReadWriteInputClaims<C>,
    ) -> Result<RegistersReadWriteOutputClaims<Vec<F>>, VerifierError> {
        let opening_point = self
            .register_dimensions
            .read_write_opening_point(sumcheck_point)
            .map_err(public_input_failed)?
            .opening_point;
        Ok(RegistersReadWriteOutputClaims {
            registers_val: opening_point.clone(),
            rs1_ra: opening_point.clone(),
            rs2_ra: opening_point.clone(),
            rd_wa: opening_point.clone(),
            rd_inc: opening_point,
        })
    }

    fn resolve_public<C: GetPoint<F>>(
        &self,
        id: &JoltDerivedId,
        inputs: &RegistersReadWriteInputClaims<C>,
        outputs: Option<&RegistersReadWriteOutputClaims<OpeningClaim<F>>>,
        _challenges: &RegistersReadWriteChallenges<F>,
    ) -> Result<F, VerifierError> {
        let outputs = outputs.ok_or(VerifierError::MissingStageClaimDerived { id: *id })?;
        let JoltDerivedId::RegistersReadWrite(public_id) = id else {
            return Err(VerifierError::MissingStageClaimDerived { id: *id });
        };
        match public_id {
            RegistersReadWritePublic::EqCycle => {
                let fixed_cycle = inputs.rd_write_value.point();
                let registers_cycle = &outputs.registers_val.point()[REGISTER_ADDRESS_BITS..];
                try_eq_mle(fixed_cycle, registers_cycle).map_err(public_input_failed)
            }
        }
    }
}
