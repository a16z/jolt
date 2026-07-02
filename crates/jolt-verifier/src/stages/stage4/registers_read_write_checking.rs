//! The stage 4 `RegistersReadWriteChecking` sumcheck instance.
//!
//! Owns the register read-write point derivation and the `EqCycle` public-value
//! computation.

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

use crate::stages::relations::ConcreteSumcheck;
use crate::stages::stage3::{Stage3OutputClaims, Stage3OutputPoints};
use crate::VerifierError;

/// Wire the consumed opening *values* from stage 3's registers claim-reduction
/// output. Takes the ZK-agnostic stage-3 output-claims aggregate.
pub fn registers_read_write_input_values_from_upstream<F: Field>(
    stage3: &Stage3OutputClaims<F>,
) -> RegistersReadWriteInputClaims<F> {
    let reduction = &stage3.registers_claim_reduction;
    RegistersReadWriteInputClaims {
        rd_write_value: reduction.rd_write_value,
        rs1_value: reduction.rs1_value,
        rs2_value: reduction.rs2_value,
    }
}

/// Wire the consumed opening *points* from stage 3's registers claim-reduction
/// output, all sharing that relation's opening point. Takes the ZK-agnostic
/// stage-3 output-points aggregate.
pub fn registers_read_write_input_points_from_upstream<F: Field>(
    stage3: &Stage3OutputPoints<F>,
) -> RegistersReadWriteInputClaims<Vec<F>> {
    let reduction = &stage3.registers_claim_reduction;
    RegistersReadWriteInputClaims {
        rd_write_value: reduction.rd_write_value().to_vec(),
        rs1_value: reduction.rs1_value().to_vec(),
        rs2_value: reduction.rs2_value().to_vec(),
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

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        _input_points: &RegistersReadWriteInputClaims<Vec<F>>,
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

    fn derive_output_term(
        &self,
        id: &JoltDerivedId,
        input_points: &RegistersReadWriteInputClaims<Vec<F>>,
        output_points: &RegistersReadWriteOutputClaims<Vec<F>>,
        _challenges: &RegistersReadWriteChallenges<F>,
    ) -> Result<F, VerifierError> {
        let JoltDerivedId::RegistersReadWrite(public_id) = id else {
            return Err(VerifierError::MissingStageClaimDerived { id: *id });
        };
        match public_id {
            RegistersReadWritePublic::EqCycle => {
                let fixed_cycle = input_points.rd_write_value();
                let registers_cycle = &output_points.registers_val()[REGISTER_ADDRESS_BITS..];
                try_eq_mle(fixed_cycle, registers_cycle).map_err(public_input_failed)
            }
        }
    }
}
