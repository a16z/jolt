use jolt_field::Field;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck_prover::{
    BackendError as UniversalBackendError, BatchedSumcheckSpec, SumcheckBackend as UniversalSumcheckBackend,
};

use crate::{Backend, BackendError, Stage4ReadWriteSumcheckBackend};

/// Stage 4 batched sumcheck over pre-materialized register and RAM states.
///
/// Materialization stays in stage code until [`ProverProgram`] lowering lands;
/// this adapter only dispatches `evaluate` / `bind` through the existing CPU kernels.
pub struct PreMaterializedStage4Backend<F, B> {
    inner: B,
    registers_state: B::RegistersReadWriteState,
    ram_state: B::RamValCheckState,
    _marker: core::marker::PhantomData<F>,
}

impl<F, B> PreMaterializedStage4Backend<F, B>
where
    F: Field,
    B: Stage4ReadWriteSumcheckBackend<F>,
{
    pub fn new(
        inner: B,
        registers_state: B::RegistersReadWriteState,
        ram_state: B::RamValCheckState,
    ) -> Self {
        Self {
            inner,
            registers_state,
            ram_state,
            _marker: core::marker::PhantomData,
        }
    }
}

impl<F, B> UniversalSumcheckBackend<F> for PreMaterializedStage4Backend<F, B>
where
    F: Field,
    B: Stage4ReadWriteSumcheckBackend<F>,
{
    type State = ();

    fn start(
        &mut self,
        spec: &BatchedSumcheckSpec<F>,
    ) -> Result<Self::State, UniversalBackendError> {
        validate_stage4_spec(spec)?;
        Ok(())
    }

    fn round_polynomials(
        &mut self,
        _state: &Self::State,
        _round: usize,
        active: &[usize],
        claims: &[F],
    ) -> Result<Vec<UnivariatePoly<F>>, UniversalBackendError> {
        if active.len() != claims.len() {
            return Err(UniversalBackendError::RoundPolynomialCountMismatch {
                expected: active.len(),
                got: claims.len(),
            });
        }

        active
            .iter()
            .zip(claims)
            .map(|(&instance_index, &previous_claim)| {
                evaluate_stage4_instance(self, instance_index, previous_claim)
                    .map_err(map_backend_error)
            })
            .collect()
    }

    fn bind(
        &mut self,
        _state: &mut Self::State,
        _round: usize,
        instance: usize,
        challenge: F,
    ) -> Result<(), UniversalBackendError> {
        match instance {
            0 => self
                .inner
                .bind_sumcheck_registers_read_write_state(&mut self.registers_state, challenge)
                .map_err(map_backend_error),
            1 => self
                .inner
                .bind_sumcheck_ram_val_check_state(&mut self.ram_state, challenge)
                .map_err(map_backend_error),
            _ => Err(UniversalBackendError::InvalidActiveIndex {
                index: instance,
                batch_size: 2,
            }),
        }
    }
}

fn validate_stage4_spec<F: Field>(
    spec: &BatchedSumcheckSpec<F>,
) -> Result<(), UniversalBackendError> {
    if spec.instances.len() != 2 {
        return Err(UniversalBackendError::UnsupportedRelation {
            label: spec.label,
        });
    }
    let [registers, ram] = spec.instances.as_slice() else {
        return Err(UniversalBackendError::UnsupportedRelation {
            label: spec.label,
        });
    };
    if registers.relation != "registers::read_write_checking"
        && registers.relation != "stage4.registers_read_write"
    {
        return Err(UniversalBackendError::UnsupportedRelation {
            label: registers.relation,
        });
    }
    if ram.relation != "ram::val_check" && ram.relation != "stage4.ram_val_check" {
        return Err(UniversalBackendError::UnsupportedRelation {
            label: ram.relation,
        });
    }
    Ok(())
}

fn evaluate_stage4_instance<F, B>(
    backend: &mut PreMaterializedStage4Backend<F, B>,
    instance_index: usize,
    previous_claim: F,
) -> Result<UnivariatePoly<F>, BackendError>
where
    F: Field,
    B: Stage4ReadWriteSumcheckBackend<F>,
{
    match instance_index {
        0 => backend
            .inner
            .evaluate_sumcheck_registers_read_write_round(
                &backend.registers_state,
                previous_claim,
            ),
        1 => backend
            .inner
            .evaluate_sumcheck_ram_val_check_round(&backend.ram_state, previous_claim),
        _ => Err(BackendError::InvalidRequest {
            backend: backend.inner.name(),
            task: "stage4 universal sumcheck round evaluation",
            reason: format!("instance index {instance_index} is outside the Stage 4 batch"),
        }),
    }
}

fn map_backend_error(error: BackendError) -> UniversalBackendError {
    let _ = error;
    UniversalBackendError::UnsupportedRelation {
        label: "cpu_stage4",
    }
}
