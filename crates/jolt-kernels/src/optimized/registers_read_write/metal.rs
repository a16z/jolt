//! Sparse cycle arithmetic. The CPU kernel owns relation progress and claims.

use std::sync::Arc;

use jolt_akita::AkitaField as F;
use jolt_field::Zero;
use jolt_metal::runtime::{
    host_name, Batch, Binding, Device, DeviceBuffer, Grid, LibrarySpec, ShaderLibrary,
};
use jolt_metal::shaders::FIELD_HEADERS;
use jolt_metal::MetalError;

use super::sparse::layout::SparseEntry;
use super::sparse::{CoeffLut, SeedTransfer};
use super::*;

mod state;
use state::DeviceCycle;

const MIN_GPU_ROWS: usize = 1 << 16;
const CPU_TAIL_ENTRIES: usize = 1 << 16;
const THREADS: usize = 256;
const PAIRS_PER_GROUP: usize = THREADS * 4;

pub(crate) struct MetalRegistersReadWrite {
    gpu: Arc<RegisterGpu>,
    min_rows: usize,
    tail_entries: usize,
}

impl MetalRegistersReadWrite {
    pub(crate) fn new() -> Result<Self, MetalError> {
        let device = Device::system_default()?;
        let mut spec = LibrarySpec::new();
        for (name, source) in FIELD_HEADERS {
            spec = spec.source(name, source);
        }
        spec = spec.source("registers_sparse.metal", include_str!("metal/sparse.metal"));
        for name in [
            "rw_seed_message",
            "rw_seed_bind",
            "rw_sparse_message",
            "rw_sparse_bind",
        ] {
            spec = spec.instantiate::<F>(name);
        }
        let library = ShaderLibrary::compile(&device, &spec)?;
        Ok(Self {
            gpu: Arc::new(RegisterGpu { device, library }),
            min_rows: MIN_GPU_ROWS,
            tail_entries: CPU_TAIL_ENTRIES,
        })
    }
}

struct RegisterGpu {
    device: Device,
    library: ShaderLibrary,
}

impl PrepareKernel<F, RegistersReadWriteChecking<F>> for MetalRegistersReadWrite {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, RegistersReadWriteChecking<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RegistersReadWriteChecking<F>>>, KernelError<F>>
    {
        let mut cpu = match OptimizedRegistersReadWrite::prepare_state(session, witness, inputs)? {
            PreparedReadWrite::Cycle(cpu) => cpu,
            PreparedReadWrite::Address(cpu) => return Ok(Box::new(cpu)),
        };
        let (rows, entries) = cpu
            .cycle
            .as_ref()
            .and_then(CycleState::seed_shape)
            .unwrap_or((0, 0));
        if rows < self.min_rows
            || entries < self.min_rows
            || cpu.dimensions.log_t() < 5
            || (1usize << cpu.dimensions.log_k()) > u128::BITS as usize
        {
            return Ok(Box::new(cpu));
        }
        let cycle = cpu.cycle.take().ok_or(KernelError::InvariantViolation {
            reason: "register cycle owner missing during preparation",
        })?;
        let device = DeviceCycle::new(&self.gpu, cycle).map_err(compute_error)?;
        Ok(Box::new(MetalRegisterKernel {
            cpu,
            device: Some(device),
            gpu: Arc::clone(&self.gpu),
            tail_entries: self.tail_entries,
        }))
    }
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct MetalRegisterKernel {
    cpu: ReadWriteKernel<F>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    device: Option<DeviceCycle>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    gpu: Arc<RegisterGpu>,
    tail_entries: usize,
}

impl ProveRounds<F> for MetalRegisterKernel {
    fn num_rounds(&self) -> usize {
        self.cpu.num_rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(r) = bind {
            if let Some(device) = self.device.take() {
                let device = device.bind(&self.gpu, r).map_err(compute_error)?;
                self.device = Some(device);
                let device = self
                    .device
                    .as_ref()
                    .ok_or_else(|| compute_error(invalid("device owner disappeared")))?;
                if device.ready_for_cpu(self.tail_entries)
                    || self.cpu.challenges.bound() + 1 == self.cpu.dimensions.log_t()
                {
                    let device = self
                        .device
                        .take()
                        .ok_or_else(|| compute_error(invalid("device owner disappeared")))?;
                    self.cpu.cycle = Some(device.restore().map_err(compute_error)?);
                }
                self.cpu.advance_cycle(r)?;
            } else {
                self.cpu.bind(r)?;
            }
        }
        if let Some(device) = self.device.as_mut() {
            let need_one = self.cpu.gruen.current_linear_evals().1.is_zero()
                && !self.cpu.gruen.current_scalar().is_zero();
            let [q0, leading, q1] = device
                .message(
                    &self.gpu,
                    self.cpu.gruen.e_in_current(),
                    self.cpu.gruen.e_out_current(),
                    need_one,
                )
                .map_err(compute_error)?;
            self.cpu
                .gruen
                .checked_cubic(q0, leading, claim, round, || q1)
        } else {
            self.cpu.prove_round(None, round, claim)
        }
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.cpu.finish_rounds(bind)
    }
}

impl SumcheckKernel<F> for MetalRegisterKernel {
    type Relation = RegistersReadWriteChecking<F>;
    fn output_claims(
        &mut self,
        inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<RegistersReadWriteOutputClaims<F>, SumcheckKernelError<F>> {
        self.cpu.output_claims(inputs)
    }
    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<F, Self::Relation>,
        output_points: &SumcheckOutputPoints<F, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<F, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<F>> {
        self.cpu
            .validate_derived_tables(relation, input_points, output_points, challenges)
    }
}

fn invalid(reason: &str) -> MetalError {
    MetalError::InvalidDispatch {
        pipeline: "registers_sparse".into(),
        reason: reason.into(),
    }
}
fn u32_shape(value: usize) -> Result<u32, MetalError> {
    u32::try_from(value).map_err(|_| invalid("sparse extent exceeds 32-bit indexing"))
}
fn compute_error(error: MetalError) -> SumcheckError<F> {
    SumcheckError::Compute {
        backend: "Metal registers read/write",
        reason: error.to_string(),
    }
}

#[cfg(all(test, target_os = "macos"))]
#[expect(clippy::unwrap_used, reason = "reference parity test")]
mod tests;
