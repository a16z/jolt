//! Sparse cycle arithmetic. The CPU kernel owns relation progress and claims.

use std::{mem::MaybeUninit, sync::Arc};

use jolt_akita::AkitaField as F;
use jolt_field::Zero;
use jolt_metal::runtime::{
    host_name, Batch, Binding, Device, DeviceBuffer, Grid, LibrarySpec, ShaderLibrary,
};
use jolt_metal::shaders::FIELD_HEADERS;
use jolt_metal::MetalError;

use super::sparse::layout::SparseEntry;
use super::sparse::{CoeffLut, SeedEntry};
use super::*;

mod message;
mod state;
use message::RoundWeights;
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
            "rw_sparse_bind_message_cooperative",
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

enum SeedStorage {
    Host(Vec<SeedEntry>),
    Device(DeviceBuffer<SeedEntry>),
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
        let dimensions = inputs.relation.register_dimensions();
        let admit =
            dimensions.log_t() >= 5 && (1usize << dimensions.log_k()) <= u128::BITS as usize;
        let prepared =
            OptimizedRegistersReadWrite::prepare_state(session, witness, inputs, |producer| {
                if !admit || producer.len() == 0 || producer.len() < self.min_rows {
                    return producer.into_vec().map(SeedStorage::Host);
                }
                let allocation = tracing::info_span!("metal_registers_seed_allocate").entered();
                let mut entries = DeviceBuffer::<MaybeUninit<SeedEntry>>::uninit(
                    &self.gpu.device,
                    producer.len(),
                )
                .map_err(compute_error)?;
                drop(allocation);
                let fill = tracing::info_span!("metal_registers_seed_fill").entered();
                producer.write(entries.write_uninit().map_err(compute_error)?)?;
                drop(fill);
                // SAFETY: the canonical producer initialized every entry and
                // its explicit reserved byte before returning success.
                let mut entries = unsafe { entries.assume_init() }.map_err(compute_error)?;
                let rows = entries
                    .read()
                    .map_err(compute_error)?
                    .last()
                    .map_or(0, |entry| entry.row as usize + 1);
                if rows < self.min_rows {
                    return Ok(SeedStorage::Host(
                        entries.read().map_err(compute_error)?.to_vec(),
                    ));
                }
                Ok(SeedStorage::Device(entries))
            })?;
        let (mut cpu, seed) = match prepared {
            PreparedReadWrite::Cycle { kernel, seed } => (kernel, seed),
            PreparedReadWrite::Address(cpu) => return Ok(Box::new(cpu)),
        };
        let SeedTransfer {
            entries,
            ra_lut,
            wa_lut,
            rd_inc,
        } = seed;
        let entries = match entries {
            SeedStorage::Host(entries) => {
                cpu.cycle = Some(
                    SeedTransfer {
                        entries,
                        ra_lut,
                        wa_lut,
                        rd_inc,
                    }
                    .into_cycle(),
                );
                return Ok(cpu);
            }
            SeedStorage::Device(entries) => entries,
        };
        let device = DeviceCycle::new(
            &self.gpu,
            SeedTransfer {
                entries,
                ra_lut,
                wa_lut,
                rd_inc,
            },
        )
        .map_err(compute_error)?;
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
    cpu: Box<ReadWriteKernel<F>>,
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
        let mut fused_message = None;
        if let Some(r) = bind {
            if let Some(device) = self.device.take() {
                let final_cycle = self.cpu.challenges.bound() + 1 == self.cpu.dimensions.log_t();
                if !final_cycle {
                    // Fused messages use the equality state after this bind.
                    self.cpu.advance_cycle(r)?;
                }
                let weights = (!final_cycle).then(|| RoundWeights::new(&self.cpu.gruen));
                let bound = device
                    .bind(&self.gpu, r, weights, self.tail_entries)
                    .map_err(compute_error)?;
                fused_message = bound.message;
                if bound.cycle.ready_for_cpu(self.tail_entries) || final_cycle {
                    self.cpu.cycle = Some(bound.cycle.restore().map_err(compute_error)?);
                } else {
                    self.device = Some(bound.cycle);
                }
                if final_cycle {
                    self.cpu.advance_cycle(r)?;
                }
            } else {
                self.cpu.bind(r)?;
            }
        }
        let message = match (fused_message, self.device.as_mut()) {
            (Some(message), _) => Some(message),
            (None, Some(device)) => Some(
                device
                    .message(&self.gpu, RoundWeights::new(&self.cpu.gruen))
                    .map_err(compute_error)?,
            ),
            (None, None) => None,
        };
        match message {
            Some([q0, leading, q1]) => {
                self.cpu
                    .gruen
                    .checked_cubic(q0, leading, claim, round, || q1)
            }
            None => self.cpu.prove_round(None, round, claim),
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
