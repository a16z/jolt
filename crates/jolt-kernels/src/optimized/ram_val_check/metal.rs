//! Donor sparse-parent traversal over increments captured by the stage-2 producer.
use std::mem::MaybeUninit;
use std::sync::Arc;

use bytemuck_derive::{Pod, Zeroable};
use jolt_akita::AkitaField as F;
use jolt_field::Zero;
use jolt_metal::runtime::{
    host_name, Batch, Binding, Device, DeviceBuffer, Grid, LibrarySpec, ShaderLibrary,
};
use jolt_metal::shaders::FIELD_HEADERS;
use jolt_metal::MetalError;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::*;
mod source;
pub(crate) use source::CaptureRamReadWrite;
use source::{branch_weights, SparseSource};

const KERNEL: &str = "ram_val_sparse_round";
const THREADS: usize = 256;
const MIN_PREFIX: usize = 1 << 16;

struct Gpu {
    device: Device,
    library: ShaderLibrary,
}
pub(crate) struct MetalRamVal {
    gpu: Arc<Gpu>,
    min_prefix: usize,
}
impl MetalRamVal {
    pub(crate) fn new() -> Result<Self, MetalError> {
        let device = Device::system_default()?;
        let mut spec = LibrarySpec::new();
        for (name, source) in FIELD_HEADERS {
            spec = spec.source(name, source);
        }
        spec = spec
            .source("ram_val.metal", include_str!("value.metal"))
            .instantiate::<F>(KERNEL);
        let library = ShaderLibrary::compile(&device, &spec)?;
        let pipeline = library.pipeline(&host_name::<F>(KERNEL))?;
        if pipeline.thread_execution_width() != 32
            || pipeline.max_total_threads_per_threadgroup() < THREADS
            || pipeline.static_threadgroup_memory_length()
                > device.limits().max_threadgroup_memory_length
        {
            return Err(invalid("RAM value reduction exceeds device limits"));
        }
        Ok(Self {
            gpu: Arc::new(Gpu { device, library }),
            min_prefix: MIN_PREFIX,
        })
    }
    pub(crate) fn preparation(&self) -> CaptureRamReadWrite {
        CaptureRamReadWrite {
            min_prefix: self.min_prefix,
        }
    }
}
impl PrepareKernel<F, RamValCheck<F>> for MetalRamVal {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, RamValCheck<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RamValCheck<F>>>, KernelError<F>> {
        let log_t = inputs.relation.trace_dimensions().log_t();
        let Some(source) = session.take::<SparseSource>() else {
            return OptimizedBackend.prepare(session, witness, inputs);
        };
        if !(6..=31).contains(&log_t) {
            drop(source);
            return OptimizedBackend.prepare(session, witness, inputs);
        }
        if source.cycles != 1usize << log_t
            || source.addresses.len() != source.cycles
            || !witness
                .random_access()
                .is_some_and(|rows| rows.same_source(&source.source))
        {
            return Err(KernelError::InvariantViolation {
                reason: "RAM increment carry belongs to a different source/domain",
            });
        }
        let max = self.gpu.device.limits().max_buffer_length;
        if source.prefix < self.min_prefix
            || source
                .addresses
                .prefix_len()
                .checked_mul(size_of::<u32>())
                .is_none_or(|bytes| bytes > max)
            || source
                .records
                .checked_mul(size_of::<F>())
                .is_none_or(|bytes| bytes > max)
        {
            // Release the candidate before the fallback collects value columns.
            drop(source);
            return OptimizedBackend.prepare(session, witness, inputs);
        }
        let cpu = RamValCheckKernel::prepare_state(
            inputs,
            Arc::clone(&source.addresses),
            IncColumn::Bound(Vec::new()),
        )?;
        let span = tracing::info_span!(
            "metal_ram_val_upload",
            records = source.records,
            address_prefix = source.addresses.prefix_len()
        );
        let _entered = span.enter();
        let uploaded = source.upload(&self.gpu.device)?;
        let addresses = DeviceBuffer::from_slice(&self.gpu.device, source.addresses.prefix())
            .map_err(compute_error)?;
        Ok(Box::new(MetalValueKernel {
            cpu,
            gpu: Arc::clone(&self.gpu),
            resident: Some(Resident {
                source,
                cycles: uploaded.cycles,
                values: uploaded.values,
                addresses,
            }),
            challenges: Vec::new(),
        }))
    }
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct Resident {
    source: SparseSource,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    cycles: DeviceBuffer<u32>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    values: DeviceBuffer<F>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    addresses: DeviceBuffer<u32>,
}
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct MetalValueKernel {
    cpu: RamValCheckKernel<F>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    gpu: Arc<Gpu>,
    resident: Option<Resident>,
    challenges: Vec<F>,
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    records: u32,
    address_prefix: u32,
    width: u32,
    radix: u32,
    lo_length: u32,
}
impl MetalValueKernel {
    fn restore(&mut self) {
        if let Some(resident) = self.resident.take() {
            self.cpu.inc = IncColumn::Bound(
                resident
                    .source
                    .materialize(&branch_weights(&self.challenges)),
            );
        }
    }
    fn bind_lazy(&mut self, challenge: F) {
        self.challenges.push(challenge);
        self.cpu.ra.bind(challenge);
        self.cpu.lt.bind(challenge);
        self.cpu.progress.advance();
    }
    fn message(&self) -> Result<[F; 3], SumcheckError<F>> {
        let Some(resident) = &self.resident else {
            return Err(compute_error(invalid("RAM source missing before message")));
        };
        let LazyFoldedRa::Lazy {
            tables,
            width,
            factors: 1,
            ..
        } = &self.cpu.ra
        else {
            return Err(compute_error(invalid(
                "RAM value message requires canonical lazy selectors",
            )));
        };
        let SplitLt::Split {
            lt_lo,
            lt_hi,
            eq_hi,
        } = &self.cpu.lt
        else {
            return Err(compute_error(invalid(
                "RAM value message requires split LT",
            )));
        };
        if !matches!(width, 1 | 2 | 4 | 8) || tables.len() != 1 || lt_lo.len() < 2 {
            return Err(compute_error(invalid(
                "RAM value lazy geometry is unsupported",
            )));
        }
        let params = Params {
            records: word(resident.source.records)?,
            address_prefix: word(resident.addresses.len())?,
            width: word(*width)?,
            radix: word(tables[0].len() / width)?,
            lo_length: word(lt_lo.len())?,
        };
        let device = &self.gpu.device;
        let table = DeviceBuffer::from_slice(device, &tables[0]).map_err(compute_error)?;
        let weights = DeviceBuffer::from_slice(device, &branch_weights(&self.challenges))
            .map_err(compute_error)?;
        let lo = DeviceBuffer::from_slice(device, lt_lo).map_err(compute_error)?;
        let hi = DeviceBuffer::from_slice(device, lt_hi).map_err(compute_error)?;
        let eq = DeviceBuffer::from_slice(device, eq_hi).map_err(compute_error)?;
        let groups = resident.source.records.div_ceil(THREADS);
        let partials =
            DeviceBuffer::<MaybeUninit<F>>::uninit(device, groups * 3).map_err(compute_error)?;
        let span = tracing::info_span!(
            "metal_ram_val_dispatch",
            width,
            records = resident.source.records,
            gpu_s = tracing::field::Empty
        );
        let _entered = span.enter();
        let mut batch = Batch::new(device).map_err(compute_error)?;
        // SAFETY: one leader per active parent gathers only validated source
        // indices. All256 lanes participate in reductions; every group writes
        // exactly three canonical field values. Sources live through completion.
        unsafe {
            batch
                .dispatch_unchecked(
                    self.gpu
                        .library
                        .pipeline(&host_name::<F>(KERNEL))
                        .map_err(compute_error)?,
                    &[
                        Binding::buffer(&resident.cycles),
                        Binding::buffer(&resident.values),
                        Binding::buffer(&resident.addresses),
                        Binding::buffer(&table),
                        Binding::buffer(&weights),
                        Binding::buffer(&lo),
                        Binding::buffer(&hi),
                        Binding::buffer(&eq),
                        Binding::buffer(&partials),
                        Binding::value(&params),
                    ],
                    Grid::linear(groups * THREADS, THREADS),
                )
                .map_err(compute_error)?;
        }
        let elapsed = batch.commit_and_wait().map_err(compute_error)?;
        let _ = span.record("gpu_s", elapsed.as_secs_f64());
        // SAFETY: successful completion wrote every partial, including empty lanes.
        let mut partials = unsafe { partials.assume_init() }.map_err(compute_error)?;
        let mut totals = [F::zero(); 3];
        for partial in partials.read().map_err(compute_error)?.chunks_exact(3) {
            for (total, value) in totals.iter_mut().zip(partial) {
                *total += *value;
            }
        }
        Ok(totals)
    }
}
impl ProveRounds<F> for MetalValueKernel {
    fn num_rounds(&self) -> usize {
        self.cpu.num_rounds()
    }
    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if self.resident.is_none() {
            return self.cpu.prove_round(bind, round, claim);
        }
        if round != self.cpu.progress.bound() + usize::from(bind.is_some())
            || (round == 0) != bind.is_none()
        {
            return Err(compute_error(invalid(
                "RAM value round progression mismatch",
            )));
        }
        if let Some(challenge) = bind {
            let handoff = self.cpu.ra.next_materialized_len().is_some()
                || !matches!(&self.cpu.lt, SplitLt::Split { lt_lo, .. } if lt_lo.len() > 2);
            if handoff {
                let _entered = tracing::info_span!(
                    "metal_ram_val_handoff",
                    width = 2usize << self.challenges.len()
                )
                .entered();
                self.bind_lazy(challenge);
                self.restore();
                return self.cpu.prove_round(None, round, claim);
            }
            self.bind_lazy(challenge);
        }
        Ok(UnivariatePoly::from_evals_and_hint(claim, &self.message()?))
    }
    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.restore();
        self.cpu.finish_rounds(bind)
    }
}
impl SumcheckKernel<F> for MetalValueKernel {
    type Relation = RamValCheck<F>;
    fn output_claims(
        &mut self,
        inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<RamValCheckOutputClaims<F>, SumcheckKernelError<F>> {
        self.cpu.output_claims(inputs)
    }
    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        inputs: &SumcheckInputPoints<F, Self::Relation>,
        outputs: &SumcheckOutputPoints<F, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<F, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<F>> {
        self.cpu
            .validate_derived_tables(relation, inputs, outputs, challenges)
    }
}
fn invalid(reason: &str) -> MetalError {
    MetalError::InvalidDispatch {
        pipeline: KERNEL.into(),
        reason: reason.into(),
    }
}
fn compute_error(error: MetalError) -> SumcheckError<F> {
    SumcheckError::Compute {
        backend: "Metal RAM value",
        reason: error.to_string(),
    }
}
fn word(value: usize) -> Result<u32, SumcheckError<F>> {
    u32::try_from(value).map_err(|_| compute_error(invalid("RAM value extent exceeds u32")))
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "reference oracle fails immediately")]
mod tests;
