use std::sync::Arc;

use jolt_akita::AkitaField;
use jolt_field::Zero;
use jolt_metal::runtime::{
    host_name, Batch, Binding, Device, DeviceBuffer, Grid, LibrarySpec, ShaderLibrary,
};
use jolt_metal::shaders::FIELD_HEADERS;
use jolt_metal::MetalError;

use super::super::lazy_ra::ChunkIndexSource;
use super::super::metal_ra::{DeviceRa, MetalRaFold, CPU_TAIL_ROWS};
use super::super::metal_rows::ResidentInstructionRows;
use super::*;

const MIN_GPU_PREFIX: usize = 1 << 16;

pub(crate) struct MetalInstructionRa(Arc<RaGpu>);

impl MetalInstructionRa {
    pub(crate) fn new() -> Result<Self, MetalError> {
        let device = Device::system_default()?;
        let mut spec = LibrarySpec::new();
        for (name, source) in FIELD_HEADERS {
            spec = spec.source(name, source);
        }
        let spec = spec
            .source("lazy_product.metal", include_str!("lazy_product.metal"))
            .instantiate::<AkitaField>("lazy_ra_product");
        let library = ShaderLibrary::compile(&device, &spec)?;
        Ok(Self(Arc::new(RaGpu {
            device,
            library,
            fold: MetalRaFold::new()?,
        })))
    }
}

impl PrepareKernel<AkitaField, InstructionRaVirtualization<AkitaField>> for MetalInstructionRa {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<AkitaField>,
        inputs: ProverInputs<'_, AkitaField, InstructionRaVirtualization<AkitaField>>,
    ) -> Result<
        Box<dyn SumcheckKernel<AkitaField, Relation = InstructionRaVirtualization<AkitaField>>>,
        KernelError<AkitaField>,
    > {
        let cpu = OptimizedInstructionRaVirtualizationKernel::prepare(session, witness, inputs)?;
        let indices = match cpu.folded_ra.as_ref() {
            Some(LazyFoldedRa::Lazy { source, .. }) => {
                Some(ResidentInstructionRows::shared(session, source.rows())?)
            }
            _ => None,
        };
        Ok(Box::new(MetalInstructionRaKernel {
            cpu,
            indices,
            device: None,
            tail_rows: CPU_TAIL_ROWS,
            gpu: Arc::clone(&self.0),
        }))
    }
}

struct RaGpu {
    device: Device,
    library: ShaderLibrary,
    fold: MetalRaFold,
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct MetalInstructionRaKernel {
    cpu: OptimizedInstructionRaVirtualizationKernel<AkitaField>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    indices: Option<Arc<ResidentInstructionRows>>,
    device: Option<DeviceRa>,
    tail_rows: usize,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    gpu: Arc<RaGpu>,
}

impl MetalInstructionRaKernel {
    fn gpu_message(
        &mut self,
        round: usize,
        previous_claim: AkitaField,
    ) -> Result<Option<UnivariatePoly<AkitaField>>, SumcheckError<AkitaField>> {
        let invalid = |reason: &str| MetalError::InvalidDispatch {
            pipeline: "lazy_ra_product".into(),
            reason: reason.into(),
        };
        let (tables, width, bits, prefix, tails, columns) = if let Some(device) = &self.device {
            (
                &[][..],
                0,
                0,
                device.prefix,
                device.tails.as_slice(),
                device.tails.len(),
            )
        } else {
            let Some(LazyFoldedRa::Lazy {
                tables,
                width,
                factors: 1,
                prefix_len,
                source,
                tail,
                ..
            }) = self.cpu.folded_ra.as_ref()
            else {
                self.indices = None;
                return Ok(None);
            };
            let radix = tables.first().map_or(0, |table| table.len() / width);
            if self.cpu.num_committed_per_virtual != 4
                || !matches!(*width, 1 | 2 | 4 | 8)
                || !matches!(radix, 16 | 256)
                || tables.len() * radix.ilog2() as usize > 128
            {
                return Ok(None);
            }
            if tables.len() != source.num_columns()
                || tables.iter().any(|table| table.len() != width * radix)
            {
                return Err(compute_error(invalid(
                    "branch tables disagree with source geometry",
                )));
            }
            (
                tables.as_slice(),
                *width,
                radix.ilog2() as usize,
                prefix_len.div_ceil(*width),
                tail.as_slice(),
                tables.len(),
            )
        };
        let resident = if self.device.is_none() {
            Some(
                self.indices
                    .as_ref()
                    .ok_or_else(|| compute_error(invalid("missing source owner")))?
                    .resident(&self.gpu.device)
                    .map_err(compute_error)?,
            )
        } else {
            None
        };
        let uploaded = resident.as_ref().and_then(|guard| guard.as_ref());
        let source_prefix = uploaded.map_or(prefix, |rows| rows.prefix.len());
        let pairs = prefix.div_ceil(2);
        let eq = &self.cpu.gruen;
        let in_len = eq.e_in_current_len();
        if in_len == 0 || pairs.div_ceil(in_len) > eq.e_out_current_len() {
            return Err(compute_error(invalid(
                "equality factors do not cover the prefix",
            )));
        }
        let need_zero = eq.current_linear_evals().0.is_zero() && !eq.current_scalar().is_zero();
        let mut totals = [AkitaField::zero(); 5];
        if pairs != 0 {
            let span = tracing::info_span!(
                "metal_instruction_ra_dispatch",
                width,
                gpu_s = tracing::field::Empty
            );
            let _span = span.enter();
            let dimensions = [
                source_prefix,
                pairs,
                width,
                columns,
                bits,
                self.cpu.active_virtuals,
                in_len,
                usize::from(need_zero),
                usize::from(self.device.is_some()),
            ];
            let mut parameters = [0u32; 9];
            for (out, value) in parameters.iter_mut().zip(dimensions) {
                *out = u32::try_from(value)
                    .map_err(|_| compute_error(invalid("shape exceeds 32-bit indexing")))?;
            }
            let flat: Vec<_> = tables.iter().flatten().copied().collect();
            let placeholder = [AkitaField::zero()];
            let tables = DeviceBuffer::from_slice(
                &self.gpu.device,
                if flat.is_empty() { &placeholder } else { &flat },
            )
            .map_err(compute_error)?;
            let empty_rows =
                DeviceBuffer::from_slice(&self.gpu.device, &[[0u64; 4]]).map_err(compute_error)?;
            let device_tails =
                DeviceBuffer::from_slice(&self.gpu.device, tails).map_err(compute_error)?;
            let (indices, packed_tail) =
                uploaded.map_or((&empty_rows, [0; 4]), |rows| (&rows.prefix, rows.tail));
            let dense = self.device.as_ref().map_or(&tables, |state| &state.values);
            let e_in = DeviceBuffer::from_slice(&self.gpu.device, eq.e_in_current())
                .map_err(compute_error)?;
            let e_out = DeviceBuffer::from_slice(&self.gpu.device, eq.e_out_current())
                .map_err(compute_error)?;
            let groups = pairs.div_ceil(in_len);
            let partial_len = groups
                .checked_mul(5)
                .filter(|len| u32::try_from(*len).is_ok())
                .ok_or_else(|| compute_error(invalid("partial output exceeds 32-bit indexing")))?;
            let mut partials = DeviceBuffer::<AkitaField>::zeroed(&self.gpu.device, partial_len)
                .map_err(compute_error)?;
            let pipeline = self
                .gpu
                .library
                .pipeline(&host_name::<AkitaField>("lazy_ra_product"))
                .map_err(compute_error)?;
            let simd_width = pipeline.thread_execution_width();
            if simd_width == 0 {
                return Err(compute_error(invalid("zero SIMD-group width")));
            }
            let group_size =
                pipeline.max_total_threads_per_threadgroup().min(128) / simd_width * simd_width;
            if group_size == 0
                || group_size / simd_width > 32
                || group_size / simd_width > simd_width
            {
                return Err(compute_error(invalid("unsupported SIMD-group geometry")));
            }
            let mut batch = Batch::new(&self.gpu.device).map_err(compute_error)?;
            // SAFETY: source loads select the tail beyond the explicit prefix;
            // tables and equality factors cover the checked shape. Each group
            // owns five output cells and all its lanes reach every barrier.
            unsafe {
                batch
                    .dispatch_unchecked(
                        pipeline,
                        &[
                            Binding::buffer(indices),
                            Binding::buffer(&tables),
                            Binding::buffer(&e_in),
                            Binding::buffer(&e_out),
                            Binding::buffer(&partials),
                            Binding::value(&parameters),
                            Binding::value(&packed_tail),
                            Binding::buffer(dense),
                            Binding::buffer(&device_tails),
                        ],
                        Grid::linear(groups * group_size, group_size),
                    )
                    .map_err(compute_error)?;
            }
            let gpu_time = batch.commit_and_wait().map_err(compute_error)?;
            let _ = span.record("gpu_s", gpu_time.as_secs_f64());
            for partial in partials.read().map_err(compute_error)?.chunks_exact(5) {
                for (total, value) in totals.iter_mut().zip(partial) {
                    *total += *value;
                }
            }
        }
        let suffix = eq.suffix_weight(pairs)
            * tails
                .chunks_exact(4)
                .take(self.cpu.active_virtuals)
                .map(|group| group.iter().copied().product::<AkitaField>())
                .sum::<AkitaField>();
        for value in totals.iter_mut().take(3) {
            *value += suffix;
        }
        totals[4] += suffix;
        eq.checked_toom(&totals[..4], previous_claim, round, || totals[4])
            .map(Some)
    }
}

impl MetalInstructionRaKernel {
    fn restore_cpu(&mut self) -> Result<(), SumcheckError<AkitaField>> {
        if let Some(device) = self.device.take() {
            self.cpu.folded_ra = Some(device.restore().map_err(compute_error)?);
        }
        Ok(())
    }

    fn try_materialize(
        &mut self,
        challenge: AkitaField,
        minimum: usize,
    ) -> Result<bool, SumcheckError<AkitaField>> {
        let Some(folded) = self.cpu.folded_ra.as_ref() else {
            return Ok(false);
        };
        let Some(prefix) = folded.next_materialized_len() else {
            return Ok(false);
        };
        let LazyFoldedRa::Lazy {
            tables,
            width,
            factors: 1,
            source,
            prefix_len,
            tail,
            ..
        } = folded
        else {
            return Ok(false);
        };
        let radix = tables.first().map_or(0, |table| table.len() / width);
        if *prefix_len < minimum
            || self.cpu.num_committed_per_virtual != 4
            || !matches!(radix, 16 | 256)
            || source.num_columns() != tables.len()
        {
            return Ok(false);
        }
        let bits = radix.ilog2() as usize;
        if tables.len() * bits > 128 {
            return Ok(false);
        }
        let selectors: Vec<_> = (0..tables.len())
            .map(|column| {
                let shift = (tables.len() - 1 - column) * bits;
                [u32::from(shift >= 64), (shift % 64) as u32]
            })
            .collect();
        let owner = self.indices.as_ref().ok_or_else(|| {
            compute_error(MetalError::InvalidDispatch {
                pipeline: "resident RA".into(),
                reason: "missing compact rows".into(),
            })
        })?;
        let resident = owner.resident(&self.gpu.device).map_err(compute_error)?;
        let rows = resident.as_ref().ok_or_else(|| {
            compute_error(MetalError::InvalidDispatch {
                pipeline: "resident RA".into(),
                reason: "missing uploaded rows".into(),
            })
        })?;
        let device = self
            .gpu
            .fold
            .materialize(rows, tables, *width, &selectors, prefix, tail, challenge)
            .map_err(compute_error)?;
        drop(resident);
        crate::mem::drop_in_background_thread(self.cpu.folded_ra.take());
        self.device = Some(device);
        self.indices = None;
        self.cpu.advance_bind(challenge);
        Ok(true)
    }

    fn round_with_cutoff(
        &mut self,
        bind: Option<AkitaField>,
        round: usize,
        claim: AkitaField,
        minimum: usize,
    ) -> Result<UnivariatePoly<AkitaField>, SumcheckError<AkitaField>> {
        if self
            .device
            .as_ref()
            .is_some_and(|state| state.prefix <= self.tail_rows)
        {
            self.restore_cpu()?;
        }
        if let Some(challenge) = bind {
            if let Some(device) = &self.device {
                self.device = Some(
                    self.gpu
                        .fold
                        .bind(device, challenge)
                        .map_err(compute_error)?,
                );
                self.cpu.advance_bind(challenge);
            } else if !self.try_materialize(challenge, minimum)? {
                self.cpu.bind(challenge)?;
            }
        }
        let admitted = self.device.is_some()
            || matches!(self.cpu.folded_ra.as_ref(), Some(LazyFoldedRa::Lazy { prefix_len, .. }) if *prefix_len >= minimum);
        if admitted {
            if let Some(message) = self.gpu_message(round, claim)? {
                return Ok(message);
            }
        } else {
            self.indices = None;
        }
        self.cpu.message(round, claim)
    }
}

impl ProveRounds<AkitaField> for MetalInstructionRaKernel {
    fn num_rounds(&self) -> usize {
        self.cpu.num_rounds()
    }
    fn prove_round(
        &mut self,
        bind: Option<AkitaField>,
        round: usize,
        claim: AkitaField,
    ) -> Result<UnivariatePoly<AkitaField>, SumcheckError<AkitaField>> {
        self.round_with_cutoff(bind, round, claim, MIN_GPU_PREFIX)
    }
    fn finish_rounds(&mut self, bind: AkitaField) -> Result<(), SumcheckError<AkitaField>> {
        self.restore_cpu()?;
        self.indices = None;
        self.cpu.finish_rounds(bind)
    }
}

impl SumcheckKernel<AkitaField> for MetalInstructionRaKernel {
    type Relation = InstructionRaVirtualization<AkitaField>;
    fn output_claims(
        &mut self,
        inputs: &SumcheckInputClaims<AkitaField, Self::Relation>,
    ) -> Result<InstructionRaVirtualizationOutputClaims<AkitaField>, SumcheckKernelError<AkitaField>>
    {
        self.cpu.output_claims(inputs)
    }
    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<AkitaField, Self::Relation>,
        output_points: &SumcheckOutputPoints<AkitaField, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<AkitaField, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<AkitaField>> {
        self.cpu
            .validate_derived_tables(relation, input_points, output_points, challenges)
    }
}

fn compute_error(error: MetalError) -> SumcheckError<AkitaField> {
    SumcheckError::Compute {
        backend: "Metal instruction RA",
        reason: error.to_string(),
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests;
