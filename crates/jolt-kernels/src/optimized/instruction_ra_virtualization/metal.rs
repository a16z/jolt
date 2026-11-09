use std::sync::Arc;

use jolt_akita::AkitaField;
use jolt_field::Zero;
use jolt_metal::runtime::{
    host_name, Batch, Binding, Device, DeviceBuffer, Grid, LibrarySpec, ShaderLibrary,
};
use jolt_metal::shaders::FIELD_HEADERS;
use jolt_metal::MetalError;

use super::super::lazy_ra::ChunkIndexSource;
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
        Ok(Self(Arc::new(RaGpu { device, library })))
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
        Ok(Box::new(MetalInstructionRaKernel {
            cpu: OptimizedInstructionRaVirtualizationKernel::prepare(session, witness, inputs)?,
            indices: None,
            gpu: Arc::clone(&self.0),
        }))
    }
}

struct RaGpu {
    device: Device,
    library: ShaderLibrary,
}

struct ResidentIndices {
    prefix: DeviceBuffer<[u64; 2]>,
    tail: [u64; 2],
}

impl ResidentIndices {
    fn upload(gpu: &RaGpu, source: &LookupIndexChunks, prefix: usize) -> Result<Self, MetalError> {
        let _span = tracing::info_span!("metal_instruction_ra_upload").entered();
        let values = map_indices(prefix, |row| {
            let index = source.packed_index(row);
            [index as u64, (index >> 64) as u64]
        });
        let tail = source.packed_index(source.cycles() - 1);
        Ok(Self {
            prefix: DeviceBuffer::from_slice(&gpu.device, &values)?,
            tail: [tail as u64, (tail >> 64) as u64],
        })
    }
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct MetalInstructionRaKernel {
    cpu: OptimizedInstructionRaVirtualizationKernel<AkitaField>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    indices: Option<ResidentIndices>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    gpu: Arc<RaGpu>,
}

impl MetalInstructionRaKernel {
    fn gpu_message(
        &mut self,
        round: usize,
        previous_claim: AkitaField,
    ) -> Result<Option<UnivariatePoly<AkitaField>>, SumcheckError<AkitaField>> {
        let LazyFoldedRa::Lazy {
            tables,
            width,
            factors: 1,
            prefix_len,
            source,
            ..
        } = &self.cpu.folded_ra
        else {
            self.indices = None;
            return Ok(None);
        };
        let radix = tables.first().map_or(0, |table| table.len() / width);
        if self.cpu.num_committed_per_virtual != 4
            || !matches!(*width, 2 | 4 | 8)
            || !matches!(radix, 16 | 256)
            || tables.len() * radix.ilog2() as usize > 128
        {
            return Ok(None);
        }
        let invalid = |reason: &str| MetalError::InvalidDispatch {
            pipeline: "lazy_ra_product".into(),
            reason: reason.into(),
        };
        if tables.len() != source.num_columns()
            || tables.iter().any(|table| table.len() != width * radix)
        {
            return Err(compute_error(invalid(
                "branch tables disagree with the source geometry",
            )));
        }
        if self.indices.is_none() {
            self.indices = Some(
                ResidentIndices::upload(&self.gpu, source, *prefix_len).map_err(compute_error)?,
            );
        }
        let indices = self
            .indices
            .as_ref()
            .ok_or_else(|| compute_error(invalid("missing source")))?;
        let pairs = self.cpu.folded_ra.prefix_len().div_ceil(2);
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
            let _span = tracing::info_span!("metal_instruction_ra_dispatch", width).entered();
            let dimensions = [
                *prefix_len,
                pairs,
                *width,
                tables.len(),
                radix.ilog2() as usize,
                self.cpu.active_virtuals,
                in_len,
                usize::from(need_zero),
            ];
            let mut parameters = [0u32; 8];
            for (out, value) in parameters.iter_mut().zip(dimensions) {
                *out = u32::try_from(value)
                    .map_err(|_| compute_error(invalid("shape exceeds 32-bit indexing")))?;
            }
            let flat: Vec<_> = tables.iter().flatten().copied().collect();
            let tables =
                DeviceBuffer::from_slice(&self.gpu.device, &flat).map_err(compute_error)?;
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
                            Binding::buffer(&indices.prefix),
                            Binding::buffer(&tables),
                            Binding::buffer(&e_in),
                            Binding::buffer(&e_out),
                            Binding::buffer(&partials),
                            Binding::value(&parameters),
                            Binding::value(&indices.tail),
                        ],
                        Grid::linear(groups * group_size, group_size),
                    )
                    .map_err(compute_error)?;
            }
            let _ = batch.commit_and_wait().map_err(compute_error)?;
            for partial in partials.read().map_err(compute_error)?.chunks_exact(5) {
                for (total, value) in totals.iter_mut().zip(partial) {
                    *total += *value;
                }
            }
        }
        let suffix = eq.suffix_weight(pairs)
            * self
                .cpu
                .folded_ra
                .tail_values()
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

impl ProveRounds<AkitaField> for MetalInstructionRaKernel {
    fn num_rounds(&self) -> usize {
        self.cpu.num_rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<AkitaField>,
        round: usize,
        previous_claim: AkitaField,
    ) -> Result<UnivariatePoly<AkitaField>, SumcheckError<AkitaField>> {
        if let Some(challenge) = bind {
            self.cpu.bind(challenge);
        }
        let admitted = matches!(&self.cpu.folded_ra, LazyFoldedRa::Lazy { prefix_len, .. } if *prefix_len >= MIN_GPU_PREFIX);
        if admitted {
            if let Some(message) = self.gpu_message(round, previous_claim)? {
                return Ok(message);
            }
        } else {
            self.indices = None;
        }
        self.cpu.message(round, previous_claim)
    }

    fn finish_rounds(&mut self, bind: AkitaField) -> Result<(), SumcheckError<AkitaField>> {
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
