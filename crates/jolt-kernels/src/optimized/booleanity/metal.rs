use jolt_akita::AkitaField;
use jolt_field::Zero;
use jolt_metal::runtime::{
    host_name, Batch, Binding, Device, DeviceBuffer, Grid, LibrarySpec, ShaderLibrary,
};
use jolt_metal::shaders::FIELD_HEADERS;
use jolt_metal::MetalError;

use super::super::metal_rows::ResidentInstructionRows;
use super::*;

const MIN_GPU_PREFIX: usize = 1 << 16;

pub(crate) struct MetalBooleanity(Arc<BooleanityGpu>);

impl MetalBooleanity {
    pub(crate) fn new() -> Result<Self, MetalError> {
        let device = Device::system_default()?;
        let mut spec = LibrarySpec::new();
        for (name, source) in FIELD_HEADERS {
            spec = spec.source(name, source);
        }
        let spec = spec
            .source(
                "lazy_booleanity.metal",
                include_str!("lazy_booleanity.metal"),
            )
            .instantiate::<AkitaField>("lazy_booleanity");
        let library = ShaderLibrary::compile(&device, &spec)?;
        Ok(Self(Arc::new(BooleanityGpu { device, library })))
    }
}

impl PrepareKernel<AkitaField, Booleanity<AkitaField>> for MetalBooleanity {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<AkitaField>,
        inputs: ProverInputs<'_, AkitaField, Booleanity<AkitaField>>,
    ) -> Result<
        Box<dyn SumcheckKernel<AkitaField, Relation = Booleanity<AkitaField>>>,
        KernelError<AkitaField>,
    > {
        let cpu = OptimizedBooleanityCycleKernel::prepare(session, witness, inputs)?;
        let rows = match &cpu.tables {
            LazyFoldedRa::Lazy { source, .. } => {
                Some(ResidentInstructionRows::shared(session, &source.rows)?)
            }
            _ => None,
        };
        Ok(Box::new(MetalBooleanityKernel {
            cpu,
            rows,
            gpu: Arc::clone(&self.0),
        }))
    }
}

struct BooleanityGpu {
    device: Device,
    library: ShaderLibrary,
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct MetalBooleanityKernel {
    cpu: OptimizedBooleanityCycleKernel<AkitaField>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    rows: Option<Arc<ResidentInstructionRows>>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    gpu: Arc<BooleanityGpu>,
}

impl MetalBooleanityKernel {
    fn gpu_message(
        &mut self,
        round: usize,
        claim: AkitaField,
    ) -> Result<Option<UnivariatePoly<AkitaField>>, SumcheckError<AkitaField>> {
        let LazyFoldedRa::Lazy {
            tables,
            width,
            factors: 1,
            prefix_len,
            source,
            ..
        } = &self.cpu.tables
        else {
            self.rows = None;
            return Ok(None);
        };
        let radix = tables.first().map_or(0, |table| table.len() / width);
        if !matches!(*width, 4 | 8) || !matches!(radix, 16 | 256) {
            return Ok(None);
        }
        let invalid = |reason: &str| MetalError::InvalidDispatch {
            pipeline: "lazy_booleanity".into(),
            reason: reason.into(),
        };
        if tables.len() != source.num_columns()
            || source.chunk_bits != radix.ilog2() as usize
            || tables.iter().any(|table| table.len() != width * radix)
        {
            return Err(compute_error(invalid(
                "branch tables disagree with the source geometry",
            )));
        }
        let active = self
            .cpu
            .gamma_powers
            .iter()
            .take_while(|rho| !rho.is_zero())
            .count();
        let mut selectors = Vec::with_capacity(active);
        for selector in source.selectors.iter().take(active) {
            let (word, shift) = match selector {
                ColumnSelector::Instruction(selector) => {
                    let shift = selector.shift();
                    if shift >= 128 {
                        return Ok(None);
                    }
                    (u32::from(shift >= 64), shift % 64)
                }
                ColumnSelector::Bytecode(selector) => (2, selector.shift()),
                ColumnSelector::Ram(selector) => (3, selector.shift()),
            };
            if shift + source.chunk_bits > 64 {
                return Ok(None);
            }
            selectors.push([word, shift as u32]);
        }
        let pairs = self.cpu.tables.prefix_len().div_ceil(2);
        let eq = &self.cpu.eq;
        let in_len = eq.e_in_current_len();
        if in_len == 0 || pairs.div_ceil(in_len) > eq.e_out_current_len() {
            return Err(compute_error(invalid(
                "equality factors do not cover the prefix",
            )));
        }
        let need_one = eq.current_linear_evals().1.is_zero() && !eq.current_scalar().is_zero();
        let mut totals = [AkitaField::zero(); 3];
        if pairs != 0 {
            let resident = self
                .rows
                .as_ref()
                .ok_or_else(|| compute_error(invalid("missing source owner")))?
                .resident(&self.gpu.device)
                .map_err(compute_error)?;
            let rows = resident
                .as_ref()
                .ok_or_else(|| compute_error(invalid("missing source")))?;
            if rows.prefix.len() != *prefix_len {
                return Err(compute_error(invalid(
                    "resident source prefix disagrees with lazy state",
                )));
            }
            let span = tracing::info_span!(
                "metal_booleanity_dispatch",
                width,
                gpu_s = tracing::field::Empty
            );
            let _span = span.enter();
            let mut params = [0u32; 7];
            for (out, value) in params.iter_mut().zip([
                *prefix_len,
                pairs,
                *width,
                radix,
                active,
                in_len,
                usize::from(need_one),
            ]) {
                *out = u32::try_from(value)
                    .map_err(|_| compute_error(invalid("shape exceeds 32-bit indexing")))?;
            }
            let flat: Vec<_> = tables.iter().take(active).flatten().copied().collect();
            let tables =
                DeviceBuffer::from_slice(&self.gpu.device, &flat).map_err(compute_error)?;
            let rho = DeviceBuffer::from_slice(&self.gpu.device, &self.cpu.gamma_powers[..active])
                .map_err(compute_error)?;
            let selectors =
                DeviceBuffer::from_slice(&self.gpu.device, &selectors).map_err(compute_error)?;
            let e_in = DeviceBuffer::from_slice(&self.gpu.device, eq.e_in_current())
                .map_err(compute_error)?;
            let e_out = DeviceBuffer::from_slice(&self.gpu.device, eq.e_out_current())
                .map_err(compute_error)?;
            let groups = pairs.div_ceil(in_len);
            let partial_len = groups
                .checked_mul(3)
                .filter(|len| u32::try_from(*len).is_ok())
                .ok_or_else(|| compute_error(invalid("partial output exceeds 32-bit indexing")))?;
            let mut partials = DeviceBuffer::<AkitaField>::zeroed(&self.gpu.device, partial_len)
                .map_err(compute_error)?;
            let pipeline = self
                .gpu
                .library
                .pipeline(&host_name::<AkitaField>("lazy_booleanity"))
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
            // SAFETY: compact-row loads use the tail beyond the checked prefix.
            // Selectors and branch tables cover every active column; each group
            // owns three partials and all its lanes reach the group barriers.
            unsafe {
                batch
                    .dispatch_unchecked(
                        pipeline,
                        &[
                            Binding::buffer(&rows.prefix),
                            Binding::buffer(&tables),
                            Binding::buffer(&rho),
                            Binding::buffer(&selectors),
                            Binding::buffer(&e_in),
                            Binding::buffer(&e_out),
                            Binding::buffer(&partials),
                            Binding::value(&params),
                            Binding::value(&rows.tail),
                        ],
                        Grid::linear(groups * group_size, group_size),
                    )
                    .map_err(compute_error)?;
            }
            let gpu_time = batch.commit_and_wait().map_err(compute_error)?;
            let _ = span.record("gpu_s", gpu_time.as_secs_f64());
            for partial in partials.read().map_err(compute_error)?.chunks_exact(3) {
                for (total, value) in totals.iter_mut().zip(partial) {
                    *total += *value;
                }
            }
        }
        let tail = self
            .cpu
            .tables
            .tail_values()
            .iter()
            .zip(&self.cpu.gamma_powers)
            .take(active)
            .map(|(value, rho)| *value * (*value - *rho))
            .sum::<AkitaField>();
        let suffix = eq.suffix_weight(pairs) * tail;
        eq.checked_cubic(totals[0] + suffix, totals[1], claim, round, || {
            totals[2] + suffix
        })
        .map(Some)
    }
}

impl ProveRounds<AkitaField> for MetalBooleanityKernel {
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
        match &self.cpu.tables {
            LazyFoldedRa::Lazy { prefix_len, .. } if *prefix_len >= MIN_GPU_PREFIX => {
                if let Some(message) = self.gpu_message(round, previous_claim)? {
                    return Ok(message);
                }
            }
            _ => self.rows = None,
        }
        self.cpu.message(round, previous_claim)
    }
    fn finish_rounds(&mut self, bind: AkitaField) -> Result<(), SumcheckError<AkitaField>> {
        self.rows = None;
        self.cpu.finish_rounds(bind)
    }
}

impl SumcheckKernel<AkitaField> for MetalBooleanityKernel {
    type Relation = Booleanity<AkitaField>;
    fn output_claims(
        &mut self,
        inputs: &SumcheckInputClaims<AkitaField, Self::Relation>,
    ) -> Result<SumcheckOutputClaims<AkitaField, Self::Relation>, SumcheckKernelError<AkitaField>>
    {
        self.cpu.output_claims(inputs)
    }
    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<AkitaField, Self::Relation>,
        output_points: &SumcheckOutputPoints<AkitaField, Self::Relation>,
        challenges: &BooleanityCyclePhaseChallenges<AkitaField>,
    ) -> Result<(), SumcheckKernelError<AkitaField>> {
        self.cpu
            .validate_derived_tables(relation, input_points, output_points, challenges)
    }
}

fn compute_error(error: MetalError) -> SumcheckError<AkitaField> {
    SumcheckError::Compute {
        backend: "Metal booleanity",
        reason: error.to_string(),
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests;
