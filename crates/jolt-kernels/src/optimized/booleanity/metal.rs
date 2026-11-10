use std::iter::once;

use jolt_akita::AkitaField;
use jolt_field::Zero;
use jolt_metal::runtime::{
    host_name, Batch, Binding, Device, DeviceBuffer, Grid, LibrarySpec, ShaderLibrary,
};
use jolt_metal::shaders::FIELD_HEADERS;
use jolt_metal::MetalError;

use super::super::metal_ra::{DeviceRa, MetalRaFold, CPU_TAIL_ROWS};
use super::super::metal_rows::ResidentInstructionRows;
use super::*;

mod tables;
use tables::CachedTables;

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
            .source(
                "cached_booleanity.metal",
                include_str!("cached_booleanity.metal"),
            )
            .instantiate::<AkitaField>("lazy_booleanity")
            .instantiate::<AkitaField>("lazy_booleanity_cached");
        let library = ShaderLibrary::compile(&device, &spec)?;
        Ok(Self(Arc::new(BooleanityGpu {
            device,
            library,
            fold: MetalRaFold::new()?,
        })))
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
        let rows = match cpu.tables.as_ref() {
            Some(LazyFoldedRa::Lazy { source, .. }) => {
                Some(ResidentInstructionRows::shared(session, &source.rows)?)
            }
            _ => None,
        };
        Ok(Box::new(MetalBooleanityKernel {
            cpu,
            rows,
            device: None,
            tail_rows: CPU_TAIL_ROWS,
            gpu: Arc::clone(&self.0),
            #[cfg(test)]
            table_policy_cycles: None,
            #[cfg(test)]
            cached_widths: 0,
        }))
    }
}

struct BooleanityGpu {
    device: Device,
    library: ShaderLibrary,
    fold: MetalRaFold,
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct MetalBooleanityKernel {
    cpu: OptimizedBooleanityCycleKernel<AkitaField>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    rows: Option<Arc<ResidentInstructionRows>>,
    device: Option<DeviceRa>,
    tail_rows: usize,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    gpu: Arc<BooleanityGpu>,
    // The dense oracle uses a small domain but exercises the same bounded
    // cache construction as a large trace, without a new full-trace fixture.
    #[cfg(test)]
    table_policy_cycles: Option<usize>,
    #[cfg(test)]
    cached_widths: u32,
}

impl MetalBooleanityKernel {
    fn gpu_message(
        &mut self,
        round: usize,
        claim: AkitaField,
    ) -> Result<Option<UnivariatePoly<AkitaField>>, SumcheckError<AkitaField>> {
        let invalid = |reason: &str| MetalError::InvalidDispatch {
            pipeline: "lazy_booleanity".into(),
            reason: reason.into(),
        };
        let active = self
            .cpu
            .gamma_powers
            .iter()
            .take_while(|rho| !rho.is_zero())
            .count();
        let (tables, width, radix, prefix, tails, selectors, columns, lazy_source) =
            if let Some(device) = &self.device {
                (
                    &[][..],
                    0,
                    0,
                    device.prefix,
                    device.tails.as_slice(),
                    Vec::new(),
                    device.tails.len(),
                    None,
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
                }) = self.cpu.tables.as_ref()
                else {
                    self.rows = None;
                    return Ok(None);
                };
                let radix = tables.first().map_or(0, |table| table.len() / width);
                if !matches!(*width, 1 | 2 | 4 | 8) || !matches!(radix, 16 | 256) {
                    return Ok(None);
                }
                if tables.len() != source.num_columns()
                    || source.chunk_bits != radix.ilog2() as usize
                    || tables.iter().any(|table| table.len() != width * radix)
                {
                    return Err(compute_error(invalid(
                        "branch tables disagree with source geometry",
                    )));
                }
                let Some(selectors) = Self::selectors(source) else {
                    return Ok(None);
                };
                (
                    tables.as_slice(),
                    *width,
                    radix,
                    prefix_len.div_ceil(*width),
                    tail.as_slice(),
                    selectors,
                    tables.len(),
                    Some(source),
                )
            };
        let resident = if self.device.is_none() {
            Some(
                self.rows
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
            let span = tracing::info_span!(
                "metal_booleanity_dispatch",
                width,
                initial_constant_lut = width == 1,
                table_mode = tracing::field::Empty,
                table_device_bytes = tracing::field::Empty,
                table_peak_bytes = tracing::field::Empty,
                gpu_s = tracing::field::Empty
            );
            let _span = span.enter();
            let mut params = [0u32; 9];
            for (out, value) in params.iter_mut().zip([
                source_prefix,
                pairs,
                width,
                radix,
                active,
                in_len,
                usize::from(need_one),
                usize::from(self.device.is_some()),
                columns,
            ]) {
                *out = u32::try_from(value)
                    .map_err(|_| compute_error(invalid("shape exceeds 32-bit indexing")))?;
            }
            let groups = pairs.div_ceil(in_len);
            let partial_len = groups
                .checked_mul(3)
                .filter(|len| u32::try_from(*len).is_ok())
                .ok_or_else(|| compute_error(invalid("partial output exceeds 32-bit indexing")))?;
            // Count the other host/device round buffers with checked arithmetic.
            // Existing retained trace rows are shared, not a new cache cost.
            let auxiliary_bytes = [
                eq.e_in_current().len(),
                eq.e_out_current().len(),
                partial_len,
                active,
                tails.len(),
                selectors.len(),
                columns,
            ]
            .into_iter()
            .try_fold(0usize, |sum, len| sum.checked_add(len))
            .and_then(|len| len.checked_mul(2 * std::mem::size_of::<AkitaField>()))
            .and_then(|bytes| bytes.checked_add(4096))
            .ok_or_else(|| compute_error(invalid("round buffer budget overflows")))?;
            let cached_pipeline = self
                .gpu
                .library
                .pipeline(&host_name::<AkitaField>("lazy_booleanity_cached"))
                .map_err(compute_error)?;
            let cached_width = cached_pipeline.thread_execution_width();
            let cached_group = if cached_width == 0 {
                0
            } else {
                cached_pipeline.max_total_threads_per_threadgroup().min(128) / cached_width
                    * cached_width
            };
            let cached_admitted = cached_group != 0
                && cached_group / cached_width <= 32
                && cached_group / cached_width <= cached_width
                && cached_pipeline.static_threadgroup_memory_length()
                    <= self.gpu.device.limits().max_threadgroup_memory_length;
            let cached = if cached_admitted {
                if let Some(source) = lazy_source {
                    #[cfg(not(test))]
                    let policy_cycles = source.cycles();
                    #[cfg(test)]
                    let policy_cycles = self.table_policy_cycles.unwrap_or_else(|| source.cycles());
                    CachedTables::prepare(
                        &self.gpu.device,
                        tables,
                        width,
                        source,
                        &self.cpu.gamma_powers,
                        auxiliary_bytes,
                        policy_cycles,
                    )
                    .map_err(compute_error)?
                } else {
                    None
                }
            } else {
                None
            };
            let mode = cached.as_ref().map_or(0, |tables| tables.mode);
            let _ = span.record("table_mode", mode);
            let _ = span.record(
                "table_device_bytes",
                cached.as_ref().map_or(0, |tables| tables.device_bytes),
            );
            let _ = span.record(
                "table_peak_bytes",
                cached.as_ref().map_or(0, |tables| tables.peak_bytes),
            );
            #[cfg(test)]
            if cached.is_some() {
                self.cached_widths |= 1 << width;
            }
            let flat: Vec<_> = if cached.is_none() {
                tables.iter().take(active).flatten().copied().collect()
            } else {
                Vec::new()
            };
            let placeholder = [AkitaField::zero()];
            // Donor 7aa549cc9 booleanity::write_initial_constant_table uses a
            // radix+1 lookup, including cold RAM, rather than a square cache.
            // Keep the canonical CPU categorical constant formula as owner.
            let initial_constants: Vec<_> = if width == 1 && cached.is_none() {
                let _ = active
                    .checked_mul(radix + 1)
                    .filter(|len| u32::try_from(*len).is_ok())
                    .ok_or_else(|| {
                        compute_error(invalid("initial constant table exceeds 32-bit indexing"))
                    })?;
                tables
                    .iter()
                    .take(active)
                    .zip(&self.cpu.gamma_powers)
                    .flat_map(|(table, rho)| {
                        categorical_constants(
                            table.iter().copied().chain(once(AkitaField::zero())),
                            *rho,
                        )
                    })
                    .collect()
            } else {
                Vec::new()
            };
            let initial_constants = if width == 1 && cached.is_none() {
                Some(
                    DeviceBuffer::from_slice(&self.gpu.device, &initial_constants)
                        .map_err(compute_error)?,
                )
            } else {
                None
            };
            let tables = DeviceBuffer::from_slice(
                &self.gpu.device,
                if flat.is_empty() { &placeholder } else { &flat },
            )
            .map_err(compute_error)?;
            let empty_rows =
                DeviceBuffer::from_slice(&self.gpu.device, &[[0u64; 4]]).map_err(compute_error)?;
            let device_tails =
                DeviceBuffer::from_slice(&self.gpu.device, tails).map_err(compute_error)?;
            let (rows, packed_tail) =
                uploaded.map_or((&empty_rows, [0; 4]), |rows| (&rows.prefix, rows.tail));
            let dense = self.device.as_ref().map_or(&tables, |state| &state.values);
            let rho = DeviceBuffer::from_slice(&self.gpu.device, &self.cpu.gamma_powers[..active])
                .map_err(compute_error)?;
            let selectors = DeviceBuffer::from_slice(
                &self.gpu.device,
                if selectors.is_empty() {
                    &[[0u32; 2]][..]
                } else {
                    &selectors
                },
            )
            .map_err(compute_error)?;
            let e_in = DeviceBuffer::from_slice(&self.gpu.device, eq.e_in_current())
                .map_err(compute_error)?;
            let e_out = DeviceBuffer::from_slice(&self.gpu.device, eq.e_out_current())
                .map_err(compute_error)?;
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
            let mut cached_params = [0u32; 12];
            cached_params[..9].copy_from_slice(&params);
            if let Some(cache) = &cached {
                cached_params[9..].copy_from_slice(&[
                    cache.mode,
                    cache.states,
                    cache.square_stride,
                ]);
            }
            let mut batch = Batch::new(&self.gpu.device).map_err(compute_error)?;
            if let Some(cache) = &cached {
                // SAFETY: canonical tables cover every checked category and
                // active column. The two layouts have distinct mode tags.
                // All row loads use the retained prefix/tail; all groups write
                // three partials. Cache owners and 48-byte params outlive wait.
                unsafe {
                    batch
                        .dispatch_unchecked(
                            cached_pipeline,
                            &[
                                Binding::buffer(rows),
                                Binding::buffer(&rho),
                                Binding::buffer(&selectors),
                                Binding::buffer(&e_in),
                                Binding::buffer(&e_out),
                                Binding::buffer(&partials),
                                Binding::value(&cached_params),
                                Binding::value(&packed_tail),
                                Binding::buffer(&cache.values),
                                Binding::buffer(cache.squares.as_ref().unwrap_or(&cache.values)),
                            ],
                            Grid::linear(groups * cached_group, cached_group),
                        )
                        .map_err(compute_error)?;
                }
            } else {
                // SAFETY: compact-row loads use the tail beyond the checked prefix.
                // Selectors and branch tables cover every active column; each group
                // owns three partials and all its lanes reach the group barriers.
                // Width-one constants cover radix+1 states for each active column;
                // other routes reuse the typed table binding which the shader never reads.
                unsafe {
                    batch
                        .dispatch_unchecked(
                            pipeline,
                            &[
                                Binding::buffer(rows),
                                Binding::buffer(&tables),
                                Binding::buffer(&rho),
                                Binding::buffer(&selectors),
                                Binding::buffer(&e_in),
                                Binding::buffer(&e_out),
                                Binding::buffer(&partials),
                                Binding::value(&params),
                                Binding::value(&packed_tail),
                                Binding::buffer(dense),
                                Binding::buffer(&device_tails),
                                Binding::buffer(initial_constants.as_ref().unwrap_or(&tables)),
                            ],
                            Grid::linear(groups * group_size, group_size),
                        )
                        .map_err(compute_error)?;
                }
            }
            let gpu_time = batch.commit_and_wait().map_err(compute_error)?;
            let _ = span.record("gpu_s", gpu_time.as_secs_f64());
            for partial in partials.read().map_err(compute_error)?.chunks_exact(3) {
                for (total, value) in totals.iter_mut().zip(partial) {
                    *total += *value;
                }
            }
        }
        let tail = tails
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

impl MetalBooleanityKernel {
    fn selectors(source: &BooleanityChunks) -> Option<Vec<[u32; 2]>> {
        source
            .selectors
            .iter()
            .map(|selector| {
                let (word, shift) = match selector {
                    ColumnSelector::Instruction(selector) => {
                        let shift = selector.shift();
                        if shift >= 128 {
                            return None;
                        }
                        (u32::from(shift >= 64), shift % 64)
                    }
                    ColumnSelector::Bytecode(selector) => (2, selector.shift()),
                    ColumnSelector::Ram(selector) => (3, selector.shift()),
                };
                (shift + source.chunk_bits <= 64).then_some([word, shift as u32])
            })
            .collect()
    }

    fn restore_cpu(&mut self) -> Result<(), SumcheckError<AkitaField>> {
        if let Some(device) = self.device.take() {
            self.cpu.tables = Some(device.restore().map_err(compute_error)?);
        }
        Ok(())
    }
    fn try_materialize(
        &mut self,
        challenge: AkitaField,
        minimum: usize,
    ) -> Result<bool, SumcheckError<AkitaField>> {
        let Some(folded) = self.cpu.tables.as_ref() else {
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
        if *prefix_len < minimum || !matches!(source.chunk_bits, 4 | 8) {
            return Ok(false);
        }
        let Some(selectors) = Self::selectors(source) else {
            return Ok(false);
        };
        let owner = self.rows.as_ref().ok_or_else(|| {
            compute_error(MetalError::InvalidDispatch {
                pipeline: "resident booleanity".into(),
                reason: "missing compact rows".into(),
            })
        })?;
        let resident = owner.resident(&self.gpu.device).map_err(compute_error)?;
        let rows = resident.as_ref().ok_or_else(|| {
            compute_error(MetalError::InvalidDispatch {
                pipeline: "resident booleanity".into(),
                reason: "missing uploaded rows".into(),
            })
        })?;
        let device = self
            .gpu
            .fold
            .materialize(rows, tables, *width, &selectors, prefix, tail, challenge)
            .map_err(compute_error)?;
        drop(resident);
        crate::mem::drop_in_background_thread(self.cpu.tables.take());
        self.device = Some(device);
        self.rows = None;
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
            || matches!(self.cpu.tables.as_ref(), Some(LazyFoldedRa::Lazy { prefix_len, .. }) if *prefix_len >= minimum);
        if admitted {
            if let Some(message) = self.gpu_message(round, claim)? {
                return Ok(message);
            }
        } else {
            self.rows = None;
        }
        self.cpu.message(round, claim)
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
        claim: AkitaField,
    ) -> Result<UnivariatePoly<AkitaField>, SumcheckError<AkitaField>> {
        self.round_with_cutoff(bind, round, claim, MIN_GPU_PREFIX)
    }
    fn finish_rounds(&mut self, bind: AkitaField) -> Result<(), SumcheckError<AkitaField>> {
        self.restore_cpu()?;
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
