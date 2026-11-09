use jolt_akita::AkitaField;
use jolt_field::Zero;
use jolt_metal::runtime::{
    host_name, Batch, Binding, Device, DeviceBuffer, Grid, LibrarySpec, ShaderLibrary,
};
use jolt_metal::shaders::FIELD_HEADERS;
use jolt_metal::MetalError;

use super::super::metal_rows::{ResidentInstructionRows, UploadedRows};
use super::*;

mod address;

const MIN_GPU_PREFIX: usize = 1 << 16;
const CPU_TAIL_ROWS: usize = 1 << 12;
const MAX_SIMDGROUPS: usize = 8;

pub(crate) struct MetalInstructionReadRaf(Arc<ReadRafGpu>);

impl MetalInstructionReadRaf {
    pub(crate) fn new() -> Result<Self, MetalError> {
        let device = Device::system_default()?;
        let mut spec = LibrarySpec::new();
        for (name, source) in FIELD_HEADERS {
            spec = spec.source(name, source);
        }
        let mut spec = spec
            .source(
                "read_raf_geometry",
                &format!("#define READ_RAF_MAX_SIMDGROUPS {MAX_SIMDGROUPS}"),
            )
            .source("read_raf.metal", include_str!("read_raf.metal"))
            .instantiate::<AkitaField>("read_raf_round9")
            .instantiate::<AkitaField>("read_raf_round17");
        spec = spec.source(
            "address_geometry",
            &format!("#define ADDRESS_BINS {CHUNK_SIZE}u"),
        );
        spec = spec.source("address.metal", include_str!("metal/address.metal"));
        for name in address::KERNELS {
            spec = spec.instantiate::<AkitaField>(name);
        }
        #[cfg(test)]
        {
            spec = spec.instantiate::<AkitaField>("address_suffix_oracle");
        }
        let library = ShaderLibrary::compile(&device, &spec)?;
        Ok(Self(Arc::new(ReadRafGpu { device, library })))
    }
}

impl PrepareKernel<AkitaField, InstructionReadRaf<AkitaField>> for MetalInstructionReadRaf {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<AkitaField>,
        inputs: ProverInputs<'_, AkitaField, InstructionReadRaf<AkitaField>>,
    ) -> Result<
        Box<dyn SumcheckKernel<AkitaField, Relation = InstructionReadRaf<AkitaField>>>,
        KernelError<AkitaField>,
    > {
        let cpu = OptimizedInstructionReadRafKernel::prepare_with_scanner(
            session,
            witness,
            inputs,
            |session, host| match address::DeviceAddressScan::try_new(
                Arc::clone(&self.0),
                session,
                &host,
                MIN_GPU_PREFIX,
            )
            .map_err(compute_error)?
            {
                Some(device) => Ok(Box::new(device)),
                None => Ok(Box::new(host)),
            },
        )?;
        let resident = ResidentInstructionRows::shared(session, &cpu.rows)?;
        Ok(Box::new(MetalReadRafKernel {
            cpu,
            rows: Some(resident),
            device: None,
            gpu: Arc::clone(&self.0),
        }))
    }
}

struct ReadRafGpu {
    device: Device,
    library: ShaderLibrary,
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct DeviceCycle {
    gruen: GruenSplitEqPolynomial<AkitaField>,
    /// Combined-value and RA prefix lengths; the allocation uses their maximum.
    prefixes: [usize; 2],
    tails: Vec<AkitaField>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    values: DeviceBuffer<AkitaField>,
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct MetalReadRafKernel {
    cpu: OptimizedInstructionReadRafKernel<AkitaField>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    rows: Option<Arc<ResidentInstructionRows>>,
    /// While resident, the CPU cycle is absent. It is restored before CPU work
    /// or extraction; address state and final flag columns stay on the CPU.
    device: Option<DeviceCycle>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    gpu: Arc<ReadRafGpu>,
}

enum RoundSource<'a> {
    Lazy {
        rows: &'a UploadedRows,
        ra: &'a [Vec<AkitaField>],
        claims: &'a [AkitaField],
        width: usize,
        bits: usize,
        bind: Option<AkitaField>,
    },
    Dense {
        values: &'a DeviceBuffer<AkitaField>,
        prefix: usize,
        bind: AkitaField,
    },
}

struct RoundOutput {
    polynomial: UnivariatePoly<AkitaField>,
    bound_prefix: Option<DeviceBuffer<AkitaField>>,
}

impl ReadRafGpu {
    fn round(
        &self,
        source: RoundSource<'_>,
        eq: &GruenSplitEqPolynomial<AkitaField>,
        prefixes: [usize; 2],
        tails: &[AkitaField],
        round: usize,
        claim: AkitaField,
    ) -> Result<RoundOutput, SumcheckError<AkitaField>> {
        let n = tails.len();
        let template = match n {
            9 => "read_raf_round9",
            17 => "read_raf_round17",
            _ => return Err(compute_error(invalid("unsupported product factor count"))),
        };
        let prefix = prefixes[0].max(prefixes[1]);
        let pairs = prefix.div_ceil(2);
        let in_len = eq.e_in_current_len();
        if in_len == 0 || pairs.div_ceil(in_len) > eq.e_out_current_len() {
            return Err(compute_error(invalid(
                "equality factors do not cover the prefix",
            )));
        }
        let need_zero = eq.current_linear_evals().0.is_zero() && !eq.current_scalar().is_zero();
        let (mode, input_prefix, width, bits, bind) = match &source {
            RoundSource::Lazy {
                rows,
                width,
                bits,
                bind,
                ..
            } => (
                usize::from(bind.is_some()),
                rows.prefix.len(),
                *width,
                *bits,
                bind.unwrap_or_else(AkitaField::zero),
            ),
            RoundSource::Dense { prefix, bind, .. } => (2, *prefix, 0, 0, *bind),
        };
        let output_len = if mode == 0 {
            // Metal validates one pointee even when this mode never writes it.
            1
        } else {
            prefix
                .checked_mul(n)
                .ok_or_else(|| compute_error(invalid("device prefix size overflows")))?
        };
        let output =
            DeviceBuffer::<AkitaField>::zeroed(&self.device, output_len).map_err(compute_error)?;
        let mut totals = vec![AkitaField::zero(); n + 1];
        if pairs != 0 {
            let span = tracing::info_span!(
                "metal_read_raf_dispatch",
                mode,
                width,
                gpu_s = tracing::field::Empty
            );
            let _span = span.enter();
            let mut params = [0u32; 9];
            for (out, value) in params.iter_mut().zip([
                input_prefix,
                pairs,
                width,
                n - 1,
                bits,
                in_len,
                mode,
                prefix,
                usize::from(need_zero),
            ]) {
                *out = u32::try_from(value)
                    .map_err(|_| compute_error(invalid("shape exceeds 32-bit indexing")))?;
            }
            // Placeholder bindings must contain a complete shader element;
            // the runtime's minimum allocation is smaller than a ulong4 row.
            let unused_fields =
                DeviceBuffer::<AkitaField>::zeroed(&self.device, 1).map_err(compute_error)?;
            let unused_rows =
                DeviceBuffer::<[u64; 4]>::zeroed(&self.device, 1).map_err(compute_error)?;
            let placeholder = [AkitaField::zero()];
            let (flat_ra, claim_table) = match &source {
                RoundSource::Lazy { ra, claims, .. } => {
                    (ra.iter().flatten().copied().collect::<Vec<_>>(), *claims)
                }
                RoundSource::Dense { .. } => (Vec::new(), placeholder.as_slice()),
            };
            let ra = DeviceBuffer::from_slice(
                &self.device,
                if flat_ra.is_empty() {
                    &placeholder
                } else {
                    &flat_ra
                },
            )
            .map_err(compute_error)?;
            let claims =
                DeviceBuffer::from_slice(&self.device, claim_table).map_err(compute_error)?;
            let device_tails =
                DeviceBuffer::from_slice(&self.device, tails).map_err(compute_error)?;
            let tail_cache = TailFactorProducts::new(&tails[1..]);
            let cached = DeviceBuffer::from_slice(
                &self.device,
                tail_cache
                    .as_ref()
                    .map_or(placeholder.as_slice(), |cache| cache.products().as_slice()),
            )
            .map_err(compute_error)?;
            let e_in =
                DeviceBuffer::from_slice(&self.device, eq.e_in_current()).map_err(compute_error)?;
            let e_out = DeviceBuffer::from_slice(&self.device, eq.e_out_current())
                .map_err(compute_error)?;
            let groups = pairs.div_ceil(in_len);
            let partial_len = groups
                .checked_mul(n + 1)
                .ok_or_else(|| compute_error(invalid("partial size overflows")))?;
            let mut partials = DeviceBuffer::<AkitaField>::zeroed(&self.device, partial_len)
                .map_err(compute_error)?;
            let pipeline = self
                .library
                .pipeline(&host_name::<AkitaField>(template))
                .map_err(compute_error)?;
            let simd = pipeline.thread_execution_width();
            if simd == 0 {
                return Err(compute_error(invalid("zero SIMD width")));
            }
            let simdgroups =
                (pipeline.max_total_threads_per_threadgroup().min(128) / simd).min(MAX_SIMDGROUPS);
            let threads = simdgroups * simd;
            if threads == 0 || simdgroups > simd {
                return Err(compute_error(invalid("unsupported SIMD geometry")));
            }
            let (rows, tail, previous) = match &source {
                RoundSource::Lazy { rows, .. } => (&rows.prefix, rows.tail, &unused_fields),
                RoundSource::Dense { values, .. } => (&unused_rows, [0; 4], *values),
            };
            let mut batch = Batch::new(&self.device).map_err(compute_error)?;
            // SAFETY: the selected mode accesses only its populated inputs.
            // Lazy loads substitute the representative suffix; dense loads use
            // factor tails. Groups own disjoint row/partial ranges, and all
            // dispatched lanes reach each reduction barrier.
            unsafe {
                batch
                    .dispatch_unchecked(
                        pipeline,
                        &[
                            Binding::buffer(rows),
                            Binding::buffer(&ra),
                            Binding::buffer(&claims),
                            Binding::buffer(previous),
                            Binding::buffer(&output),
                            Binding::buffer(&device_tails),
                            Binding::buffer(&cached),
                            Binding::buffer(&e_in),
                            Binding::buffer(&e_out),
                            Binding::buffer(&partials),
                            Binding::value(&params),
                            Binding::value(&tail),
                            Binding::value(&bind),
                        ],
                        Grid::linear(groups * threads, threads),
                    )
                    .map_err(compute_error)?;
            }
            let elapsed = batch.commit_and_wait().map_err(compute_error)?;
            let _ = span.record("gpu_s", elapsed.as_secs_f64());
            for values in partials.read().map_err(compute_error)?.chunks_exact(n + 1) {
                for (sum, value) in totals.iter_mut().zip(values) {
                    *sum += *value;
                }
            }
        }
        let suffix = eq.suffix_weight(pairs) * tails.iter().copied().product::<AkitaField>();
        for value in totals.iter_mut().take(n - 1) {
            *value += suffix;
        }
        totals[n] += suffix;
        let polynomial = eq.checked_toom(&totals[..n], claim, round, || totals[n])?;
        Ok(RoundOutput {
            polynomial,
            bound_prefix: (mode != 0).then_some(output),
        })
    }
}

impl MetalReadRafKernel {
    fn lazy_geometry(&self) -> Option<usize> {
        let cycle = self.cpu.cycle.as_ref()?;
        let (
            LazyFoldedRa::Lazy {
                tables: ra,
                width,
                factors: 1,
                prefix_len,
                source,
                ..
            },
            LazyFoldedRa::Lazy {
                tables: claims,
                width: claim_width,
                factors: 1,
                ..
            },
        ) = (&cycle.ra, &cycle.combined_val)
        else {
            return None;
        };
        let columns = ra.len();
        if !matches!(columns, 8 | 16)
            || !matches!(*width, 1 | 2 | 4)
            || claim_width != width
            || claims.len() != 1
            || claims[0].len() != width * 256
            || !matches!(source.chunk_bits, 8 | 16)
            || source.chunks != columns
            || columns * source.chunk_bits != 128
            || ra
                .iter()
                .any(|table| table.len() != width * (1 << source.chunk_bits))
        {
            return None;
        }
        Some(*prefix_len)
    }

    fn lazy_round(
        &mut self,
        bind: Option<AkitaField>,
        round: usize,
        claim: AkitaField,
    ) -> Result<Option<UnivariatePoly<AkitaField>>, SumcheckError<AkitaField>> {
        if self.lazy_geometry().is_none() {
            return Ok(None);
        }
        let cycle = self
            .cpu
            .cycle
            .as_ref()
            .ok_or_else(|| compute_error(invalid("missing CPU cycle")))?;
        let prefixes = if bind.is_some() {
            match (
                cycle.combined_val.next_materialized_len(),
                cycle.ra.next_materialized_len(),
            ) {
                (Some(c), Some(r)) => [c, r],
                _ => return Ok(None),
            }
        } else {
            [cycle.combined_val.prefix_len(), cycle.ra.prefix_len()]
        };
        let (
            LazyFoldedRa::Lazy {
                tables: ra,
                width,
                source,
                ..
            },
            LazyFoldedRa::Lazy { tables: claims, .. },
        ) = (&cycle.ra, &cycle.combined_val)
        else {
            return Ok(None);
        };
        if bind.is_none() && *width == 1 {
            return Ok(None);
        }
        let resident = self
            .rows
            .as_ref()
            .ok_or_else(|| compute_error(invalid("missing shared rows")))?
            .resident(&self.gpu.device)
            .map_err(compute_error)?;
        let rows = resident
            .as_ref()
            .ok_or_else(|| compute_error(invalid("missing uploaded rows")))?;
        if rows.prefix.len() != source.rows.prefix_len() {
            return Err(compute_error(invalid("resident source prefix changed")));
        }
        let tails: Vec<_> = cycle
            .combined_val
            .tail_values()
            .iter()
            .chain(cycle.ra.tail_values())
            .copied()
            .collect();
        let mut gruen = cycle.gruen.clone();
        if let Some(challenge) = bind {
            gruen.bind(challenge);
        }
        let RoundOutput {
            polynomial: message,
            bound_prefix: values,
        } = self.gpu.round(
            RoundSource::Lazy {
                rows,
                ra,
                claims: &claims[0],
                width: *width,
                bits: source.chunk_bits,
                bind,
            },
            &gruen,
            prefixes,
            &tails,
            round,
            claim,
        )?;
        drop(resident);
        if let Some(challenge) = bind {
            let values =
                values.ok_or_else(|| compute_error(invalid("materialization omitted output")))?;
            let old = self.cpu.cycle.take();
            crate::mem::drop_in_background_thread(old);
            self.device = Some(DeviceCycle {
                gruen,
                prefixes,
                tails,
                values,
            });
            self.cpu.record_cycle_bind(challenge);
        }
        Ok(Some(message))
    }

    fn restore_cpu(&mut self) -> Result<(), SumcheckError<AkitaField>> {
        let Some(mut device) = self.device.take() else {
            return Ok(());
        };
        let n = device.tails.len();
        let rows = device.values.read().map_err(compute_error)?;
        let combined = map_indices(device.prefixes[0], |row| rows[row * n]);
        let ra = map_indices(device.prefixes[1] * (n - 1), |i| {
            rows[(i / (n - 1)) * n + 1 + i % (n - 1)]
        });
        let mut tails = device.tails;
        let combined_tail = tails.remove(0);
        self.cpu.cycle = Some(CycleState {
            gruen: device.gruen,
            combined_val: LazyFoldedRa::Prefix {
                values: vec![combined],
                tail: vec![combined_tail],
            },
            ra: LazyFoldedRa::Rows {
                values: ra,
                tail: tails,
            },
        });
        Ok(())
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
            .is_some_and(|state| state.prefixes[0].max(state.prefixes[1]) <= CPU_TAIL_ROWS)
        {
            self.restore_cpu()?;
        }
        if let Some(device) = &self.device {
            let challenge =
                bind.ok_or_else(|| compute_error(invalid("resident round has no bind")))?;
            let prefixes = device.prefixes.map(|length| length.div_ceil(2));
            let mut gruen = device.gruen.clone();
            gruen.bind(challenge);
            let RoundOutput {
                polynomial: message,
                bound_prefix: values,
            } = self.gpu.round(
                RoundSource::Dense {
                    values: &device.values,
                    prefix: device.prefixes[0].max(device.prefixes[1]),
                    bind: challenge,
                },
                &gruen,
                prefixes,
                &device.tails,
                round,
                claim,
            )?;
            let values =
                values.ok_or_else(|| compute_error(invalid("dense bind omitted output")))?;
            let tails = device.tails.clone();
            self.device = Some(DeviceCycle {
                gruen,
                prefixes,
                tails,
                values,
            });
            self.cpu.record_cycle_bind(challenge);
            return Ok(message);
        }
        if let Some(challenge) = bind {
            if self.lazy_geometry().is_some_and(|prefix| prefix >= minimum) {
                if let Some(message) = self.lazy_round(Some(challenge), round, claim)? {
                    return Ok(message);
                }
            }
            self.cpu.bind(challenge)?;
        }
        if self.cpu.progress.bound() < self.cpu.address_bits() {
            return Ok(self.cpu.address_message(claim));
        }
        if self.lazy_geometry().is_some_and(|prefix| prefix >= minimum) {
            if let Some(message) = self.lazy_round(None, round, claim)? {
                return Ok(message);
            }
        }
        self.cpu.cycle_message(round, claim)
    }
}

impl ProveRounds<AkitaField> for MetalReadRafKernel {
    fn num_rounds(&self) -> usize {
        self.cpu.num_rounds()
    }
    fn prove_round(
        &mut self,
        bind: Option<AkitaField>,
        round: usize,
        previous_claim: AkitaField,
    ) -> Result<UnivariatePoly<AkitaField>, SumcheckError<AkitaField>> {
        self.round_with_cutoff(bind, round, previous_claim, MIN_GPU_PREFIX)
    }
    fn finish_rounds(&mut self, bind: AkitaField) -> Result<(), SumcheckError<AkitaField>> {
        self.restore_cpu()?;
        self.cpu.finish_rounds(bind)
    }
}

impl SumcheckKernel<AkitaField> for MetalReadRafKernel {
    type Relation = InstructionReadRaf<AkitaField>;
    fn output_claims(
        &mut self,
        inputs: &SumcheckInputClaims<AkitaField, Self::Relation>,
    ) -> Result<InstructionReadRafOutputClaims<AkitaField>, SumcheckKernelError<AkitaField>> {
        self.cpu.output_claims(inputs)
    }
    fn park_residue(mut self: Box<Self>, session: &mut ProofSession) {
        if let Some(rows) = self.rows.take() {
            rows.carry_to_next_stage(session);
        }
        crate::mem::drop_in_background_thread(self);
    }
}

fn invalid(reason: &str) -> MetalError {
    MetalError::InvalidDispatch {
        pipeline: "instruction read RAF".into(),
        reason: reason.into(),
    }
}
fn compute_error(error: MetalError) -> SumcheckError<AkitaField> {
    SumcheckError::Compute {
        backend: "Metal instruction read RAF",
        reason: error.to_string(),
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests;
