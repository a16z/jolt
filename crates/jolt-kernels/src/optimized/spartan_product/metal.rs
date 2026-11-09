//! Native stage-2 product computation. Source storage is bounded by a tile;
//! only the shrinking remainder fields remain resident between challenges.
use std::time::Duration;

use bytemuck::{Pod, Zeroable};
use jolt_akita::AkitaField;
use jolt_field::{Ring, Zero};
use jolt_metal::runtime::{
    host_name, Batch, Binding, Device, DeviceBuffer, Grid, LibrarySpec, ShaderLibrary,
};
use jolt_metal::shaders::FIELD_HEADERS;
use jolt_metal::MetalError;

use super::*;

const TILE_ROWS: usize = 1 << 20;
const MIN_PREFIX: usize = 1 << 16;
const CPU_TAIL: usize = 1 << 12;
const THREADS: usize = 256;
const NATIVE: &str = "spartan_product_native";
const BIND: &str = "spartan_product_bind";
const ROUND: &str = "spartan_product_round";

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct NativeRow {
    left: u64,
    lookup: u64,
    right: [u64; 2],
    flags: u32,
    reserved: u32,
}
const _: () = assert!(size_of::<NativeRow>() == 40);

impl NativeRow {
    fn from_row(row: SpartanProductRow) -> Self {
        let right = row.right_instruction_input.0 as u128;
        let flags = [
            row.jump_flag.0,
            row.write_lookup_output_to_rd.0,
            row.branch_flag.0,
            row.next_is_noop.0,
            row.virtual_instruction.0,
        ];
        Self {
            left: row.left_instruction_input.0,
            lookup: row.lookup_output.0,
            right: [right as u64, (right >> 64) as u64],
            flags: flags
                .into_iter()
                .enumerate()
                .fold(0, |bits, (i, set)| bits | (u32::from(set) << i)),
            reserved: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    first: u32,
    count: u32,
    in_len: u32,
    outputs: u32,
    mode: u32,
    prefix: u32,
    pairs: u32,
    need_one: u32,
}

struct ProductGpu {
    device: Device,
    library: ShaderLibrary,
}

#[derive(Clone)]
pub(crate) struct MetalProductUniskip(Arc<ProductGpu>);

impl MetalProductUniskip {
    pub(crate) fn new() -> Result<Self, MetalError> {
        let device = Device::system_default()?;
        let mut spec = LibrarySpec::new();
        for (name, source) in FIELD_HEADERS {
            spec = spec.source(name, source);
        }
        spec = spec.source("spartan_product.metal", include_str!("product.metal"));
        for name in [NATIVE, BIND, ROUND] {
            spec = spec.instantiate::<AkitaField>(name);
        }
        let library = ShaderLibrary::compile(&device, &spec)?;
        for name in [NATIVE, BIND, ROUND] {
            let pipeline = library.pipeline(&host_name::<AkitaField>(name))?;
            if pipeline.thread_execution_width() != 32
                || pipeline.max_total_threads_per_threadgroup() < THREADS
            {
                return Err(invalid("product requires SIMD32 and 256 threads"));
            }
        }
        Ok(Self(Arc::new(ProductGpu { device, library })))
    }
}

impl
    UniskipKernel<
        AkitaField,
        ProductRemainder<AkitaField>,
        SumcheckInputClaims<AkitaField, ProductUniskip<AkitaField>>,
    > for MetalProductUniskip
{
    #[tracing::instrument(skip_all, name = "SpartanProductUniskip::prepare")]
    fn prepare(
        &self,
        session: &mut ProofSession,
        log_t: usize,
        tau_low: &[AkitaField],
        witness: &dyn JoltWitnessPlane<AkitaField>,
    ) -> Result<(), KernelError<AkitaField>> {
        OptimizedProductUniskip::prepare_with_compute(
            session,
            log_t,
            tau_low,
            witness,
            Some(Arc::new(self.clone())),
        )
    }
    fn first_round_poly(
        &self,
        session: &mut ProofSession,
        late_tau: &[AkitaField],
        inputs: &SumcheckInputClaims<AkitaField, ProductUniskip<AkitaField>>,
    ) -> Result<UnivariatePoly<AkitaField>, KernelError<AkitaField>> {
        OptimizedProductUniskip.first_round_poly(session, late_tau, inputs)
    }
}

fn word(value: usize) -> Result<u32, MetalError> {
    u32::try_from(value).map_err(|_| invalid("product index exceeds u32"))
}
fn field_len(rows: usize) -> Result<usize, MetalError> {
    rows.checked_mul(2)
        .ok_or_else(|| invalid("product field extent overflows"))
}
fn invalid(reason: &str) -> MetalError {
    MetalError::InvalidDispatch {
        pipeline: "Spartan product".into(),
        reason: reason.into(),
    }
}
fn compute_error(error: MetalError) -> SumcheckError<AkitaField> {
    SumcheckError::Compute {
        backend: "Metal Spartan product",
        reason: error.to_string(),
    }
}
fn kernel_error(error: MetalError) -> KernelError<AkitaField> {
    compute_error(error).into()
}

fn pack(
    rows: &BundleAccess<'_, SpartanProductRow>,
    first: usize,
    count: usize,
) -> Result<Vec<NativeRow>, WitnessError> {
    let _span = tracing::info_span!("metal_spartan_product_pack", first, count).entered();
    #[cfg(feature = "parallel")]
    return (first..first + count)
        .into_par_iter()
        .map(|i| rows.row(i).map(NativeRow::from_row))
        .collect();
    #[cfg(not(feature = "parallel"))]
    (first..first + count)
        .map(|i| rows.row(i).map(NativeRow::from_row))
        .collect()
}

struct Equality {
    inner: DeviceBuffer<AkitaField>,
    outer: DeviceBuffer<AkitaField>,
    inner_len: usize,
    outer_len: usize,
}
impl Equality {
    fn new(
        device: &Device,
        inner: &[AkitaField],
        outer: &[AkitaField],
    ) -> Result<Self, MetalError> {
        if inner.is_empty() || outer.is_empty() || !inner.len().is_power_of_two() {
            return Err(invalid("invalid split equality"));
        }
        let _ = word(inner.len())?;
        Ok(Self {
            inner: DeviceBuffer::from_slice(device, inner)?,
            outer: DeviceBuffer::from_slice(device, outer)?,
            inner_len: inner.len(),
            outer_len: outer.len(),
        })
    }
}

fn sum_partials(
    partials: &mut DeviceBuffer<AkitaField>,
    outputs: usize,
) -> Result<Vec<AkitaField>, MetalError> {
    let mut result = vec![AkitaField::zero(); outputs];
    for row in partials.read()?.chunks_exact(outputs) {
        for (sum, value) in result.iter_mut().zip(row) {
            *sum += *value;
        }
    }
    Ok(result)
}

struct NativeRequest<'a> {
    count: usize,
    mode: u32,
    outputs: usize,
    weights: &'a [AkitaField],
    eq: &'a Equality,
    values: &'a DeviceBuffer<AkitaField>,
    need_one: bool,
}

impl ProductGpu {
    /// Tiles align to complete equality blocks, including pairs in producer mode.
    /// The input is never materialized beyond this bounded transfer window.
    fn native(
        &self,
        rows: &BundleAccess<'_, SpartanProductRow>,
        request: NativeRequest<'_>,
    ) -> Result<Vec<AkitaField>, KernelError<AkitaField>> {
        let NativeRequest {
            count,
            mode,
            outputs,
            weights,
            eq,
            values,
            need_one,
        } = request;
        let span = tracing::info_span!(
            "metal_spartan_product_native",
            mode,
            rows = count,
            gpu_s = tracing::field::Empty
        );
        let _span = span.enter();
        if outputs == 0 || outputs > 8 {
            return Err(kernel_error(invalid(
                "native product output count exceeds scratch",
            )));
        }
        let valid_layout = match mode {
            0 => weights.len() >= 3 * outputs,
            1 => {
                outputs == 3
                    && weights.len() >= 3
                    && values.len() >= field_len(count).map_err(kernel_error)?
            }
            2 => outputs == 8 && !weights.is_empty(),
            _ => false,
        };
        if !valid_layout {
            return Err(kernel_error(invalid(
                "native product buffer layout disagrees with mode",
            )));
        }
        let units = if mode == 1 { 2 } else { 1 };
        let block_rows = eq
            .inner_len
            .checked_mul(units)
            .ok_or_else(|| kernel_error(invalid("equality block extent overflows")))?;
        if block_rows > TILE_ROWS
            || count.div_ceil(block_rows) > eq.outer_len
            || (mode == 1 && count % 2 != 0)
        {
            return Err(kernel_error(invalid(
                "native product shape exceeds bounded tiles",
            )));
        }
        let _ = word(count).map_err(kernel_error)?;
        let weights = DeviceBuffer::from_slice(&self.device, weights).map_err(kernel_error)?;
        let mut totals = vec![AkitaField::zero(); outputs];
        let mut gpu_time = Duration::ZERO;
        let tile_rows = (TILE_ROWS / block_rows) * block_rows;
        for first in (0..count).step_by(tile_rows) {
            let count = tile_rows.min(count - first);
            let packed = pack(rows, first, count)?;
            let upload =
                tracing::info_span!("metal_spartan_product_upload", rows = count).entered();
            let source = DeviceBuffer::from_slice(&self.device, &packed).map_err(kernel_error)?;
            drop(upload);
            drop(packed);
            let groups = count.div_ceil(block_rows);
            let mut partials =
                DeviceBuffer::zeroed(&self.device, groups * outputs).map_err(kernel_error)?;
            let params = Params {
                first: word(first).map_err(kernel_error)?,
                count: word(count).map_err(kernel_error)?,
                in_len: word(eq.inner_len).map_err(kernel_error)?,
                outputs: word(outputs).map_err(kernel_error)?,
                mode,
                prefix: 0,
                pairs: 0,
                need_one: u32::from(need_one),
            };
            let mut batch = Batch::new(&self.device).map_err(kernel_error)?;
            // SAFETY: checked source/equality blocks bound every read; producer
            // writes only its disjoint global output window. Every lane reaches
            // all reductions, and <=TILE_ROWS terms fit each accumulator.
            unsafe {
                batch
                    .dispatch_unchecked(
                        self.library
                            .pipeline(&host_name::<AkitaField>(NATIVE))
                            .map_err(kernel_error)?,
                        &[
                            Binding::buffer(&source),
                            Binding::buffer(&weights),
                            Binding::buffer(&eq.inner),
                            Binding::buffer(&eq.outer),
                            Binding::buffer(values),
                            Binding::buffer(&partials),
                            Binding::value(&params),
                        ],
                        Grid::linear(groups * THREADS, THREADS),
                    )
                    .map_err(kernel_error)?;
            }
            gpu_time += batch.commit_and_wait().map_err(kernel_error)?;
            for (total, value) in totals
                .iter_mut()
                .zip(sum_partials(&mut partials, outputs).map_err(kernel_error)?)
            {
                *total += value;
            }
        }
        let _ = span.record("gpu_s", gpu_time.as_secs_f64());
        Ok(totals)
    }
}

fn factors(row: &SpartanProductRow, weights: &[AkitaField]) -> [AkitaField; 2] {
    [
        weights[0] * AkitaField::from_u64(row.left_instruction_input.0)
            + weights[1] * AkitaField::from_u64(row.lookup_output.0)
            + weights[2] * AkitaField::from_u64(u64::from(row.jump_flag.0)),
        weights[0] * AkitaField::from_i128(row.right_instruction_input.0)
            + weights[1] * AkitaField::from_u64(u64::from(row.branch_flag.0))
            + weights[2] * AkitaField::from_u64(1 - u64::from(row.next_is_noop.0)),
    ]
}

impl ProductCompute<AkitaField> for MetalProductUniskip {
    fn admits(&self, prefix: usize, log_t: usize) -> bool {
        let Some(bytes) = prefix
            .div_ceil(2)
            .checked_mul(2 * 2 * size_of::<AkitaField>())
        else {
            return false;
        };
        let limits = self.0.device.limits();
        // Admission precedes uniskip/device ownership. Allocation failures from
        // competing live buffers still propagate, rather than changing backends.
        prefix >= MIN_PREFIX
            && prefix <= u32::MAX as usize - 1
            && log_t <= 32
            && DOMAIN >= 3
            && EXTENSION_SIZE <= 8
            && bytes <= limits.max_buffer_length
            && (bytes as u128 * 3 / 2 + 2 * (TILE_ROWS * size_of::<NativeRow>()) as u128)
                <= u128::from(limits.recommended_max_working_set_size)
    }
    fn extension_values(
        &self,
        rows: &BundleAccess<'_, SpartanProductRow>,
        prefix: usize,
        point: &[AkitaField],
        coefficients: &[[i64; DOMAIN]; EXTENSION_SIZE],
    ) -> Result<Vec<AkitaField>, KernelError<AkitaField>> {
        let _span = tracing::info_span!("metal_spartan_product_uniskip").entered();
        let split = point.len() / 2;
        let outer = EqPolynomial::evals(&point[..split], None);
        let inner = EqPolynomial::evals(&point[split..], None);
        let eq = Equality::new(&self.0.device, &inner, &outer).map_err(kernel_error)?;
        let weights: Vec<_> = coefficients
            .iter()
            .flat_map(|row| row[..3].iter().copied().map(AkitaField::from_i64))
            .collect();
        let unused = DeviceBuffer::<AkitaField>::zeroed(&self.0.device, 1).map_err(kernel_error)?;
        let mut totals = self.0.native(
            rows,
            NativeRequest {
                count: prefix,
                mode: 0,
                outputs: EXTENSION_SIZE,
                weights: &weights,
                eq: &eq,
                values: &unused,
                need_one: false,
            },
        )?;
        if prefix < 1usize << point.len() {
            let mass = EqPolynomial::suffix_sum(point, prefix);
            let row = rows.row(prefix)?;
            for (total, weights) in totals.iter_mut().zip(weights.chunks_exact(3)) {
                let [a, b] = factors(&row, weights);
                *total += mass * a * b;
            }
        }
        Ok(totals)
    }
    fn prepare_columns(
        &self,
        rows: &BundleAccess<'_, SpartanProductRow>,
        prefix: usize,
        cycles: usize,
        weights: &[AkitaField],
        eq: &GruenSplitEqPolynomial<AkitaField>,
    ) -> Result<Box<dyn ProductDeviceColumns<AkitaField>>, KernelError<AkitaField>> {
        let _span = tracing::info_span!("metal_spartan_product_prepare").entered();
        let stored = prefix.div_ceil(2) * 2;
        let tails = if prefix < cycles {
            factors(&rows.row(prefix)?, weights)
        } else {
            [AkitaField::zero(); 2]
        };
        let values = DeviceBuffer::zeroed(&self.0.device, field_len(stored).map_err(kernel_error)?)
            .map_err(kernel_error)?;
        let equality = Equality::new(&self.0.device, eq.e_in_current(), eq.e_out_current())
            .map_err(kernel_error)?;
        let need_one = eq.current_linear_evals().1.is_zero() && !eq.current_scalar().is_zero();
        let result = self.0.native(
            rows,
            NativeRequest {
                count: stored,
                mode: 1,
                outputs: 3,
                weights: &weights[..3],
                eq: &equality,
                values: &values,
                need_one,
            },
        )?;
        let tail = eq.suffix_weight(stored / 2) * tails[0] * tails[1];
        let pending = [
            result[0] + tail,
            result[1],
            result[2] + if need_one { tail } else { AkitaField::zero() },
        ];
        Ok(Box::new(DeviceProduct {
            gpu: Arc::clone(&self.0),
            values,
            prefix: stored,
            tails,
            pending: Some(pending),
        }))
    }
    fn claimed_inputs(
        &self,
        rows: &BundleAccess<'_, SpartanProductRow>,
        weights: &PrefixCycleWeights<AkitaField>,
    ) -> Result<Vec<AkitaField>, SumcheckKernelError<AkitaField>> {
        let _span = tracing::info_span!("metal_spartan_product_openings").entered();
        let run = || -> Result<Vec<AkitaField>, KernelError<AkitaField>> {
            let final_block = weights.block(weights.blocks() - 1);
            let (inner, outer) = weights.split_evals();
            let eq = Equality::new(&self.0.device, inner, outer).map_err(kernel_error)?;
            let unused =
                DeviceBuffer::<AkitaField>::zeroed(&self.0.device, 1).map_err(kernel_error)?;
            let mut result = self.0.native(
                rows,
                NativeRequest {
                    count: final_block.start,
                    mode: 2,
                    outputs: 8,
                    weights: &[AkitaField::zero()],
                    eq: &eq,
                    values: &unused,
                    need_one: false,
                },
            )?;
            let row = rows.row(final_block.start)?;
            let f = |value: bool| AkitaField::from_u64(u64::from(value));
            let values = [
                AkitaField::from_u64(row.left_instruction_input.0),
                AkitaField::from_i128(row.right_instruction_input.0),
                f(row.jump_flag.0),
                f(row.write_lookup_output_to_rd.0),
                AkitaField::from_u64(row.lookup_output.0),
                f(row.branch_flag.0),
                f(row.next_is_noop.0),
                f(row.virtual_instruction.0),
            ];
            let mass = final_block.inner[0] * final_block.outer;
            for (sum, value) in result.iter_mut().zip(values) {
                *sum += mass * value;
            }
            Ok(result)
        };
        run().map_err(|error| SumcheckKernelError::Compute {
            backend: "Metal Spartan product",
            reason: error.to_string(),
        })
    }
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct DeviceProduct {
    #[cfg_attr(feature = "allocative", allocative(skip))]
    gpu: Arc<ProductGpu>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    values: DeviceBuffer<AkitaField>,
    prefix: usize,
    tails: [AkitaField; 2],
    pending: Option<[AkitaField; 3]>,
}

impl ProductDeviceColumns<AkitaField> for DeviceProduct {
    fn bind(
        &mut self,
        challenge: AkitaField,
    ) -> Result<Option<[PrefixColumn<AkitaField>; 2]>, SumcheckError<AkitaField>> {
        let span = tracing::info_span!(
            "metal_spartan_product_bind",
            prefix = self.prefix,
            gpu_s = tracing::field::Empty
        );
        let _span = span.enter();
        let next_prefix = self.prefix.div_ceil(2);
        let output = DeviceBuffer::zeroed(
            &self.gpu.device,
            field_len(next_prefix).map_err(compute_error)?,
        )
        .map_err(compute_error)?;
        let prefix = word(self.prefix).map_err(compute_error)?;
        let mut batch = Batch::new(&self.gpu.device).map_err(compute_error)?;
        // SAFETY: one thread writes one pair into a separate output buffer;
        // missing input rows use the canonical constant tail.
        unsafe {
            batch
                .dispatch_unchecked(
                    self.gpu
                        .library
                        .pipeline(&host_name::<AkitaField>(BIND))
                        .map_err(compute_error)?,
                    &[
                        Binding::buffer(&self.values),
                        Binding::buffer(&output),
                        Binding::value(&self.tails),
                        Binding::value(&challenge),
                        Binding::value(&prefix),
                    ],
                    Grid::linear(next_prefix, THREADS),
                )
                .map_err(compute_error)?;
        }
        let gpu_time = batch.commit_and_wait().map_err(compute_error)?;
        let _ = span.record("gpu_s", gpu_time.as_secs_f64());
        self.values = output;
        self.prefix = next_prefix;
        self.pending = None;
        if next_prefix <= CPU_TAIL {
            let _span =
                tracing::info_span!("metal_spartan_product_restore", rows = next_prefix).entered();
            let values = self.values.read().map_err(compute_error)?;
            let columns = std::array::from_fn(|column| PrefixColumn {
                values: values.chunks_exact(2).map(|row| row[column]).collect(),
                tail: [self.tails[column]; 2],
            });
            return Ok(Some(columns));
        }
        Ok(None)
    }
    fn products(
        &mut self,
        eq: &GruenSplitEqPolynomial<AkitaField>,
    ) -> Result<[AkitaField; 3], SumcheckError<AkitaField>> {
        if let Some(pending) = self.pending.take() {
            return Ok(pending);
        }
        let span = tracing::info_span!(
            "metal_spartan_product_round",
            prefix = self.prefix,
            gpu_s = tracing::field::Empty
        );
        let _span = span.enter();
        let pairs = self.prefix.div_ceil(2);
        let equality = Equality::new(&self.gpu.device, eq.e_in_current(), eq.e_out_current())
            .map_err(compute_error)?;
        let groups = pairs.div_ceil(equality.inner_len);
        if groups > equality.outer_len {
            return Err(compute_error(invalid(
                "round equality does not cover prefix",
            )));
        }
        let mut partials =
            DeviceBuffer::zeroed(&self.gpu.device, groups * 3).map_err(compute_error)?;
        let need_one = eq.current_linear_evals().1.is_zero() && !eq.current_scalar().is_zero();
        let params = Params {
            first: 0,
            count: 0,
            in_len: word(equality.inner_len).map_err(compute_error)?,
            outputs: 3,
            mode: 0,
            prefix: word(self.prefix).map_err(compute_error)?,
            pairs: word(pairs).map_err(compute_error)?,
            need_one: u32::from(need_one),
        };
        let mut batch = Batch::new(&self.gpu.device).map_err(compute_error)?;
        // SAFETY: groups cover exactly the live pairs. Each input access checks
        // prefix and substitutes its constant tail; partial writes are disjoint.
        unsafe {
            batch
                .dispatch_unchecked(
                    self.gpu
                        .library
                        .pipeline(&host_name::<AkitaField>(ROUND))
                        .map_err(compute_error)?,
                    &[
                        Binding::buffer(&self.values),
                        Binding::buffer(&equality.inner),
                        Binding::buffer(&equality.outer),
                        Binding::buffer(&partials),
                        Binding::value(&self.tails),
                        Binding::value(&params),
                    ],
                    Grid::linear(groups * THREADS, THREADS),
                )
                .map_err(compute_error)?;
        }
        let gpu_time = batch.commit_and_wait().map_err(compute_error)?;
        let _ = span.record("gpu_s", gpu_time.as_secs_f64());
        let sums = sum_partials(&mut partials, 3).map_err(compute_error)?;
        let tail = eq.suffix_weight(pairs) * self.tails[0] * self.tails[1];
        Ok([
            sums[0] + tail,
            sums[1],
            sums[2] + if need_one { tail } else { AkitaField::zero() },
        ])
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests;
