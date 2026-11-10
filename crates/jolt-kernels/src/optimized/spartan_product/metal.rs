//! Resident stage-2 products and shared native-operand opening reduction.
use super::super::instruction_claim_reduction::metal::MetalInstructionClaimReduction;
use super::super::instruction_input::metal::MetalInstructionInput;
use super::super::native_operands::{
    ForClaimReduction, ForInput, InputTerms, OpeningClaims, Stage2Source, SOURCE_HEADER,
};
use std::mem::MaybeUninit;

use bytemuck_derive::{NoUninit, Pod, Zeroable};
use jolt_akita::AkitaField;
use jolt_field::{Ring, Zero};
use jolt_metal::runtime::{
    host_name, Batch, Binding, Device, DeviceBuffer, Grid, LibrarySpec, ShaderLibrary,
};
use jolt_metal::shaders::FIELD_HEADERS;
use jolt_metal::MetalError;

use super::*;
use jolt_verifier::stages::stage2::instruction_claim_reduction::InstructionClaimReduction;

const MIN_PREFIX: usize = 1 << 16;
const CPU_TAIL: usize = 1 << 12;
const THREADS: usize = 256;
const NATIVE: &str = "spartan_product_native";
const BIND: &str = "spartan_product_bind";
const ROUND: &str = "spartan_product_round";
const OPENINGS: &str = "spartan_operands_openings";

#[repr(C)]
#[derive(Clone, Copy, NoUninit)]
struct ProductTails {
    left: AkitaField,
    right: AkitaField,
}
const _: () = assert!(size_of::<ProductTails>() == 32);
impl From<[AkitaField; 2]> for ProductTails {
    fn from([left, right]: [AkitaField; 2]) -> Self {
        Self { left, right }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct ProductParams {
    pub(crate) first: u32,
    pub(crate) count: u32,
    pub(crate) in_len: u32,
    pub(crate) outputs: u32,
    pub(crate) mode: u32,
    pub(crate) prefix: u32,
    pub(crate) pairs: u32,
    pub(crate) need_one: u32,
}

pub(crate) struct ProductGpu {
    pub(crate) device: Arc<Device>,
    pub(crate) library: ShaderLibrary,
    pub(crate) terms: InputTerms,
}

#[derive(Clone)]
pub(crate) struct MetalProductUniskip(Arc<ProductGpu>);

impl MetalProductUniskip {
    pub(crate) fn claim_reduction(&self) -> MetalInstructionClaimReduction {
        MetalInstructionClaimReduction(Arc::clone(&self.0))
    }
    pub(crate) fn input(&self) -> Result<MetalInstructionInput, MetalError> {
        MetalInstructionInput::with_device(Arc::clone(&self.0.device))
    }
    pub(crate) fn new() -> Result<Self, MetalError> {
        let device = Arc::new(Device::system_default()?);
        let terms = InputTerms::lower()?;
        let aliases = InstructionClaimReduction::<AkitaField>::aliased_output_openings();
        let sources = [
            lookup_output_product(),
            left_instruction_input_product(),
            right_instruction_input_product(),
        ];
        if aliases.len() != sources.len()
            || sources.iter().any(|source| {
                !aliases.iter().any(|(alias, target)| {
                    target == source && alias.polynomial_id() == source.polynomial_id()
                })
            })
        {
            return Err(invalid("native operand claim aliases changed"));
        }
        let mut spec = LibrarySpec::new();
        for (name, source) in FIELD_HEADERS {
            spec = spec.source(name, source);
        }
        spec = spec
            .source("native_operands.metal", SOURCE_HEADER)
            .source("spartan_product.metal", include_str!("product.metal"));
        for name in [NATIVE, BIND, ROUND, OPENINGS, "spartan_operand_combined"] {
            spec = spec.instantiate::<AkitaField>(name);
        }
        let library = ShaderLibrary::compile(&device, &spec)?;
        for name in [NATIVE, BIND, ROUND, OPENINGS, "spartan_operand_combined"] {
            let pipeline = library.pipeline(&host_name::<AkitaField>(name))?;
            if pipeline.thread_execution_width() != 32
                || pipeline.max_total_threads_per_threadgroup() < THREADS
                || pipeline.static_threadgroup_memory_length()
                    > device.limits().max_threadgroup_memory_length
            {
                return Err(invalid("product requires SIMD32 and 256 threads"));
            }
        }
        Ok(Self(Arc::new(ProductGpu {
            device,
            library,
            terms,
        })))
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
        #[cfg(feature = "field-inline")]
        if witness.field_inline().is_some() {
            return OptimizedProductUniskip.prepare(session, log_t, tau_low, witness);
        }
        if !(6..=31).contains(&log_t) {
            return OptimizedProductUniskip.prepare(session, log_t, tau_low, witness);
        }
        let Some(identity) = witness.random_access() else {
            return OptimizedProductUniskip.prepare(session, log_t, tau_low, witness);
        };
        let Some(source) = Stage2Source::capture(&self.0.device, identity, 1 << log_t, MIN_PREFIX)?
        else {
            return OptimizedProductUniskip.prepare(session, log_t, tau_low, witness);
        };
        let compute = Arc::new(ResidentProduct {
            gpu: Arc::clone(&self.0),
            source: Arc::clone(&source),
        });
        OptimizedProductUniskip::prepare_with_compute(
            session,
            log_t,
            tau_low,
            witness,
            Some(compute),
        )?;
        session.park(ForInput(Arc::clone(&source.core)));
        session.park(ForClaimReduction(source));
        Ok(())
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

pub(crate) struct ResidentProduct {
    pub(crate) gpu: Arc<ProductGpu>,
    pub(crate) source: Arc<Stage2Source>,
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
    values: &'a DeviceBuffer<MaybeUninit<AkitaField>>,
    need_one: bool,
}

impl ProductGpu {
    /// Each threadgroup reduces one split-equality block from the resident source.
    fn native(
        &self,
        source: &Stage2Source,
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
        if count.div_ceil(block_rows) > eq.outer_len || (mode == 1 && count % 2 != 0) {
            return Err(kernel_error(invalid(
                "native source equality does not cover rows",
            )));
        }
        let core = source
            .core
            .lock()
            .map_err(|_| kernel_error(invalid("native source lock poisoned")))?;
        if count > core.buffer.len() {
            return Err(kernel_error(invalid(
                "native source prefix exceeds allocation",
            )));
        }
        let weights = DeviceBuffer::from_slice(&self.device, weights).map_err(kernel_error)?;
        let groups = count.div_ceil(block_rows);
        let partials =
            DeviceBuffer::<MaybeUninit<AkitaField>>::uninit(&self.device, groups * outputs)
                .map_err(kernel_error)?;
        let params = ProductParams {
            first: 0,
            count: word(count).map_err(kernel_error)?,
            in_len: word(eq.inner_len).map_err(kernel_error)?,
            outputs: word(outputs).map_err(kernel_error)?,
            mode,
            prefix: 0,
            pairs: 0,
            need_one: u32::from(need_one),
        };
        let mut batch = Batch::new(&self.device).map_err(kernel_error)?;
        // SAFETY: the checked resident prefix and equality blocks cover reads;
        // mode1 writes disjoint field rows, every group writes all partials.
        unsafe {
            batch
                .dispatch_unchecked(
                    self.library
                        .pipeline(&host_name::<AkitaField>(NATIVE))
                        .map_err(kernel_error)?,
                    &[
                        Binding::buffer(&core.buffer),
                        Binding::buffer(&weights),
                        Binding::buffer(&eq.inner),
                        Binding::buffer(&eq.outer),
                        Binding::buffer(values),
                        Binding::buffer(&partials),
                        Binding::value(&params),
                        Binding::value(&self.terms),
                    ],
                    Grid::linear(groups * THREADS, THREADS),
                )
                .map_err(kernel_error)?;
        }
        let elapsed = batch.commit_and_wait().map_err(kernel_error)?;
        let _ = span.record("gpu_s", elapsed.as_secs_f64());
        // SAFETY: all dispatched groups completed their partial writes.
        let mut partials = unsafe { partials.assume_init() }.map_err(kernel_error)?;
        let totals = sum_partials(&mut partials, outputs).map_err(kernel_error)?;
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

impl ProductCompute<AkitaField> for ResidentProduct {
    fn admits(&self, prefix: usize, log_t: usize) -> bool {
        prefix > 0 && log_t <= 31 && EXTENSION_SIZE <= 8
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
        let eq = Equality::new(&self.gpu.device, &inner, &outer).map_err(kernel_error)?;
        let weights: Vec<_> = coefficients
            .iter()
            .flat_map(|row| row[..3].iter().copied().map(AkitaField::from_i64))
            .collect();
        let unused = DeviceBuffer::<MaybeUninit<AkitaField>>::uninit(&self.gpu.device, 1)
            .map_err(kernel_error)?;
        let mut totals = self.gpu.native(
            &self.source,
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
        let values = DeviceBuffer::<MaybeUninit<AkitaField>>::uninit(
            &self.gpu.device,
            field_len(stored).map_err(kernel_error)?,
        )
        .map_err(kernel_error)?;
        let equality = Equality::new(&self.gpu.device, eq.e_in_current(), eq.e_out_current())
            .map_err(kernel_error)?;
        let need_one = eq.current_linear_evals().1.is_zero() && !eq.current_scalar().is_zero();
        let result = self.gpu.native(
            &self.source,
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
            gpu: Arc::clone(&self.gpu),
            // SAFETY: native mode1 completed writes to both fields of every stored row.
            values: unsafe { values.assume_init() }.map_err(kernel_error)?,
            prefix: stored,
            tails,
            pending: Some(pending),
        }))
    }
    fn finish_openings(&self, point: &[AkitaField]) -> Result<(), SumcheckError<AkitaField>> {
        self.openings(point).map(|_| ()).map_err(compute_error)
    }
    fn claimed_inputs(
        &self,
        _rows: &BundleAccess<'_, SpartanProductRow>,
        _weights: &PrefixCycleWeights<AkitaField>,
        point: &[AkitaField],
    ) -> Result<Vec<AkitaField>, SumcheckKernelError<AkitaField>> {
        self.cached_openings(point)
            .map(|values| values[..8].to_vec())
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
        let output = DeviceBuffer::<MaybeUninit<AkitaField>>::uninit(
            &self.gpu.device,
            field_len(next_prefix).map_err(compute_error)?,
        )
        .map_err(compute_error)?;
        let prefix = word(self.prefix).map_err(compute_error)?;
        let tails = ProductTails::from(self.tails);
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
                        Binding::value(&tails),
                        Binding::value(&challenge),
                        Binding::value(&prefix),
                    ],
                    Grid::linear(next_prefix, THREADS),
                )
                .map_err(compute_error)?;
        }
        let gpu_time = batch.commit_and_wait().map_err(compute_error)?;
        let _ = span.record("gpu_s", gpu_time.as_secs_f64());
        // SAFETY: the completed bind wrote both fields for every output row.
        self.values = unsafe { output.assume_init() }.map_err(compute_error)?;
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
        let params = ProductParams {
            first: 0,
            count: 0,
            in_len: word(equality.inner_len).map_err(compute_error)?,
            outputs: 3,
            mode: 0,
            prefix: word(self.prefix).map_err(compute_error)?,
            pairs: word(pairs).map_err(compute_error)?,
            need_one: u32::from(need_one),
        };
        let tails = ProductTails::from(self.tails);
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
                        Binding::value(&tails),
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

impl ResidentProduct {
    /// The typed extraction seam reads values finalized by `finish_rounds`;
    /// all fallible GPU work stays in the compute error boundary.
    pub(crate) fn cached_openings(
        &self,
        point: &[AkitaField],
    ) -> Result<[AkitaField; 10], SumcheckKernelError<AkitaField>> {
        let cache =
            self.source
                .claims
                .lock()
                .map_err(|_| SumcheckKernelError::InvariantViolation {
                    reason: "native operand claim cache poisoned",
                })?;
        let claims = cache
            .as_ref()
            .ok_or(SumcheckKernelError::InvariantViolation {
                reason: "native operand claims were not finalized",
            })?;
        if claims.point != point {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "aliased native operand opening points differ",
            });
        }
        Ok(claims.values)
    }

    /// Product and ICR share only verifier-declared aliases at the exact same point.
    pub(crate) fn openings(&self, point: &[AkitaField]) -> Result<[AkitaField; 10], MetalError> {
        let mut cache = self
            .source
            .claims
            .lock()
            .map_err(|_| invalid("operand claim cache poisoned"))?;
        if let Some(claims) = cache.as_ref() {
            if claims.point != point {
                return Err(invalid("aliased operand opening points differ"));
            }
            return Ok(claims.values);
        }
        let span = tracing::info_span!(
            "metal_native_operand_openings",
            gpu_s = tracing::field::Empty
        );
        let _span = span.enter();
        let core = self
            .source
            .core
            .lock()
            .map_err(|_| invalid("native core lock poisoned"))?;
        if point.len() != core.cycles.ilog2() as usize {
            return Err(invalid("operand opening point has wrong dimension"));
        }
        let weights = PrefixCycleWeights::new(point, core.prefix);
        let final_block = weights.block(weights.blocks() - 1);
        let (inner, outer) = weights.split_evals();
        let eq = Equality::new(&self.gpu.device, inner, outer)?;
        let count = final_block.start;
        let groups = count.div_ceil(eq.inner_len);
        let partials =
            DeviceBuffer::<MaybeUninit<AkitaField>>::uninit(&self.gpu.device, groups * 10)?;
        let lookup = self
            .source
            .lookup
            .lock()
            .map_err(|_| invalid("native lookup lock poisoned"))?;
        let params = ProductParams {
            first: 0,
            count: word(count)?,
            in_len: word(eq.inner_len)?,
            outputs: 10,
            mode: 0,
            prefix: 0,
            pairs: 0,
            need_one: 0,
        };
        let mut batch = Batch::new(&self.gpu.device)?;
        // SAFETY: canonical prefix weights and the complete shared source cover
        // all reads; all groups write ten disjoint partials after uniform reductions.
        unsafe {
            batch.dispatch_unchecked(
                self.gpu
                    .library
                    .pipeline(&host_name::<AkitaField>(OPENINGS))?,
                &[
                    Binding::buffer(&core.buffer),
                    Binding::buffer(&*lookup),
                    Binding::buffer(&eq.inner),
                    Binding::buffer(&eq.outer),
                    Binding::buffer(&partials),
                    Binding::value(&params),
                    Binding::value(&self.gpu.terms),
                ],
                Grid::linear(groups * THREADS, THREADS),
            )?;
        }
        let gpu_s = batch.commit_and_wait()?.as_secs_f64();
        let _ = span.record("gpu_s", gpu_s);
        // SAFETY: completed groups wrote every scalar partial.
        let mut partials = unsafe { partials.assume_init() }?;
        let sums = sum_partials(&mut partials, 10)?;
        let [left, right] = self.gpu.terms.evaluate(&core.tail);
        let row = core.tail;
        let tail = [
            left,
            right,
            row.product_flag(0),
            row.product_flag(1),
            AkitaField::from_u64(row.lookup),
            row.product_flag(2),
            row.product_flag(3),
            row.product_flag(4),
            AkitaField::from_u64(self.source.tail_lookup.left),
            self.source.tail_lookup.right(),
        ];
        let mass = final_block.inner[0] * final_block.outer;
        let values = std::array::from_fn(|i| sums[i] + mass * tail[i]);
        *cache = Some(OpeningClaims {
            point: point.to_vec(),
            values,
        });
        Ok(values)
    }
}

#[cfg(all(test, not(feature = "field-inline")))]
mod tests;
