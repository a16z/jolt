//! Bounded native input production and resident fused bind/message rounds.
mod source;
mod state;

use std::mem::MaybeUninit;
use std::sync::Arc;

use bytemuck_derive::{NoUninit, Pod, Zeroable};
use jolt_akita::AkitaField as F;
use jolt_field::{Ring, Zero};
use jolt_metal::runtime::{
    host_name, Batch, Binding, Device, DeviceBuffer, Grid, LibrarySpec, ShaderLibrary,
};
use jolt_metal::shaders::FIELD_HEADERS;
use jolt_metal::MetalError;

use super::*;
use state::ResidentInput;

const NATIVE: &str = "instruction_input_native";
const BIND_ROUND: &str = "instruction_input_bind_round";
const THREADS: usize = 256;
const TILE_ROWS: usize = 1 << 20;
const MIN_PREFIX: usize = 1 << 16;
const CPU_TAIL: usize = 1 << 12;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    prefix: u32,
    output_len: u32,
    first_pair: u32,
    pairs: u32,
    in_len: u32,
    old_bits: u32,
    new_bits: u32,
    need_one: u32,
}

#[repr(C)]
#[derive(Clone, Copy, NoUninit)]
struct Constants {
    tails: [F; NUM_TABLES],
    tail_word: [u32; 4],
    challenge: F,
    gamma: F,
}
const _: () = assert!(size_of::<Constants>() == 208);

struct Gpu {
    device: Device,
    library: ShaderLibrary,
    unused_words: DeviceBuffer<[u32; 4]>,
    unused_lut: DeviceBuffer<F>,
    min_prefix: usize,
    cpu_tail: usize,
    tile_rows: usize,
}

#[derive(Clone)]
pub(crate) struct MetalInstructionInput(Arc<Gpu>);

impl MetalInstructionInput {
    pub(crate) fn new() -> Result<Self, MetalError> {
        let device = Device::system_default()?;
        let mut spec = LibrarySpec::new();
        for (name, source) in FIELD_HEADERS {
            spec = spec.source(name, source);
        }
        spec = spec.source("instruction_input.metal", include_str!("input.metal"));
        for name in [NATIVE, BIND_ROUND] {
            spec = spec.instantiate::<F>(name);
        }
        let library = ShaderLibrary::compile(&device, &spec)?;
        for name in [NATIVE, BIND_ROUND] {
            let pipeline = library.pipeline(&host_name::<F>(name))?;
            if pipeline.thread_execution_width() != 32
                || pipeline.max_total_threads_per_threadgroup() < THREADS
                || pipeline.static_threadgroup_memory_length()
                    > device.limits().max_threadgroup_memory_length
            {
                return Err(invalid("input kernels exceed SIMD/threadgroup limits"));
            }
        }
        let unused_words = DeviceBuffer::zeroed(&device, 1)?;
        let unused_lut = DeviceBuffer::zeroed(&device, 1)?;
        Ok(Self(Arc::new(Gpu {
            device,
            library,
            unused_words,
            unused_lut,
            min_prefix: MIN_PREFIX,
            cpu_tail: CPU_TAIL,
            tile_rows: TILE_ROWS,
        })))
    }
}

impl PrepareKernel<F, InstructionInput<F>> for MetalInstructionInput {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, InstructionInput<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = InstructionInput<F>>>, KernelError<F>> {
        let point = inputs.relation.product_remainder_opening_point();
        let rows = BundleStore::resolve(witness, 1usize << point.len())?;
        let mut kernel =
            OptimizedInstructionInputKernel::new(point, rows, inputs.challenges.gamma)?;
        kernel.compute = Some(Arc::new(self.clone()));
        Ok(Box::new(kernel))
    }
}

fn invalid(reason: &str) -> MetalError {
    MetalError::InvalidDispatch {
        pipeline: "instruction input".into(),
        reason: reason.into(),
    }
}
fn compute_error(error: MetalError) -> SumcheckError<F> {
    SumcheckError::Compute {
        backend: "Metal instruction input",
        reason: error.to_string(),
    }
}
fn word(value: usize) -> Result<u32, MetalError> {
    u32::try_from(value).map_err(|_| invalid("input geometry exceeds u32"))
}
fn field_len(rows: usize, packed: bool) -> Result<usize, MetalError> {
    rows.checked_mul(if packed { 5 } else { NUM_TABLES })
        .ok_or_else(|| invalid("input field extent overflows"))
}
fn words(value: u128) -> [u32; 4] {
    std::array::from_fn(|i| (value >> (32 * i)) as u32)
}

struct Equality {
    inner: DeviceBuffer<F>,
    outer: DeviceBuffer<F>,
    inner_len: usize,
    outer_len: usize,
    need_one: u32,
}
impl Equality {
    fn new(device: &Device, eq: &GruenSplitEqPolynomial<F>) -> Result<Self, MetalError> {
        let inner = eq.e_in_current();
        let outer = eq.e_out_current();
        if inner.is_empty() || outer.is_empty() {
            return Err(invalid("empty input equality factor"));
        }
        Ok(Self {
            inner: DeviceBuffer::from_slice(device, inner)?,
            outer: DeviceBuffer::from_slice(device, outer)?,
            inner_len: inner.len(),
            outer_len: outer.len(),
            need_one: u32::from(
                eq.current_linear_evals().1.is_zero() && !eq.current_scalar().is_zero(),
            ),
        })
    }
    fn groups(&self, first_pair: usize, pairs: usize) -> Result<usize, MetalError> {
        let groups = pairs.div_ceil(self.inner_len);
        if !first_pair.is_multiple_of(self.inner_len)
            || first_pair / self.inner_len + groups > self.outer_len
        {
            return Err(invalid("input tile does not align with equality blocks"));
        }
        Ok(groups)
    }
}

/// Every logical element is written by exactly one shader invocation before
/// `complete`; no zero-filled multi-gigabyte output allocation is needed.
struct Pending {
    values: DeviceBuffer<MaybeUninit<F>>,
    words: Option<DeviceBuffer<MaybeUninit<[u32; 4]>>>,
}
impl Pending {
    fn new(gpu: &Gpu, rows: usize, packed: bool) -> Result<Self, MetalError> {
        Ok(Self {
            values: DeviceBuffer::uninit(&gpu.device, field_len(rows, packed)?)?,
            words: if packed {
                Some(DeviceBuffer::uninit(&gpu.device, rows)?)
            } else {
                None
            },
        })
    }
    fn words_binding<'a>(&'a self, gpu: &'a Gpu) -> Binding<'a> {
        match &self.words {
            Some(words) => Binding::buffer(words),
            None => Binding::buffer(&gpu.unused_words),
        }
    }
    /// # Safety
    /// All output rows must have been written by successful batches.
    unsafe fn complete(self) -> Result<Columns, MetalError> {
        // SAFETY: forwarded initialization contract; runtime also checks that
        // every buffer's submitted GPU work completed.
        let values = unsafe { self.values.assume_init()? };
        let words = match self.words {
            // SAFETY: same complete-write contract, including unused word bits.
            Some(words) => Some(unsafe { words.assume_init()? }),
            None => None,
        };
        Ok(Columns { values, words })
    }
}
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct Columns {
    #[cfg_attr(feature = "allocative", allocative(skip))]
    values: DeviceBuffer<F>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    words: Option<DeviceBuffer<[u32; 4]>>,
}

fn partials(groups: usize, device: &Device) -> Result<DeviceBuffer<MaybeUninit<F>>, MetalError> {
    DeviceBuffer::uninit(
        device,
        groups
            .checked_mul(3)
            .ok_or_else(|| invalid("input partial extent overflows"))?,
    )
}
/// # Safety
/// Each of the three partial lanes per group has been written and its
/// batch has completed, including the zero endpoint lane when not requested.
unsafe fn sum_partials(partials: DeviceBuffer<MaybeUninit<F>>) -> Result<[F; 3], MetalError> {
    // SAFETY: forwarded full-write contract; completion is checked by runtime.
    let mut partials = unsafe { partials.assume_init()? };
    let mut sum = [F::zero(); 3];
    for row in partials.read()?.chunks_exact(3) {
        for (sum, value) in sum.iter_mut().zip(row) {
            *sum += *value;
        }
    }
    Ok(sum)
}

impl InputCompute<F> for MetalInstructionInput {
    fn admits(&self, prefix: usize, log_t: usize) -> bool {
        let half = prefix.div_ceil(2);
        let limits = self.0.device.limits();
        let bytes = half as u128 * 96;
        prefix != 0
            && prefix >= self.0.min_prefix
            && prefix < u32::MAX as usize
            && (6..=32).contains(&log_t)
            && half as u128 * 80 <= limits.max_buffer_length as u128
            && bytes * 3 / 2 + (self.0.tile_rows as u128 * 49) + (1 << 22)
                <= u128::from(limits.recommended_max_working_set_size)
    }
    fn materialize(
        &self,
        rows: &BundleStore<InstructionInputRow>,
        prefix: usize,
        challenge: F,
        tails: [F; NUM_TABLES],
        eq: &GruenSplitEqPolynomial<F>,
        gamma: F,
    ) -> Result<InputPrepared<F>, SumcheckError<F>> {
        source::materialize(
            Arc::clone(&self.0),
            rows,
            prefix,
            challenge,
            tails,
            eq,
            gamma,
        )
    }
}

#[cfg(test)]
mod tests;
