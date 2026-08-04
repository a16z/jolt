//! Canonical 128-bit Solinas-field arithmetic on Metal.
//!
//! [`Fp128`] is the buffer ABI, not a host field implementation. Arithmetic is
//! performed by the shader specialized for `2^128 - C`; host callers supply
//! canonical values for the selected offset.

use std::{cell::Cell, ffi::c_void, slice, time::Duration};

use jolt_field::FixedBytes;
use metal::{
    objc::{rc::autoreleasepool, runtime::Sel, Message},
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, Library,
    MTLCommandBufferStatus, MTLResourceOptions, MTLSize,
};
use thiserror::Error;

const FIELD_SOURCE: &str = include_str!("fp128.metal");
const ADDRESS_RAF_SOURCE: &str = include_str!("address_raf.metal");
const PROBE_SOURCE: &str = include_str!("probes.metal");
const PRODUCT5_SOURCE: &str = include_str!("product5.metal");

mod address_raf;
mod product5;

pub use address_raf::{
    AddressRafScanConfig, AddressRafScanInvocation, AddressRafScanRow, AddressRafSums,
    ADDRESS_RAF_BINS, ADDRESS_RAF_LANES,
};
pub use product5::{
    Product5Config, Product5Invocation, Product5Sequence, Product5SequenceConfig, PRODUCT5_FACTORS,
};

pub const OFFSET_275: u32 = 275;
pub const AKITA_OFFSET_FFFFA7F7: u32 = 0xffff_a7f7;

/// Little-endian limbs shared by Rust and Metal buffers.
///
/// Dispatch validates canonicality for the selected Solinas offset.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Fp128 {
    limbs: [u32; 4],
}

impl Fp128 {
    pub const ZERO: Self = Self::from_u128(0);
    pub const ONE: Self = Self::from_u128(1);

    pub const fn from_limbs(limbs: [u32; 4]) -> Self {
        Self { limbs }
    }

    pub const fn from_u128(value: u128) -> Self {
        Self {
            limbs: [
                value as u32,
                (value >> 32) as u32,
                (value >> 64) as u32,
                (value >> 96) as u32,
            ],
        }
    }

    pub const fn limbs(self) -> [u32; 4] {
        self.limbs
    }

    pub const fn to_u128(self) -> u128 {
        (self.limbs[0] as u128)
            | ((self.limbs[1] as u128) << 32)
            | ((self.limbs[2] as u128) << 64)
            | ((self.limbs[3] as u128) << 96)
    }

    pub const fn is_canonical(self, offset: u32) -> bool {
        offset != 0 && self.to_u128() <= u128::MAX - offset as u128
    }

    pub fn from_jolt_field<F: FixedBytes<16>>(value: &F) -> Self {
        Self::from_u128(u128::from_le_bytes(value.to_bytes_array()))
    }

    pub fn into_jolt_field<F: FixedBytes<16>>(self) -> F {
        F::from_bytes_array(&self.to_u128().to_le_bytes())
    }
}

/// A compiled entry point used to characterize one part of the field pipeline.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Probe {
    Noop,
    Copy,
    Add,
    Sub,
    MulWide,
    ChainWide1,
    ChainWide2,
    ChainWide4,
    ChainWide8,
    U32MadIlp8,
}

impl Probe {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Noop => "solinas_noop",
            Self::Copy => "solinas_copy",
            Self::Add => "solinas_add_probe",
            Self::Sub => "solinas_sub_probe",
            Self::MulWide => "solinas_mul_wide_probe",
            Self::ChainWide1 => "solinas_chain_wide_1",
            Self::ChainWide2 => "solinas_chain_wide_2",
            Self::ChainWide4 => "solinas_chain_wide_4",
            Self::ChainWide8 => "solinas_chain_wide_8",
            Self::U32MadIlp8 => "solinas_u32_mad_ilp8",
        }
    }

    pub const fn independent_chains(self) -> usize {
        match self {
            Self::ChainWide2 => 2,
            Self::ChainWide4 => 4,
            Self::ChainWide8 => 8,
            _ => 1,
        }
    }

    const fn accepts_noncanonical_output(self) -> bool {
        matches!(self, Self::U32MadIlp8)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DispatchConfig {
    pub iterations: u32,
    pub threads_per_threadgroup: Option<usize>,
}

impl Default for DispatchConfig {
    fn default() -> Self {
        Self {
            iterations: 1,
            threads_per_threadgroup: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PipelineLimits {
    pub thread_execution_width: usize,
    pub max_total_threads_per_threadgroup: usize,
    pub static_threadgroup_memory_length: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceInfo {
    pub name: String,
    pub max_buffer_length: u64,
    pub max_threadgroup_memory_length: u64,
    pub offset: u32,
}

#[derive(Debug, Error)]
pub enum MetalError {
    #[error("no Metal device is available")]
    DeviceUnavailable,
    #[error("Solinas offset must be nonzero")]
    InvalidOffset,
    #[error("failed to compile the Solinas Metal library: {0}")]
    LibraryCompilation(String),
    #[error("Metal entry point `{name}` was not found: {message}")]
    FunctionLookup { name: &'static str, message: String },
    #[error("failed to compile Metal entry point `{name}`: {message}")]
    PipelineCompilation { name: &'static str, message: String },
    #[error("a non-noop dispatch requires at least one element")]
    EmptyInput,
    #[error("use `prepare_noop` for the no-op probe")]
    NoopPreparation,
    #[error("input lengths differ: lhs={lhs}, rhs={rhs}")]
    LengthMismatch { lhs: usize, rhs: usize },
    #[error("input length {0} exceeds the shader's 32-bit element count")]
    InputTooLong(usize),
    #[error("buffer requires {requested} bytes but the Metal device limit is {maximum}")]
    BufferTooLong { requested: u64, maximum: u64 },
    #[error("input {side}[{index}] is not canonical for 2^128 - {offset}")]
    NonCanonicalInput {
        side: &'static str,
        index: usize,
        offset: u32,
    },
    #[error("output[{index}] is not canonical for 2^128 - {offset}")]
    NonCanonicalOutput { index: usize, offset: u32 },
    #[error("{probe} requires an element count divisible by its ILP ({ilp})")]
    MisalignedElementCount { probe: &'static str, ilp: usize },
    #[error("iteration count must be nonzero")]
    ZeroIterations,
    #[error("address RAF row and weight lengths differ: rows={rows}, weights={weights}")]
    AddressRafLengthMismatch { rows: usize, weights: usize },
    #[error("address RAF suffix length must be a multiple of eight in 0..=120, got {0}")]
    InvalidAddressRafSuffixLength(u32),
    #[error("address RAF rows per threadgroup must be nonzero, got {0}")]
    InvalidAddressRafRowsPerThreadgroup(usize),
    #[error(
        "address RAF pipeline `{pipeline}` requires SIMD width {expected}, but the device reports {got}"
    )]
    UnsupportedAddressRafExecutionWidth {
        pipeline: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("hybrid cutoff must be a power of two of at least two, got {0}")]
    InvalidHybridCutoff(usize),
    #[error(
        "five-factor kernels require a power-of-two table length of at least {minimum}, got {got}"
    )]
    InvalidProduct5TableLength { minimum: usize, got: usize },
    #[error("five-factor table storage has length {got}, expected {expected}")]
    Product5StorageLength { expected: usize, got: usize },
    #[error(
        "split equality tables cover {covered} pairs, but the five-factor kernel needs {expected}"
    )]
    Product5WeightShape { expected: usize, covered: usize },
    #[error(
        "five-factor pipeline `{pipeline}` requires SIMD width {expected}, but the device reports {got}"
    )]
    UnsupportedProduct5ExecutionWidth {
        pipeline: &'static str,
        expected: usize,
        got: usize,
    },
    #[error(
        "threadgroup width {requested} must be a multiple of {execution_width} and at most {maximum}"
    )]
    InvalidThreadgroupWidth {
        requested: usize,
        execution_width: usize,
        maximum: usize,
    },
    #[error("Metal command buffer finished with status {0:?}")]
    CommandFailed(MTLCommandBufferStatus),
    #[error("failed to read Metal command-buffer timestamp `{name}`: {message}")]
    GpuTimestampLookup { name: &'static str, message: String },
    #[error("Metal returned invalid GPU timestamps: start={start}, end={end}")]
    InvalidGpuTimestamps { start: f64, end: f64 },
    #[error("execute the invocation before reading its output")]
    NotExecuted,
}

#[derive(Clone)]
pub struct SolinasMetal {
    device: Device,
    queue: CommandQueue,
    library: Library,
    offset: u32,
}

impl SolinasMetal {
    pub fn for_akita() -> Result<Self, MetalError> {
        Self::new(AKITA_OFFSET_FFFFA7F7)
    }

    pub fn for_offset_275() -> Result<Self, MetalError> {
        Self::new(OFFSET_275)
    }

    pub fn new(offset: u32) -> Result<Self, MetalError> {
        if offset == 0 {
            return Err(MetalError::InvalidOffset);
        }
        let device = Device::system_default().ok_or(MetalError::DeviceUnavailable)?;
        let options = CompileOptions::new();
        let source = format!(
            "#define SOLINAS_OFFSET {offset}u\n{FIELD_SOURCE}\n{ADDRESS_RAF_SOURCE}\n{PROBE_SOURCE}\n{PRODUCT5_SOURCE}"
        );
        let library = device
            .new_library_with_source(&source, &options)
            .map_err(MetalError::LibraryCompilation)?;
        let queue = device.new_command_queue();

        Ok(Self {
            device,
            queue,
            library,
            offset,
        })
    }

    pub fn device_info(&self) -> DeviceInfo {
        DeviceInfo {
            name: self.device.name().to_owned(),
            max_buffer_length: self.device.max_buffer_length(),
            max_threadgroup_memory_length: self.device.max_threadgroup_memory_length(),
            offset: self.offset,
        }
    }

    pub fn pipeline_limits(&self, probe: Probe) -> Result<PipelineLimits, MetalError> {
        let pipeline = self.compile_pipeline(probe)?;
        Ok(Self::limits(&pipeline))
    }

    pub fn prepare_noop(&self) -> Result<Invocation<'_>, MetalError> {
        let pipeline = self.compile_pipeline(Probe::Noop)?;
        let limits = Self::limits(&pipeline);
        let threads_per_threadgroup =
            Self::resolve_threadgroup_width(Some(limits.thread_execution_width), limits)?;

        Ok(Invocation {
            context: self,
            probe: Probe::Noop,
            pipeline,
            buffers: None,
            limits,
            threads_per_threadgroup,
            grid_threads: 1,
            elements: 0,
            iterations: 1,
            completed: Cell::new(false),
        })
    }

    pub fn prepare(
        &self,
        probe: Probe,
        lhs: &[Fp128],
        rhs: &[Fp128],
        config: DispatchConfig,
    ) -> Result<Invocation<'_>, MetalError> {
        if probe == Probe::Noop {
            return Err(MetalError::NoopPreparation);
        }
        if lhs.is_empty() {
            return Err(MetalError::EmptyInput);
        }
        if lhs.len() != rhs.len() {
            return Err(MetalError::LengthMismatch {
                lhs: lhs.len(),
                rhs: rhs.len(),
            });
        }
        if config.iterations == 0 {
            return Err(MetalError::ZeroIterations);
        }
        let elements = u32::try_from(lhs.len()).map_err(|_| MetalError::InputTooLong(lhs.len()))?;
        let buffer_bytes =
            u64::try_from(size_of_val(lhs)).map_err(|_| MetalError::InputTooLong(lhs.len()))?;
        let max_buffer_length = self.device.max_buffer_length();
        if buffer_bytes > max_buffer_length {
            return Err(MetalError::BufferTooLong {
                requested: buffer_bytes,
                maximum: max_buffer_length,
            });
        }
        self.validate_inputs("lhs", lhs)?;
        self.validate_inputs("rhs", rhs)?;

        let ilp = probe.independent_chains();
        if !lhs.len().is_multiple_of(ilp) {
            return Err(MetalError::MisalignedElementCount {
                probe: probe.name(),
                ilp,
            });
        }

        let pipeline = self.compile_pipeline(probe)?;
        let limits = Self::limits(&pipeline);
        let threads_per_threadgroup =
            Self::resolve_threadgroup_width(config.threads_per_threadgroup, limits)?;
        let grid_threads = lhs.len() / ilp;
        let params = ProbeParams {
            elements,
            iterations: config.iterations,
        };
        let buffers = Buffers {
            lhs: buffer_from_slice(&self.device, lhs),
            rhs: buffer_from_slice(&self.device, rhs),
            output: self
                .device
                .new_buffer(buffer_bytes, MTLResourceOptions::StorageModeShared),
            params: buffer_from_slice(&self.device, slice::from_ref(&params)),
        };

        Ok(Invocation {
            context: self,
            probe,
            pipeline,
            buffers: Some(buffers),
            limits,
            threads_per_threadgroup,
            grid_threads,
            elements: lhs.len(),
            iterations: config.iterations,
            completed: Cell::new(false),
        })
    }

    fn compile_pipeline(&self, probe: Probe) -> Result<ComputePipelineState, MetalError> {
        self.compile_named_pipeline(probe.name())
    }

    fn compile_named_pipeline(
        &self,
        name: &'static str,
    ) -> Result<ComputePipelineState, MetalError> {
        let function = self
            .library
            .get_function(name, None)
            .map_err(|message| MetalError::FunctionLookup { name, message })?;
        self.device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|message| MetalError::PipelineCompilation { name, message })
    }

    fn validate_inputs(&self, side: &'static str, values: &[Fp128]) -> Result<(), MetalError> {
        if let Some((index, _)) = values
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_canonical(self.offset))
        {
            return Err(MetalError::NonCanonicalInput {
                side,
                index,
                offset: self.offset,
            });
        }
        Ok(())
    }

    fn limits(pipeline: &ComputePipelineState) -> PipelineLimits {
        PipelineLimits {
            thread_execution_width: pipeline.thread_execution_width() as usize,
            max_total_threads_per_threadgroup: pipeline.max_total_threads_per_threadgroup()
                as usize,
            static_threadgroup_memory_length: pipeline.static_threadgroup_memory_length(),
        }
    }

    fn resolve_threadgroup_width(
        requested: Option<usize>,
        limits: PipelineLimits,
    ) -> Result<usize, MetalError> {
        let execution_width = limits.thread_execution_width;
        let maximum = limits.max_total_threads_per_threadgroup;
        let default = (execution_width * 8).min(maximum);
        let width = requested.unwrap_or(default);
        if width == 0 || width > maximum || !width.is_multiple_of(execution_width) {
            return Err(MetalError::InvalidThreadgroupWidth {
                requested: width,
                execution_width,
                maximum,
            });
        }
        Ok(width)
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ProbeParams {
    elements: u32,
    iterations: u32,
}

struct Buffers {
    lhs: Buffer,
    rhs: Buffer,
    output: Buffer,
    params: Buffer,
}

pub struct Invocation<'a> {
    context: &'a SolinasMetal,
    probe: Probe,
    pipeline: ComputePipelineState,
    buffers: Option<Buffers>,
    limits: PipelineLimits,
    threads_per_threadgroup: usize,
    grid_threads: usize,
    elements: usize,
    iterations: u32,
    completed: Cell<bool>,
}

impl Invocation<'_> {
    pub const fn probe(&self) -> Probe {
        self.probe
    }

    pub const fn pipeline_limits(&self) -> PipelineLimits {
        self.limits
    }

    pub const fn threads_per_threadgroup(&self) -> usize {
        self.threads_per_threadgroup
    }

    pub const fn grid_threads(&self) -> usize {
        self.grid_threads
    }

    pub const fn iterations(&self) -> u32 {
        self.iterations
    }

    pub const fn field_operation_count(&self) -> u64 {
        match self.probe {
            Probe::Add | Probe::Sub | Probe::MulWide => self.elements as u64,
            Probe::ChainWide1 | Probe::ChainWide2 | Probe::ChainWide4 | Probe::ChainWide8 => {
                self.elements as u64 * self.iterations as u64
            }
            _ => 0,
        }
    }

    pub const fn logical_bytes(&self) -> u64 {
        let bytes_per_element = match self.probe {
            Probe::Copy => 32,
            Probe::Add
            | Probe::Sub
            | Probe::MulWide
            | Probe::ChainWide1
            | Probe::ChainWide2
            | Probe::ChainWide4
            | Probe::ChainWide8
            | Probe::U32MadIlp8 => 48,
            Probe::Noop => 0,
        };
        self.elements as u64 * bytes_per_element
    }

    pub fn execute(&self) -> Result<(), MetalError> {
        self.execute_timed().map(|_| ())
    }

    /// Executes the command and returns time spent running on the GPU.
    pub fn execute_timed(&self) -> Result<Duration, MetalError> {
        autoreleasepool(|| {
            let command_buffer = self.context.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.pipeline);
            if let Some(buffers) = &self.buffers {
                encoder.set_buffer(0, Some(&buffers.lhs), 0);
                encoder.set_buffer(1, Some(&buffers.rhs), 0);
                encoder.set_buffer(2, Some(&buffers.output), 0);
                encoder.set_buffer(3, Some(&buffers.params), 0);
            }
            let threads_per_threadgroup = MTLSize {
                width: self.threads_per_threadgroup as u64,
                height: 1,
                depth: 1,
            };
            let threadgroups = MTLSize {
                width: self.grid_threads.div_ceil(self.threads_per_threadgroup) as u64,
                height: 1,
                depth: 1,
            };
            encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            let status = command_buffer.status();
            if status != MTLCommandBufferStatus::Completed {
                return Err(MetalError::CommandFailed(status));
            }
            let start = command_buffer_timestamp(command_buffer, "GPUStartTime")?;
            let end = command_buffer_timestamp(command_buffer, "GPUEndTime")?;
            if !start.is_finite() || !end.is_finite() || start <= 0.0 || end < start {
                return Err(MetalError::InvalidGpuTimestamps { start, end });
            }
            self.completed.set(true);
            Ok(Duration::from_secs_f64(end - start))
        })
    }

    pub fn read_output(&self) -> Result<Vec<Fp128>, MetalError> {
        if !self.completed.get() {
            return Err(MetalError::NotExecuted);
        }
        let Some(buffers) = &self.buffers else {
            return Ok(Vec::new());
        };
        // SAFETY: `output` is shared storage allocated for exactly `elements`
        // `Fp128` values and GPU execution is complete before callers read it.
        let output = unsafe {
            slice::from_raw_parts(buffers.output.contents().cast::<Fp128>(), self.elements).to_vec()
        };
        if !self.probe.accepts_noncanonical_output() {
            if let Some((index, _)) = output
                .iter()
                .enumerate()
                .find(|(_, value)| !value.is_canonical(self.context.offset))
            {
                return Err(MetalError::NonCanonicalOutput {
                    index,
                    offset: self.context.offset,
                });
            }
        }
        Ok(output)
    }
}

fn buffer_from_slice<T>(device: &Device, values: &[T]) -> Buffer {
    debug_assert!(!values.is_empty());
    device.new_buffer_with_data(
        values.as_ptr().cast::<c_void>(),
        size_of_val(values) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn command_buffer_timestamp(
    command_buffer: &metal::CommandBufferRef,
    name: &'static str,
) -> Result<f64, MetalError> {
    // SAFETY: both selectors are required, argument-free MTLCommandBuffer
    // properties returning CFTimeInterval, which is an f64.
    unsafe { command_buffer.send_message::<(), f64>(Sel::register(name), ()) }.map_err(|error| {
        MetalError::GpuTimestampLookup {
            name,
            message: error.to_string(),
        }
    })
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "test module")]
mod tests {
    use std::mem::{align_of, size_of};

    use super::{AddressRafScanConfig, AddressRafScanRow, Fp128, SolinasMetal, OFFSET_275};

    #[test]
    fn fp128_has_the_metal_buffer_layout() {
        assert_eq!(size_of::<Fp128>(), 16);
        assert_eq!(align_of::<Fp128>(), 16);
    }

    #[test]
    fn limbs_are_little_endian() {
        let value = 0x0123_4567_89ab_cdef_fedc_ba98_7654_3210;
        let encoded = Fp128::from_u128(value);

        assert_eq!(encoded.to_u128(), value);
        assert_eq!(
            encoded.limbs(),
            [0x7654_3210, 0xfedc_ba98, 0x89ab_cdef, 0x0123_4567]
        );
    }

    #[test]
    fn canonicality_uses_the_selected_offset() {
        let largest = Fp128::from_u128(u128::MAX - OFFSET_275 as u128);
        let modulus = Fp128::from_u128(u128::MAX - OFFSET_275 as u128 + 1);

        assert!(largest.is_canonical(OFFSET_275));
        assert!(!modulus.is_canonical(OFFSET_275));
        assert!(!Fp128::ZERO.is_canonical(0));
    }

    #[test]
    fn address_raf_scan_reduces_exact_field_bins() {
        let context = SolinasMetal::for_akita().expect("Akita Metal context should compile");
        let rows = vec![AddressRafScanRow::new(0, false); 64];
        let weights: Vec<Fp128> = (1..=64).map(Fp128::from_u128).collect();
        let invocation = context
            .prepare_address_raf_scan(
                &rows,
                &weights,
                AddressRafScanConfig {
                    suffix_len: 120,
                    ..AddressRafScanConfig::default()
                },
            )
            .expect("address RAF scan should prepare");
        assert_eq!(
            invocation.intermediate_contribution_bytes(),
            rows.len() as u64 * 32
        );

        invocation
            .execute()
            .expect("address RAF scan should execute");
        let sums = invocation
            .read_output()
            .expect("address RAF output should be readable");
        let expected = (1u128..=64).sum();
        assert_eq!(sums.shift_half()[0], Fp128::from_u128(expected));
        assert!(sums
            .as_flat_slice()
            .iter()
            .enumerate()
            .all(|(index, value)| index == 0 || *value == Fp128::ZERO));
    }
}
