//! Upload and extraction on a caller-owned CUDA stream.
//!
//! NVRTC compiles the extraction module on first use, and function handles are
//! cached per device. Each `DeviceTrace` owns its packed window and instruction
//! tables; it does not cache witness rows globally. Mapping kernels report
//! unrepresentable indices through a device rejection flag checked on the host.
//! Large row-source windows alternate two pinned staging buffers, overlapping
//! packing with uploads without allocating a host copy of the entire window.
//! When random access identifies an implicit no-op suffix, only physical rows
//! are uploaded; a device fill expands the canonical packed padding row.

use super::packed::EXTRA_KIND_BITS;
use core::fmt::Display;
use cudarc::driver::DriverError;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{
    CudaFunction, CudaSlice, CudaStream, CudaViewMut, DeviceRepr, LaunchConfig, PinnedHostSlice,
    PushKernelArg, ValidAsZeroBits,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions, Ptx};
use jolt_program::preprocess::JoltProgramPreprocessing;
use jolt_riscv::JoltTraceRow as TraceRow;

use super::packed::{
    source_window_len, visit_source, PackedTrace, EXTRA_WORDS as PACKED_EXTRA_WORDS,
};
use super::staging::{PinnedTrace, MIN_PINNED_CYCLES, STAGING_CYCLES};
use super::tables::{kind_tables, KindTables};
use super::{
    DeviceAtomColumns, DeviceBytecodeColumns, DeviceExecutionColumns, DeviceMemoryColumns,
    DeviceTraceColumn, HotSource, NarrowColumn,
};
use crate::witnesses::RaChunkSelector;
use crate::{RowSource, WitnessError, JOLT_VM_LABEL};

pub const COLD: u32 = u32::MAX;

const EXTRACT_SRC: &str = concat!(
    include_str!("kernels/extract.cu"),
    include_str!("kernels/atoms.cu")
);

const BLOCK: u32 = 256;

const NO_REJECTION: u64 = u64::MAX;

struct ExtractFunctions {
    fill_padding: CudaFunction,
    mapped_pc: CudaFunction,
    remapped_ram: CudaFunction,
    lookup_index: CudaFunction,
    atom_columns: CudaFunction,
    flag_bit: CudaFunction,
    extra_word: CudaFunction,
    flag_bit_bytes: CudaFunction,
    narrow_u64: CudaFunction,
    hot_chunk_limbs: CudaFunction,
    hot_chunk_words: CudaFunction,
}

fn extract_ptx() -> Result<&'static Ptx, &'static String> {
    static PTX: OnceLock<Result<Ptx, String>> = OnceLock::new();
    PTX.get_or_init(|| {
        let options = CompileOptions {
            options: vec!["--device-int128".to_owned()],
            ..Default::default()
        };
        compile_ptx_with_opts(EXTRACT_SRC, options).map_err(|error| error.to_string())
    })
    .as_ref()
}

#[tracing::instrument(skip_all, name = "cuda_witness_nvrtc")]
fn extract_functions(stream: &CudaStream) -> Result<&'static ExtractFunctions, WitnessError> {
    static FUNCTIONS: OnceLock<Mutex<HashMap<usize, &'static Result<ExtractFunctions, String>>>> =
        OnceLock::new();
    let cache = FUNCTIONS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache.lock().unwrap_or_else(|poisoned| {
        cache.clear_poison();
        poisoned.into_inner()
    });
    let built = *cache
        .entry(stream.context().ordinal())
        .or_insert_with(|| Box::leak(Box::new(load_extract_functions(stream))));
    built.as_ref().map_err(device_error)
}

fn load_extract_functions(stream: &CudaStream) -> Result<ExtractFunctions, String> {
    let ptx = extract_ptx().map_err(Clone::clone)?.clone();
    let module = stream
        .context()
        .load_module(ptx)
        .map_err(|error| error.to_string())?;
    let function = |name: &str| {
        module
            .load_function(name)
            .map_err(|error: DriverError| error.to_string())
    };
    Ok(ExtractFunctions {
        fill_padding: function("fill_padding_kernel")?,
        mapped_pc: function("mapped_pc_words_kernel")?,
        remapped_ram: function("remapped_ram_words_kernel")?,
        lookup_index: function("lookup_index_limbs_kernel")?,
        atom_columns: function("atom_columns_kernel")?,
        flag_bit: function("flag_bit_column_kernel")?,
        extra_word: function("extra_word_column_kernel")?,
        flag_bit_bytes: function("flag_bit_bytes_kernel")?,
        narrow_u64: function("narrow_u64_kernel")?,
        hot_chunk_limbs: function("hot_chunk_limbs_kernel")?,
        hot_chunk_words: function("hot_chunk_words_kernel")?,
    })
}

/// A packed trace window whose extraction outputs use local cycle indices.
///
/// Callers needing next-row flags must include a successor row at non-final
/// window boundaries. The supplied stream also owns the resulting allocations;
/// extraction does not create an independent CUDA context.
pub struct DeviceTrace {
    stream: Arc<CudaStream>,
    functions: &'static ExtractFunctions,
    cycles: usize,
    rows: DeviceRows,
    kinds: DeviceKindTables,
    lowest_ram_address: u64,
}

struct DeviceRows {
    is_noop: CudaSlice<u8>,
    address: CudaSlice<u64>,
    virtual_sequence: CudaSlice<u32>,
    ram_address: CudaSlice<u64>,
    extras: CudaSlice<u64>,
}

fn copy_pinned<T: DeviceRepr + ValidAsZeroBits>(
    stream: &Arc<CudaStream>,
    source: &PinnedHostSlice<T>,
    destination: &mut CudaViewMut<'_, T>,
) -> Result<(), DriverError> {
    if source.len() == destination.len() {
        stream.memcpy_htod(source, destination)
    } else {
        // cudarc's plain-slice copy waits before releasing the borrow. This
        // handles short source chunks without uploading unused buffer entries.
        stream.memcpy_htod(&source.as_slice()?[..destination.len()], destination)
    }
}

impl DeviceRows {
    fn upload(
        &mut self,
        stream: &Arc<CudaStream>,
        staging: &PackedTrace,
        range: Range<usize>,
    ) -> Result<(), DriverError> {
        stream.memcpy_htod(&staging.is_noop, &mut self.is_noop.slice_mut(range.clone()))?;
        stream.memcpy_htod(&staging.address, &mut self.address.slice_mut(range.clone()))?;
        stream.memcpy_htod(
            &staging.virtual_sequence,
            &mut self.virtual_sequence.slice_mut(range.clone()),
        )?;
        stream.memcpy_htod(
            &staging.ram_address,
            &mut self.ram_address.slice_mut(range.clone()),
        )?;
        stream.memcpy_htod(
            &staging.extras,
            &mut self
                .extras
                .slice_mut(range.start * PACKED_EXTRA_WORDS..range.end * PACKED_EXTRA_WORDS),
        )
    }

    fn upload_pinned(
        &mut self,
        stream: &Arc<CudaStream>,
        staging: &PinnedTrace,
        range: Range<usize>,
    ) -> Result<(), DriverError> {
        copy_pinned(
            stream,
            &staging.is_noop,
            &mut self.is_noop.slice_mut(range.clone()),
        )?;
        copy_pinned(
            stream,
            &staging.address,
            &mut self.address.slice_mut(range.clone()),
        )?;
        copy_pinned(
            stream,
            &staging.virtual_sequence,
            &mut self.virtual_sequence.slice_mut(range.clone()),
        )?;
        copy_pinned(
            stream,
            &staging.ram_address,
            &mut self.ram_address.slice_mut(range.clone()),
        )?;
        copy_pinned(
            stream,
            &staging.extras,
            &mut self
                .extras
                .slice_mut(range.start * PACKED_EXTRA_WORDS..range.end * PACKED_EXTRA_WORDS),
        )
    }
}

struct DeviceKindTables {
    input: CudaSlice<u8>,
    operand: CudaSlice<u8>,
    output: CudaSlice<u8>,
    index: CudaSlice<u8>,
    flags: CudaSlice<u32>,
    table_index: CudaSlice<u32>,
    count: u32,
}

fn device_error(reason: impl Display) -> WitnessError {
    WitnessError::InvalidWitnessData {
        label: JOLT_VM_LABEL,
        reason: reason.to_string(),
    }
}

fn launch_config(count: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (count.div_ceil(BLOCK), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

impl DeviceTrace {
    pub fn upload(
        stream: Arc<CudaStream>,
        rows: &[TraceRow],
        cycles: usize,
        preprocessing: &JoltProgramPreprocessing,
    ) -> Result<Self, WitnessError> {
        Self::upload_window(stream, rows, cycles, 0, cycles, preprocessing)
    }

    pub fn upload_window(
        stream: Arc<CudaStream>,
        rows: &[TraceRow],
        cycles: usize,
        base: usize,
        len: usize,
        preprocessing: &JoltProgramPreprocessing,
    ) -> Result<Self, WitnessError> {
        PackedTrace::require_domain(rows, cycles)?;
        if base + len > cycles {
            return Err(WitnessError::InvalidDimensions {
                label: JOLT_VM_LABEL,
                reason: format!(
                    "cycle window {base}..{} lies outside the cycle domain {cycles}",
                    base + len
                ),
            });
        }
        let mut staging = PackedTrace::with_capacity(len);
        staging.fill_range(rows, cycles, base, len);
        Self::upload_packed(stream, staging, preprocessing)
    }

    pub fn upload_source(
        stream: Arc<CudaStream>,
        source: &dyn RowSource,
        range: Range<usize>,
        preprocessing: &JoltProgramPreprocessing,
    ) -> Result<Self, WitnessError> {
        let cycles = source_window_len(&range)?;
        if cycles < MIN_PINNED_CYCLES {
            return Self::upload_packed(
                stream,
                PackedTrace::from_source(source, range)?,
                preprocessing,
            );
        }

        let physical_end = source
            .random_access()
            .filter(|access| range.end <= access.cycles())
            .map_or(range.end, |access| {
                access.physical_cycles().clamp(range.start, range.end)
            });
        let tables = kind_tables()?;
        let mut trace = Self::allocate(Arc::clone(&stream), cycles, preprocessing, &tables)?;
        if range.start < physical_end {
            let mut staging = [
                PinnedTrace::new(stream.context()).map_err(device_error)?,
                PinnedTrace::new(stream.context()).map_err(device_error)?,
            ];
            let mut next = 0;
            visit_source(
                source,
                range.start..physical_end,
                STAGING_CYCLES,
                |rows, destination| {
                    if rows.is_empty() {
                        return Ok(());
                    }
                    let buffer = &mut staging[next];
                    buffer.pack(rows).map_err(device_error)?;
                    tables.require_supported(
                        buffer.extras.as_slice().map_err(device_error)?
                            [..rows.len() * PACKED_EXTRA_WORDS]
                            .chunks_exact(PACKED_EXTRA_WORDS)
                            .map(|row| row[EXTRA_KIND_BITS] as u16),
                    )?;
                    trace
                        .rows
                        .upload_pinned(&stream, buffer, destination)
                        .map_err(device_error)?;
                    next ^= 1;
                    Ok(())
                },
            )?;
        }
        if physical_end < range.end {
            let padding = PackedTrace::pack(&[], 1)?;
            tables.require_supported(
                padding
                    .extras
                    .chunks_exact(PACKED_EXTRA_WORDS)
                    .map(|row| row[EXTRA_KIND_BITS] as u16),
            )?;
            trace.fill_padding(physical_end - range.start, &padding)?;
        }
        stream.synchronize().map_err(device_error)?;
        Ok(trace)
    }

    fn fill_padding(&mut self, start: usize, padding: &PackedTrace) -> Result<(), WitnessError> {
        self.rows
            .upload(&self.stream, padding, start..start + 1)
            .map_err(device_error)?;
        let count = self.count()?;
        let start = u32::try_from(start).map_err(device_error)?;
        let remaining = count - start - 1;
        if remaining == 0 {
            return Ok(());
        }
        let mut builder = self.stream.launch_builder(&self.functions.fill_padding);
        let _ = builder.arg(&mut self.rows.is_noop);
        let _ = builder.arg(&mut self.rows.address);
        let _ = builder.arg(&mut self.rows.virtual_sequence);
        let _ = builder.arg(&mut self.rows.ram_address);
        let _ = builder.arg(&mut self.rows.extras);
        let _ = builder.arg(&start);
        let _ = builder.arg(&remaining);
        // SAFETY: upload_source calls only for a nonempty suffix inside cycles.
        // The seed row is uploaded on this stream and never written by the
        // kernel. Each thread writes one later row in the same allocations;
        // extras uses the shared PACKED_EXTRA_WORDS layout.
        let _ = unsafe { builder.launch(launch_config(remaining)) }.map_err(device_error)?;
        Ok(())
    }

    fn upload_packed(
        stream: Arc<CudaStream>,
        staging: PackedTrace,
        preprocessing: &JoltProgramPreprocessing,
    ) -> Result<Self, WitnessError> {
        let cycles = staging.cycles;
        let tables = kind_tables()?;
        tables.require_supported(
            staging
                .extras
                .chunks_exact(PACKED_EXTRA_WORDS)
                .map(|row| row[EXTRA_KIND_BITS] as u16),
        )?;

        let mut trace = Self::allocate(Arc::clone(&stream), cycles, preprocessing, &tables)?;
        trace
            .rows
            .upload(&stream, &staging, 0..cycles)
            .map_err(device_error)?;
        drop(staging);

        Ok(trace)
    }

    fn allocate(
        stream: Arc<CudaStream>,
        cycles: usize,
        preprocessing: &JoltProgramPreprocessing,
        tables: &KindTables,
    ) -> Result<Self, WitnessError> {
        let functions = extract_functions(&stream)?;
        let device = DeviceRows {
            is_noop: stream.alloc_zeros(cycles).map_err(device_error)?,
            address: stream.alloc_zeros(cycles).map_err(device_error)?,
            virtual_sequence: stream.alloc_zeros(cycles).map_err(device_error)?,
            ram_address: stream.alloc_zeros(cycles).map_err(device_error)?,
            extras: stream
                .alloc_zeros(cycles * PACKED_EXTRA_WORDS)
                .map_err(device_error)?,
        };
        let kinds = DeviceKindTables {
            input: stream.clone_htod(&tables.input).map_err(device_error)?,
            operand: stream.clone_htod(&tables.operand).map_err(device_error)?,
            output: stream.clone_htod(&tables.output).map_err(device_error)?,
            index: stream.clone_htod(&tables.index).map_err(device_error)?,
            flags: stream.clone_htod(&tables.flags).map_err(device_error)?,
            table_index: stream
                .clone_htod(&tables.table_index)
                .map_err(device_error)?,
            count: tables.count,
        };

        Ok(Self {
            stream,
            functions,
            cycles,
            rows: device,
            kinds,
            lowest_ram_address: preprocessing.memory_layout.get_lowest_address(),
        })
    }

    pub fn cycles(&self) -> usize {
        self.cycles
    }

    pub fn lowest_ram_address(&self) -> u64 {
        self.lowest_ram_address
    }

    pub fn unexpanded_pc(&self) -> &CudaSlice<u64> {
        &self.rows.address
    }

    pub fn extras(&self) -> &CudaSlice<u64> {
        &self.rows.extras
    }

    pub fn ram_address(&self) -> &CudaSlice<u64> {
        &self.rows.ram_address
    }

    pub fn device_bytes(&self) -> usize {
        let rows = self.rows.is_noop.len()
            + self.rows.address.len() * size_of::<u64>()
            + self.rows.virtual_sequence.len() * size_of::<u32>()
            + self.rows.ram_address.len() * size_of::<u64>()
            + self.rows.extras.len() * size_of::<u64>();
        let kinds = self.kinds.input.len()
            + self.kinds.operand.len()
            + self.kinds.output.len()
            + self.kinds.index.len()
            + self.kinds.flags.len() * size_of::<u32>()
            + self.kinds.table_index.len() * size_of::<u32>();
        rows + kinds
    }

    fn count(&self) -> Result<u32, WitnessError> {
        u32::try_from(self.cycles)
            .map_err(|_| device_error(format!("{} cycles exceed a 32-bit grid", self.cycles)))
    }

    fn outputs(&self) -> Result<(CudaSlice<u32>, CudaSlice<u64>), WitnessError> {
        let out = self
            .stream
            .alloc_zeros::<u32>(self.cycles)
            .map_err(device_error)?;
        let rejected = self
            .stream
            .clone_htod(&[NO_REJECTION])
            .map_err(device_error)?;
        Ok((out, rejected))
    }

    fn finish(
        &self,
        out: CudaSlice<u32>,
        rejected: CudaSlice<u64>,
        label: &str,
    ) -> Result<CudaSlice<u32>, WitnessError> {
        self.stream.synchronize().map_err(device_error)?;
        let rejected = self.stream.clone_dtoh(&rejected).map_err(device_error)?;
        match rejected.first() {
            Some(&NO_REJECTION) => Ok(out),
            Some(&value) => Err(device_error(format!(
                "{label} {value} does not fit its packed word"
            ))),
            None => Err(device_error("the rejection flag was not read back")),
        }
    }

    fn unmapped_flag(&self) -> Result<CudaSlice<u32>, WitnessError> {
        self.stream.clone_htod(&[0u32]).map_err(device_error)
    }

    fn check_mapped(&self, unmapped: &CudaSlice<u32>, oracle: &str) -> Result<(), WitnessError> {
        self.stream.synchronize().map_err(device_error)?;
        let unmapped = self.stream.clone_dtoh(unmapped).map_err(device_error)?;
        if unmapped.first().is_some_and(|&flag| flag != 0) {
            return Err(WitnessError::NotServed {
                oracle: oracle.to_owned(),
                reason: "the trace contains an instruction kind with no device descriptor",
            });
        }
        Ok(())
    }

    /// The 128-bit lookup index as consecutive low/high words per local cycle.
    #[tracing::instrument(skip_all, name = "cuda_witness_lookup_limbs", fields(cycles = self.cycles))]
    pub fn lookup_index_limbs(&self) -> Result<CudaSlice<u64>, WitnessError> {
        let count = self.count()?;
        let mut out = self
            .stream
            .alloc_zeros::<u64>(self.cycles * 2)
            .map_err(device_error)?;
        let mut unmapped = self.unmapped_flag()?;

        let mut builder = self.stream.launch_builder(&self.functions.lookup_index);
        let _ = builder.arg(&self.rows.extras);
        let _ = builder.arg(&self.rows.address);
        let _ = builder.arg(&self.kinds.input);
        let _ = builder.arg(&self.kinds.operand);
        let _ = builder.arg(&self.kinds.index);
        let _ = builder.arg(&self.kinds.count);
        let _ = builder.arg(&mut out);
        let _ = builder.arg(&mut unmapped);
        let _ = builder.arg(&count);
        // SAFETY: thread `i < count` reads index `i` of `address` and the
        // `EXTRA_WORDS` consecutive words at `i * EXTRA_WORDS` of `extras`
        // (allocated as `cycles * EXTRA_WORDS`), and writes exactly `out[2*i]`
        // and `out[2*i + 1]` of a `cycles * 2` allocation. The kind tables are
        // bounds-checked against `count` in the kernel before indexing.
        // `unmapped` is a single flag written only by `atomicExch`.
        let _ = unsafe { builder.launch(launch_config(count)) }.map_err(device_error)?;

        self.check_mapped(&unmapped, "lookup index limbs")?;
        Ok(out)
    }

    #[tracing::instrument(skip_all, name = "cuda_witness_atom_columns", fields(cycles = self.cycles))]
    pub fn atom_columns(&self) -> Result<DeviceAtomColumns, WitnessError> {
        let mut execution = self.allocate_execution()?;
        let mut memory = self.allocate_memory()?;
        let mut bytecode = self.allocate_bytecode()?;
        self.extract_columns(Some(&mut execution), Some(&mut memory), Some(&mut bytecode))?;
        Ok(DeviceAtomColumns {
            execution: Arc::new(execution),
            memory: Arc::new(memory),
            bytecode: Arc::new(bytecode),
        })
    }

    #[tracing::instrument(skip_all, name = "cuda_witness_execution_columns", fields(cycles = self.cycles))]
    pub fn execution_columns(&self) -> Result<DeviceExecutionColumns, WitnessError> {
        let mut columns = self.allocate_execution()?;
        self.extract_columns(Some(&mut columns), None, None)?;
        Ok(columns)
    }

    fn allocate_execution(&self) -> Result<DeviceExecutionColumns, WitnessError> {
        let _ = self.count()?;
        Ok(DeviceExecutionColumns {
            flags: self
                .stream
                .alloc_zeros::<u32>(self.cycles)
                .map_err(device_error)?,
            table_index: self
                .stream
                .alloc_zeros::<u32>(self.cycles)
                .map_err(device_error)?,
            left_instruction_input: self
                .stream
                .alloc_zeros::<u64>(self.cycles)
                .map_err(device_error)?,
            right_instruction_input: self
                .stream
                .alloc_zeros::<u64>(self.cycles * 2)
                .map_err(device_error)?,
            left_lookup_operand: self
                .stream
                .alloc_zeros::<u64>(self.cycles)
                .map_err(device_error)?,
            right_lookup_operand: self
                .stream
                .alloc_zeros::<u64>(self.cycles * 2)
                .map_err(device_error)?,
            lookup_output: self
                .stream
                .alloc_zeros::<u64>(self.cycles)
                .map_err(device_error)?,
            product_magnitude: self
                .stream
                .alloc_zeros::<u64>(self.cycles * 2)
                .map_err(device_error)?,
        })
    }

    #[tracing::instrument(skip_all, name = "cuda_witness_memory_columns", fields(cycles = self.cycles))]
    pub fn memory_columns(&self) -> Result<DeviceMemoryColumns, WitnessError> {
        let mut columns = self.allocate_memory()?;
        self.extract_columns(None, Some(&mut columns), None)?;
        Ok(columns)
    }

    fn allocate_memory(&self) -> Result<DeviceMemoryColumns, WitnessError> {
        let _ = self.count()?;
        Ok(DeviceMemoryColumns {
            rd_pre_value: self
                .stream
                .alloc_zeros::<u64>(self.cycles)
                .map_err(device_error)?,
            rs1_address: self
                .stream
                .alloc_zeros::<u32>(self.cycles)
                .map_err(device_error)?,
            rs2_address: self
                .stream
                .alloc_zeros::<u32>(self.cycles)
                .map_err(device_error)?,
            rd_address: self
                .stream
                .alloc_zeros::<u32>(self.cycles)
                .map_err(device_error)?,
            rd_inc: self
                .stream
                .alloc_zeros::<u64>(self.cycles * 2)
                .map_err(device_error)?,
            ram_inc: self
                .stream
                .alloc_zeros::<u64>(self.cycles * 2)
                .map_err(device_error)?,
        })
    }

    #[tracing::instrument(skip_all, name = "cuda_witness_bytecode_columns", fields(cycles = self.cycles))]
    pub fn bytecode_columns(&self) -> Result<DeviceBytecodeColumns, WitnessError> {
        let mut columns = self.allocate_bytecode()?;
        self.extract_columns(None, None, Some(&mut columns))?;
        Ok(columns)
    }

    fn allocate_bytecode(&self) -> Result<DeviceBytecodeColumns, WitnessError> {
        let _ = self.count()?;
        Ok(DeviceBytecodeColumns {
            bytecode_pc: self
                .stream
                .alloc_zeros::<u64>(self.cycles)
                .map_err(device_error)?,
        })
    }

    fn extract_columns(
        &self,
        execution: Option<&mut DeviceExecutionColumns>,
        memory: Option<&mut DeviceMemoryColumns>,
        bytecode: Option<&mut DeviceBytecodeColumns>,
    ) -> Result<(), WitnessError> {
        let count = self.count()?;
        let groups = u32::from(execution.is_some())
            | (u32::from(memory.is_some()) << 1)
            | (u32::from(bytecode.is_some()) << 2);
        let mut unmapped = self.unmapped_flag()?;
        let null = 0u64;
        let (
            flags,
            table_index,
            left_instruction_input,
            right_instruction_input,
            left_lookup_operand,
            right_lookup_operand,
            lookup_output,
            product_magnitude,
        ) = match execution {
            Some(columns) => (
                Some(&mut columns.flags),
                Some(&mut columns.table_index),
                Some(&mut columns.left_instruction_input),
                Some(&mut columns.right_instruction_input),
                Some(&mut columns.left_lookup_operand),
                Some(&mut columns.right_lookup_operand),
                Some(&mut columns.lookup_output),
                Some(&mut columns.product_magnitude),
            ),
            None => (None, None, None, None, None, None, None, None),
        };
        let (rd_pre_value, rs1_address, rs2_address, rd_address, rd_inc, ram_inc) = match memory {
            Some(columns) => (
                Some(&mut columns.rd_pre_value),
                Some(&mut columns.rs1_address),
                Some(&mut columns.rs2_address),
                Some(&mut columns.rd_address),
                Some(&mut columns.rd_inc),
                Some(&mut columns.ram_inc),
            ),
            None => (None, None, None, None, None, None),
        };
        let (bytecode_pc,) = match bytecode {
            Some(columns) => (Some(&mut columns.bytecode_pc),),
            None => (None,),
        };
        let mut builder = self.stream.launch_builder(&self.functions.atom_columns);
        let _ = builder.arg(&self.rows.is_noop);
        let _ = builder.arg(&self.rows.address);
        let _ = builder.arg(&self.rows.extras);
        let _ = builder.arg(&self.rows.virtual_sequence);
        let _ = builder.arg(&self.rows.ram_address);
        let _ = builder.arg(&self.kinds.flags);
        let _ = builder.arg(&self.kinds.table_index);
        let _ = builder.arg(&self.kinds.input);
        let _ = builder.arg(&self.kinds.operand);
        let _ = builder.arg(&self.kinds.output);
        let _ = builder.arg(&self.kinds.index);
        let _ = builder.arg(&self.kinds.count);
        macro_rules! output {
            ($column:ident) => {
                if let Some(column) = $column {
                    let _ = builder.arg(column);
                } else {
                    let _ = builder.arg(&null);
                }
            };
        }
        output!(flags);
        output!(table_index);
        output!(bytecode_pc);
        output!(rd_pre_value);
        output!(rs1_address);
        output!(rs2_address);
        output!(rd_address);
        output!(rd_inc);
        output!(ram_inc);
        output!(left_instruction_input);
        output!(right_instruction_input);
        output!(left_lookup_operand);
        output!(right_lookup_operand);
        output!(lookup_output);
        output!(product_magnitude);
        let _ = builder.arg(&mut unmapped);
        let _ = builder.arg(&groups);
        let _ = builder.arg(&count);
        // SAFETY: the mask is derived from the supplied output groups. Disabled
        // groups receive null pointers and are never accessed. Enabled outputs
        // cover `cycles` rows, with two u64 limbs for wide values. Thread `i`
        // reads row `i` and its EXTRA_WORDS extras; successor reads require
        // `i + 1 < cycles`. The kernel bounds-checks each descriptor kind, and
        // its sequence variant is in 0..VARIANTS. Each output is a separate
        // allocation, and the rejection flag is written with atomicExch.
        let _ = unsafe { builder.launch(launch_config(count)) }.map_err(device_error)?;
        self.check_mapped(&unmapped, "atom columns")
    }

    pub fn flag_bit_bytes(
        &self,
        flags: &CudaSlice<u32>,
        bit: u32,
    ) -> Result<CudaSlice<u8>, WitnessError> {
        if bit >= u32::BITS {
            return Err(device_error(format!(
                "flag bit {bit} is outside the {}-bit mask",
                u32::BITS
            )));
        }
        if flags.len() < self.cycles {
            return Err(device_error(format!(
                "the flag column holds {} entries for {} cycles",
                flags.len(),
                self.cycles
            )));
        }
        let count = self.count()?;
        let mut out = self
            .stream
            .alloc_zeros::<u8>(self.cycles)
            .map_err(device_error)?;
        let mut builder = self.stream.launch_builder(&self.functions.flag_bit_bytes);
        let _ = builder.arg(flags);
        let _ = builder.arg(&bit);
        let _ = builder.arg(&mut out);
        let _ = builder.arg(&count);
        // SAFETY: thread `i < count` reads `flags[i]` (checked above to cover
        // `cycles`) and writes only `out[i]` of a fresh `cycles`-byte
        // allocation, a distinct buffer. Threads with `i >= count` return first.
        let _ = unsafe { builder.launch(launch_config(count)) }.map_err(device_error)?;
        Ok(out)
    }

    pub fn extra_word_column(&self, word: usize) -> Result<CudaSlice<u64>, WitnessError> {
        if word >= PACKED_EXTRA_WORDS {
            return Err(device_error(format!(
                "packed word {word} is outside the {PACKED_EXTRA_WORDS}-word row stride"
            )));
        }
        let count = self.count()?;
        let word = u32::try_from(word).map_err(|_| {
            device_error("the packed row stride does not fit a 32-bit word index".to_owned())
        })?;
        let mut out = self
            .stream
            .alloc_zeros::<u64>(self.cycles)
            .map_err(device_error)?;
        let mut builder = self.stream.launch_builder(&self.functions.extra_word);
        let _ = builder.arg(&self.rows.extras);
        let _ = builder.arg(&word);
        let _ = builder.arg(&mut out);
        let _ = builder.arg(&count);
        // SAFETY: thread `i < count` reads the single word at
        // `i * EXTRA_WORDS + word` — inside the `cycles * EXTRA_WORDS` extras
        // buffer because `word < EXTRA_WORDS` is checked above — and writes only
        // `out[i]` of a fresh `cycles`-element allocation. The two are distinct
        // allocations. Threads with `i >= count` return before any access.
        let _ = unsafe { builder.launch(launch_config(count)) }.map_err(device_error)?;
        Ok(out)
    }

    pub fn flag_bit_column(
        &self,
        flags: &CudaSlice<u32>,
        bit: u32,
    ) -> Result<CudaSlice<u64>, WitnessError> {
        if bit >= u32::BITS {
            return Err(device_error(format!(
                "flag bit {bit} is outside the {}-bit mask",
                u32::BITS
            )));
        }
        if flags.len() < self.cycles {
            return Err(device_error(format!(
                "the flag column holds {} entries for {} cycles",
                flags.len(),
                self.cycles
            )));
        }
        let count = self.count()?;
        let mut out = self
            .stream
            .alloc_zeros::<u64>(self.cycles)
            .map_err(device_error)?;

        let mut builder = self.stream.launch_builder(&self.functions.flag_bit);
        let _ = builder.arg(flags);
        let _ = builder.arg(&bit);
        let _ = builder.arg(&mut out);
        let _ = builder.arg(&count);
        // SAFETY: thread `i < count` reads only `flags[i]`, checked above to hold
        // at least `cycles >= count` entries, and writes only `out[i]` of a
        // `cycles`-element allocation. The two are distinct allocations.
        let _ = unsafe { builder.launch(launch_config(count)) }.map_err(device_error)?;
        Ok(out)
    }

    pub fn narrow_u64_column(
        &self,
        source: &CudaSlice<u64>,
        bound: u64,
    ) -> Result<NarrowColumn, WitnessError> {
        if source.len() < self.cycles {
            return Err(device_error(format!(
                "the narrow source holds {} entries for {} cycles",
                source.len(),
                self.cycles
            )));
        }
        let count = self.count()?;
        let mut out = self
            .stream
            .alloc_zeros::<u32>(self.cycles)
            .map_err(device_error)?;
        let mut facts = self
            .stream
            .clone_htod(&[NO_REJECTION, 0u64, 0u64])
            .map_err(device_error)?;

        let mut builder = self.stream.launch_builder(&self.functions.narrow_u64);
        let _ = builder.arg(source);
        let _ = builder.arg(&bound);
        let _ = builder.arg(&mut out);
        let _ = builder.arg(&mut facts);
        let _ = builder.arg(&count);
        // SAFETY: thread `i < count` reads only `source[i]`, checked above to hold
        // at least `cycles >= count` entries, and writes only `out[i]` of a
        // `cycles`-element allocation. `facts` holds three words: index 0 and 1
        // are mutated only through `atomicMin`/`atomicMax`, and index 2 is
        // written only by thread 0. Every buffer is a distinct allocation.
        let _ = unsafe { builder.launch(launch_config(count)) }.map_err(device_error)?;
        self.stream.synchronize().map_err(device_error)?;

        let facts = self.stream.clone_dtoh(&facts).map_err(device_error)?;
        let &[rejected, span, first] = facts.as_slice() else {
            return Err(device_error("the narrow column facts were not read back"));
        };
        if rejected != NO_REJECTION {
            return Err(device_error(format!(
                "the column value {rejected} does not fit the {bound}-bound 32-bit word"
            )));
        }
        Ok(NarrowColumn {
            column: out,
            span: usize::try_from(span)
                .map_err(|_| device_error("a narrowed column span exceeds the host word"))?,
            first,
        })
    }

    #[tracing::instrument(skip_all, name = "cuda_witness_hot_chunks", fields(requests = requests.len(), cycles))]
    pub fn hot_chunk_columns(
        &self,
        requests: &[(HotSource<'_>, RaChunkSelector)],
        addresses: usize,
        cycles: usize,
    ) -> Result<Vec<(CudaSlice<u32>, usize)>, WitnessError> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        if cycles == 0 || cycles > self.cycles {
            return Err(device_error(format!(
                "a committed chunk range of {cycles} cycles does not fit this {}-cycle residency",
                self.cycles
            )));
        }
        let count = u32::try_from(cycles)
            .map_err(|_| device_error(format!("{cycles} cycles exceed a 32-bit grid")))?;
        let mut spans = self
            .stream
            .alloc_zeros::<u64>(requests.len())
            .map_err(device_error)?;
        let mut columns = Vec::with_capacity(requests.len());
        for (slot, (source, selector)) in requests.iter().enumerate() {
            let selector = *selector;
            let mask = selector.mask();
            if mask >= u128::from(COLD) {
                return Err(device_error(format!(
                    "a {}-bit committed chunk does not fit the 32-bit word reserving {COLD} for a \
                     cold cycle",
                    mask.count_ones()
                )));
            }
            let shift = u32::try_from(selector.shift())
                .map_err(|_| device_error("a committed chunk shift exceeds 32 bits"))?;
            let mask = mask as u64;
            let slot = u32::try_from(slot)
                .map_err(|_| device_error("a committed chunk batch exceeds a 32-bit slot"))?;
            let (function, expected, source_len) = match source {
                HotSource::Interleaved(limbs) => {
                    (&self.functions.hot_chunk_limbs, cycles * 2, limbs.len())
                }
                HotSource::Word(words) => (&self.functions.hot_chunk_words, cycles, words.len()),
            };
            if source_len < expected {
                return Err(device_error(format!(
                    "the chunk source holds {source_len} entries for {cycles} cycles"
                )));
            }
            let mut out = self
                .stream
                .alloc_zeros::<u32>(cycles)
                .map_err(device_error)?;

            let mut builder = self.stream.launch_builder(function);
            match source {
                HotSource::Interleaved(limbs) => {
                    let _ = builder.arg(limbs);
                }
                HotSource::Word(words) => {
                    let _ = builder.arg(words);
                }
            }
            let _ = builder.arg(&shift);
            let _ = builder.arg(&mask);
            let _ = builder.arg(&mut out);
            let _ = builder.arg(&mut spans);
            let _ = builder.arg(&slot);
            let _ = builder.arg(&count);
            // SAFETY: thread `i < count` reads `limbs[2i]`/`limbs[2i + 1]` or
            // `words[i]` of the caller's view — the view was checked above to hold
            // at least `cycles * 2` or `cycles` entries from its own base — and
            // writes only `out[i]` of a
            // fresh `cycles`-element allocation. `spans[slot]` is one element of
            // a `requests.len()`-element buffer, `slot` is this request's index,
            // and warp leaders update it only through `atomicMax`. All threads
            // reach the warp reduction, including cold/out-of-range lanes with
            // value zero; BLOCK is a multiple of 32. The checked mask is below
            // COLD, so chunk + 1 fits u32. Every buffer is a distinct allocation.
            let _ = unsafe { builder.launch(launch_config(count)) }.map_err(device_error)?;
            columns.push(out);
        }

        self.stream.synchronize().map_err(device_error)?;
        let spans = self.stream.clone_dtoh(&spans).map_err(device_error)?;
        let mut chunked = Vec::with_capacity(columns.len());
        for (column, &span) in columns.into_iter().zip(&spans) {
            let span = usize::try_from(span)
                .map_err(|_| device_error("a committed chunk span exceeds the host word"))?;
            if span > addresses {
                return Err(device_error(format!(
                    "a committed one-hot column reaches address {} of a {addresses}-address chunk",
                    span - 1
                )));
            }
            chunked.push((column, span));
        }
        Ok(chunked)
    }

    pub fn u32_column(&self, column: DeviceTraceColumn) -> Result<CudaSlice<u32>, WitnessError> {
        match column {
            DeviceTraceColumn::MappedPcWord => self.mapped_pc_words(),
            DeviceTraceColumn::RemappedRamWord { addresses } => {
                self.remapped_ram_words(addresses).map(|(column, _)| column)
            }
        }
    }

    pub fn mapped_pc_words(&self) -> Result<CudaSlice<u32>, WitnessError> {
        let count = self.count()?;
        let (mut out, mut rejected) = self.outputs()?;

        let mut builder = self.stream.launch_builder(&self.functions.mapped_pc);
        let _ = builder.arg(&self.rows.extras);
        let _ = builder.arg(&mut out);
        let _ = builder.arg(&mut rejected);
        let _ = builder.arg(&count);
        // SAFETY: each thread reads its cached PC from an EXTRA_WORDS-sized
        // row and writes one output; the rejection scalar uses atomicMin.
        let _ = unsafe { builder.launch(launch_config(count)) }.map_err(device_error)?;

        self.finish(out, rejected, "the mapped PC")
    }

    pub fn remapped_ram_words(
        &self,
        addresses: usize,
    ) -> Result<(CudaSlice<u32>, usize), WitnessError> {
        let count = self.count()?;
        let (mut out, mut rejected) = self.outputs()?;
        let mut span = self.stream.clone_htod(&[0u64]).map_err(device_error)?;
        let lowest = self.lowest_ram_address;
        let addresses = addresses as u64;

        let mut builder = self.stream.launch_builder(&self.functions.remapped_ram);
        let _ = builder.arg(&self.rows.ram_address);
        let _ = builder.arg(&lowest);
        let _ = builder.arg(&addresses);
        let _ = builder.arg(&mut out);
        let _ = builder.arg(&mut rejected);
        let _ = builder.arg(&mut span);
        let _ = builder.arg(&count);
        // SAFETY: thread `i < count` writes only `out[i]` and reads only
        // `ram_address[i]`; both hold `cycles >= count` elements. `rejected` and
        // `span` are single elements mutated only through `atomicMin` and
        // `atomicMax`. Every buffer is a distinct allocation.
        let _ = unsafe { builder.launch(launch_config(count)) }.map_err(device_error)?;

        let out = self.finish(out, rejected, "the remapped RAM address")?;
        let span = self.stream.clone_dtoh(&span).map_err(device_error)?;
        let span = usize::try_from(
            *span
                .first()
                .ok_or_else(|| device_error("the remapped RAM span was not read back"))?,
        )
        .map_err(|_| device_error("the remapped RAM span exceeds the host word"))?;
        Ok((out, span))
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "test setup and device failures fail loudly"
)]
mod tests {
    use super::*;
    use crate::backend::ProgramSource;
    use crate::testing::with_sample_backend;
    use crate::witnesses::WitnessEnv;
    use crate::{ChunkVisitor, RandomAccessRows};
    use cudarc::driver::CudaContext;
    use jolt_program::execution::TraceData;
    use jolt_riscv::{
        JoltInstructionKind, JoltInstructionRow, NormalizedOperands, RamAccess, RegisterRead,
        RegisterState, RegisterWrite,
    };

    use super::super::packed::{
        EXTRA_BYTECODE_PC, EXTRA_RD_POST, EXTRA_RS1, NO_SEQUENCE, RAM_NO_ACCESS,
    };

    #[test]
    fn cuda_source_compiles_without_device() -> Result<(), String> {
        super::extract_ptx().map(|_| ()).map_err(Clone::clone)
    }

    #[test]
    fn source_upload_preserves_implicit_padding() {
        struct Source<'a> {
            rows: Vec<TraceRow>,
            access: RandomAccessRows,
            env: WitnessEnv<'a>,
        }

        impl RowSource for Source<'_> {
            fn visit_chunks(
                &self,
                range: Range<usize>,
                chunk_size: usize,
                visitor: &mut ChunkVisitor<'_>,
            ) -> Result<(), WitnessError> {
                if range.start > range.end || range.end > self.access.cycles() {
                    return Err(device_error("source range exceeds its domain"));
                }
                for start in (range.start..range.end).step_by(chunk_size) {
                    let end = (start + chunk_size).min(range.end);
                    let chunk: Vec<_> = (start..end)
                        .map(|index| self.rows.get(index).copied().unwrap_or_default())
                        .collect();
                    let padding = TraceRow::default();
                    let next = (end < self.access.cycles())
                        .then(|| self.rows.get(end).unwrap_or(&padding));
                    visitor(&chunk, next, &self.env)?;
                }
                Ok(())
            }

            fn random_access(&self) -> Option<RandomAccessRows> {
                Some(self.access.clone())
            }
        }

        with_sample_backend(|backend| {
            let stream = CudaContext::new(0).unwrap().default_stream();
            let mut rows = Vec::new();
            backend
                .visit_chunks(0..2, 2, &mut |chunk, _, _| {
                    rows.extend_from_slice(chunk);
                    Ok(())
                })
                .unwrap();
            let cycles = 2 * MIN_PINNED_CYCLES;
            let source = Source {
                access: RandomAccessRows::new(
                    Arc::new(TraceData::new(rows.clone())),
                    cycles,
                    Arc::new(backend.program_preprocessing().clone()),
                )
                .unwrap(),
                rows,
                env: WitnessEnv::new(backend.program_preprocessing()),
            };
            for range in [
                0..MIN_PINNED_CYCLES + 17,
                1..MIN_PINNED_CYCLES + 1,
                MIN_PINNED_CYCLES..cycles,
            ] {
                let expected = PackedTrace::from_source(&source, range.clone()).unwrap();
                let trace = DeviceTrace::upload_source(
                    Arc::clone(&stream),
                    &source,
                    range,
                    backend.program_preprocessing(),
                )
                .unwrap();
                assert_eq!(trace.cycles(), expected.cycles);
                assert_eq!(
                    stream.clone_dtoh(&trace.rows.is_noop).unwrap(),
                    expected.is_noop
                );
                assert_eq!(
                    stream.clone_dtoh(&trace.rows.address).unwrap(),
                    expected.address
                );
                assert_eq!(
                    stream.clone_dtoh(&trace.rows.virtual_sequence).unwrap(),
                    expected.virtual_sequence
                );
                assert_eq!(
                    stream.clone_dtoh(&trace.rows.ram_address).unwrap(),
                    expected.ram_address
                );
                assert_eq!(
                    stream.clone_dtoh(&trace.rows.extras).unwrap(),
                    expected.extras
                );
            }
            assert!(DeviceTrace::upload_source(
                stream,
                &source,
                MIN_PINNED_CYCLES..cycles + 1,
                backend.program_preprocessing(),
            )
            .is_err());
        });
    }

    #[test]
    fn source_upload_preserves_rows_across_staging_reuse() {
        struct Source<'a> {
            rows: &'a [TraceRow],
            env: WitnessEnv<'a>,
            chunk_size: usize,
        }

        impl RowSource for Source<'_> {
            fn visit_chunks(
                &self,
                range: Range<usize>,
                chunk_size: usize,
                visitor: &mut ChunkVisitor<'_>,
            ) -> Result<(), WitnessError> {
                let mut end = range.start;
                for chunk in self.rows[range].chunks(chunk_size.min(self.chunk_size)) {
                    end += chunk.len();
                    visitor(chunk, self.rows.get(end), &self.env)?;
                }
                Ok(())
            }
        }

        with_sample_backend(|backend| {
            let stream = CudaContext::new(0).unwrap().default_stream();
            let rows: Vec<_> = (0..MIN_PINNED_CYCLES + 41)
                .map(|index| {
                    if index % 17 == 0 {
                        return TraceRow::default();
                    }
                    TraceRow::new(
                        JoltInstructionRow {
                            instruction_kind: JoltInstructionKind::ADDI,
                            address: 0x8000_0000 + 4 * index,
                            operands: NormalizedOperands {
                                rd: Some(1),
                                rs1: Some(2),
                                imm: index as i128,
                                ..Default::default()
                            },
                            virtual_sequence_remaining: Some((index % 31) as u16),
                            ..Default::default()
                        },
                        RegisterState {
                            rs1: Some(RegisterRead {
                                register: 2,
                                value: index as u64,
                            }),
                            rd: Some(RegisterWrite {
                                register: 1,
                                pre_value: index as u64 + 1,
                                post_value: 2 * index as u64,
                            }),
                            ..Default::default()
                        },
                        RamAccess::NoOp,
                        index as u32 + 1,
                    )
                    .unwrap()
                })
                .collect();
            let range = 17..rows.len() - 5;
            // Full chunks exercise asynchronous copies and reuse of the first
            // buffer; shorter chunks exercise the RowSource chunk-size contract.
            for chunk_size in [STAGING_CYCLES, STAGING_CYCLES - 17] {
                let source = Source {
                    rows: &rows,
                    env: WitnessEnv::new(backend.program_preprocessing()),
                    chunk_size,
                };
                let trace = DeviceTrace::upload_source(
                    Arc::clone(&stream),
                    &source,
                    range.clone(),
                    backend.program_preprocessing(),
                )
                .unwrap();
                assert_eq!(trace.cycles, range.len());
                let is_noop = stream.clone_dtoh(&trace.rows.is_noop).unwrap();
                let address = stream.clone_dtoh(&trace.rows.address).unwrap();
                let virtual_sequence = stream.clone_dtoh(&trace.rows.virtual_sequence).unwrap();
                let ram_address = stream.clone_dtoh(&trace.rows.ram_address).unwrap();
                let extras = stream.clone_dtoh(&trace.rows.extras).unwrap();
                for (index, row) in rows[range.clone()].iter().enumerate() {
                    assert_eq!(is_noop[index], u8::from(row.is_noop()));
                    assert_eq!(address[index], row.unexpanded_pc());
                    assert_eq!(
                        virtual_sequence[index],
                        row.virtual_sequence_remaining()
                            .map_or(NO_SEQUENCE, u32::from),
                    );
                    assert_eq!(ram_address[index], RAM_NO_ACCESS);
                    let words =
                        &extras[index * PACKED_EXTRA_WORDS..(index + 1) * PACKED_EXTRA_WORDS];
                    assert_eq!(words[EXTRA_BYTECODE_PC], row.pc());
                    assert_eq!(words[EXTRA_RS1], row.rs1_value());
                    assert_eq!(words[EXTRA_RD_POST], row.rd_write_value());
                }
            }
        });
    }
}
