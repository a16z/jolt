//! Upload and extraction on a caller-owned CUDA stream.
//!
//! NVRTC compiles the extraction module on first use, and function handles are
//! cached per device. Each `DeviceTrace` owns its packed window and instruction
//! tables; it does not cache witness rows globally. Mapping kernels report
//! unrepresentable indices through a device rejection flag checked on the host.

use super::packed::EXTRA_KIND_BITS;
use core::fmt::Display;
use cudarc::driver::DriverError;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions, Ptx};
use jolt_program::preprocess::JoltProgramPreprocessing;
use jolt_riscv::JoltTraceRow as TraceRow;

use super::packed::{PackedTrace, EXTRA_WORDS as PACKED_EXTRA_WORDS};
use super::tables::kind_tables;
use super::{DeviceAtomColumns, DeviceTraceColumn};
use crate::{RowSource, WitnessError, JOLT_VM_LABEL};

pub const COLD: u32 = u32::MAX;

const EXTRACT_SRC: &str = concat!(
    include_str!("kernels/extract.cu"),
    include_str!("kernels/atoms.cu")
);

const BLOCK: u32 = 256;

const NO_REJECTION: u64 = u64::MAX;

struct ExtractFunctions {
    mapped_pc: CudaFunction,
    remapped_ram: CudaFunction,
    lookup_index: CudaFunction,
    atom_columns: CudaFunction,
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
        mapped_pc: function("mapped_pc_words_kernel")?,
        remapped_ram: function("remapped_ram_words_kernel")?,
        lookup_index: function("lookup_index_limbs_kernel")?,
        atom_columns: function("atom_columns_kernel")?,
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
        Self::upload_packed(
            stream,
            PackedTrace::from_source(source, range)?,
            preprocessing,
        )
    }

    fn upload_packed(
        stream: Arc<CudaStream>,
        staging: PackedTrace,
        preprocessing: &JoltProgramPreprocessing,
    ) -> Result<Self, WitnessError> {
        let cycles = staging.cycles;
        let functions = extract_functions(&stream)?;
        let tables = kind_tables()?;
        tables.require_supported(
            staging
                .extras
                .chunks_exact(PACKED_EXTRA_WORDS)
                .map(|row| row[EXTRA_KIND_BITS] as u16),
        )?;

        let alloc_u8 = |len: usize| stream.alloc_zeros::<u8>(len).map_err(device_error);
        let alloc_u32 = |len: usize| stream.alloc_zeros::<u32>(len).map_err(device_error);
        let alloc_u64 = |len: usize| stream.alloc_zeros::<u64>(len).map_err(device_error);
        let mut device = DeviceRows {
            is_noop: alloc_u8(cycles)?,
            address: alloc_u64(cycles)?,
            virtual_sequence: alloc_u32(cycles)?,
            ram_address: alloc_u64(cycles)?,
            extras: alloc_u64(cycles * PACKED_EXTRA_WORDS)?,
        };

        stream
            .memcpy_htod(&staging.is_noop, &mut device.is_noop.slice_mut(..))
            .map_err(device_error)?;
        stream
            .memcpy_htod(&staging.address, &mut device.address.slice_mut(..))
            .map_err(device_error)?;
        stream
            .memcpy_htod(
                &staging.virtual_sequence,
                &mut device.virtual_sequence.slice_mut(..),
            )
            .map_err(device_error)?;
        stream
            .memcpy_htod(&staging.ram_address, &mut device.ram_address.slice_mut(..))
            .map_err(device_error)?;
        stream
            .memcpy_htod(&staging.extras, &mut device.extras.slice_mut(..))
            .map_err(device_error)?;
        drop(staging);

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
        let count = self.count()?;

        let alloc32 = |len: usize| self.stream.alloc_zeros::<u32>(len).map_err(device_error);
        let alloc64 = |len: usize| self.stream.alloc_zeros::<u64>(len).map_err(device_error);
        let mut columns = DeviceAtomColumns {
            flags: alloc32(self.cycles)?,
            table_index: alloc32(self.cycles)?,
            bytecode_pc: alloc64(self.cycles)?,
            rd_pre_value: alloc64(self.cycles)?,
            rs1_address: alloc32(self.cycles)?,
            rs2_address: alloc32(self.cycles)?,
            rd_address: alloc32(self.cycles)?,
            rd_inc: alloc64(self.cycles * 2)?,
            ram_inc: alloc64(self.cycles * 2)?,
            left_instruction_input: alloc64(self.cycles)?,
            right_instruction_input: alloc64(self.cycles * 2)?,
            left_lookup_operand: alloc64(self.cycles)?,
            right_lookup_operand: alloc64(self.cycles * 2)?,
            lookup_output: alloc64(self.cycles)?,
            product_magnitude: alloc64(self.cycles * 2)?,
        };
        let mut unmapped = self.unmapped_flag()?;

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
        let _ = builder.arg(&mut columns.flags);
        let _ = builder.arg(&mut columns.table_index);
        let _ = builder.arg(&mut columns.bytecode_pc);
        let _ = builder.arg(&mut columns.rd_pre_value);
        let _ = builder.arg(&mut columns.rs1_address);
        let _ = builder.arg(&mut columns.rs2_address);
        let _ = builder.arg(&mut columns.rd_address);
        let _ = builder.arg(&mut columns.rd_inc);
        let _ = builder.arg(&mut columns.ram_inc);
        let _ = builder.arg(&mut columns.left_instruction_input);
        let _ = builder.arg(&mut columns.right_instruction_input);
        let _ = builder.arg(&mut columns.left_lookup_operand);
        let _ = builder.arg(&mut columns.right_lookup_operand);
        let _ = builder.arg(&mut columns.lookup_output);
        let _ = builder.arg(&mut columns.product_magnitude);
        let _ = builder.arg(&mut unmapped);
        let _ = builder.arg(&count);
        // SAFETY: thread `i < count` writes exactly index `i` of each
        // single-word output (all allocated at `cycles`) and `2*i`, `2*i + 1`
        // of the four two-limb outputs (allocated at `cycles * 2`). It reads
        // index `i` of the row arrays and the `EXTRA_WORDS` consecutive words
        // at `i * EXTRA_WORDS`, plus `is_noop[i + 1]` guarded by
        // `i + 1 < cycles`. The kind tables, including the four mode
        // tables, are bounds-checked against `count` and each holds `count`
        // entries. `unmapped` is written only by `atomicExch`. Every buffer is
        // a distinct allocation.
        let _ = unsafe { builder.launch(launch_config(count)) }.map_err(device_error)?;

        self.check_mapped(&unmapped, "atom columns")?;
        Ok(columns)
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
