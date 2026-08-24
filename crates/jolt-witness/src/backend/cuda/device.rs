use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions, Ptx};
use jolt_program::execution::TraceRow;
use jolt_program::preprocess::{bytecode::BytecodePCMapper, JoltProgramPreprocessing};

use super::packed::{PackedTrace, EXTRA_WORDS as PACKED_EXTRA_WORDS};
use super::tables::kind_tables;
use super::{DeviceAtomColumns, DeviceTraceColumn, HotSource, NarrowColumn};
use crate::witnesses::RaChunkSelector;
use crate::{WitnessError, JOLT_VM_LABEL};

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
            .map_err(|error: cudarc::driver::DriverError| error.to_string())
    };
    Ok(ExtractFunctions {
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

pub struct DeviceTrace {
    stream: Arc<CudaStream>,
    functions: &'static ExtractFunctions,
    cycles: usize,
    rows: DeviceRows,
    pc_map: DevicePcMap,
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

struct DevicePcMap {
    buckets: u32,
    bucket_offsets: CudaSlice<u32>,
    sequences: CudaSlice<u32>,
    values: CudaSlice<u64>,
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

struct PcMapCsr {
    bucket_offsets: Vec<u32>,
    sequences: Vec<u32>,
    values: Vec<u64>,
}

fn device_error(reason: impl core::fmt::Display) -> WitnessError {
    WitnessError::InvalidWitnessData {
        label: JOLT_VM_LABEL,
        reason: reason.to_string(),
    }
}

fn pc_map_csr(preprocessing: &JoltProgramPreprocessing) -> Result<PcMapCsr, WitnessError> {
    let bytecode = &preprocessing.bytecode;
    let max_address = bytecode
        .bytecode
        .iter()
        .map(|instruction| instruction.address)
        .max()
        .unwrap_or(0);
    let buckets = if max_address == 0 {
        1
    } else {
        BytecodePCMapper::get_index(max_address)
            .checked_add(1)
            .ok_or_else(|| device_error("bytecode bucket count overflows"))?
    };

    let mut per_bucket: Vec<Vec<(u32, u64)>> = vec![Vec::new(); buckets];
    for instruction in &bytecode.bytecode {
        if instruction.address == 0 {
            continue;
        }
        let bucket = BytecodePCMapper::get_index(instruction.address);
        let sequence = instruction.virtual_sequence_remaining.unwrap_or(0);
        let pc = bytecode
            .pc_map
            .get_pc(instruction.address, sequence)
            .ok_or_else(|| {
                device_error(format!(
                    "bytecode address {:#x} sequence {sequence} has no PC mapping",
                    instruction.address
                ))
            })?;
        per_bucket
            .get_mut(bucket)
            .ok_or_else(|| device_error(format!("bytecode bucket {bucket} exceeds {buckets}")))?
            .push((u32::from(sequence), pc as u64));
    }

    let entries = per_bucket.iter().map(Vec::len).sum::<usize>();
    let mut csr = PcMapCsr {
        bucket_offsets: Vec::with_capacity(buckets + 1),
        sequences: Vec::with_capacity(entries),
        values: Vec::with_capacity(entries),
    };
    for bucket in &per_bucket {
        csr.bucket_offsets.push(
            u32::try_from(csr.sequences.len())
                .map_err(|_| device_error("bytecode PC map exceeds a 32-bit entry count"))?,
        );
        for &(sequence, pc) in bucket {
            csr.sequences.push(sequence);
            csr.values.push(pc);
        }
    }
    csr.bucket_offsets.push(
        u32::try_from(csr.sequences.len())
            .map_err(|_| device_error("bytecode PC map exceeds a 32-bit entry count"))?,
    );
    Ok(csr)
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
        let domain = cycles;
        let cycles = len;
        let functions = extract_functions(&stream)?;
        let csr = pc_map_csr(preprocessing)?;
        let buckets = u32::try_from(csr.bucket_offsets.len().saturating_sub(1))
            .map_err(|_| device_error("bytecode PC map exceeds a 32-bit bucket count"))?;
        let tables = kind_tables()?;

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

        let mut staging = PackedTrace::with_capacity(cycles);
        staging.fill_range(rows, domain, base, cycles);
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

        let pc_map = DevicePcMap {
            buckets,
            bucket_offsets: stream
                .clone_htod(&csr.bucket_offsets)
                .map_err(device_error)?,
            sequences: stream.clone_htod(&csr.sequences).map_err(device_error)?,
            values: stream.clone_htod(&csr.values).map_err(device_error)?,
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
            pc_map,
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
        let pc_map = self.pc_map.bucket_offsets.len() * size_of::<u32>()
            + self.pc_map.sequences.len() * size_of::<u32>()
            + self.pc_map.values.len() * size_of::<u64>();
        let kinds = self.kinds.input.len()
            + self.kinds.operand.len()
            + self.kinds.output.len()
            + self.kinds.index.len()
            + self.kinds.flags.len() * size_of::<u32>()
            + self.kinds.table_index.len() * size_of::<u32>();
        rows + pc_map + kinds
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
        let ram_start = common::constants::RAM_START_ADDRESS;
        let alignment = common::constants::ALIGNMENT_FACTOR_BYTECODE as u64;

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
        let _ = builder.arg(&self.pc_map.bucket_offsets);
        let _ = builder.arg(&self.pc_map.sequences);
        let _ = builder.arg(&self.pc_map.values);
        let _ = builder.arg(&self.pc_map.buckets);
        let _ = builder.arg(&ram_start);
        let _ = builder.arg(&alignment);
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
        // `i + 1 < cycles`. Bucket reads are bounds-checked against
        // `pc_buckets`, and `bucket_offsets` holds `pc_buckets + 1` entries so
        // `bucket + 1` is in range. The kind tables, including the four mode
        // tables, are bounds-checked against `count` and each holds `count`
        // entries. `unmapped` is written only by `atomicExch`. Every buffer is
        // a distinct allocation.
        let _ = unsafe { builder.launch(launch_config(count)) }.map_err(device_error)?;

        self.check_mapped(&unmapped, "atom columns")?;
        Ok(columns)
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
            // and it is mutated only through `atomicMax`. Every buffer is a
            // distinct allocation.
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
        let buckets = self.pc_map.buckets;
        let ram_start = common::constants::RAM_START_ADDRESS;
        let alignment = common::constants::ALIGNMENT_FACTOR_BYTECODE as u64;

        let mut builder = self.stream.launch_builder(&self.functions.mapped_pc);
        let _ = builder.arg(&self.rows.is_noop);
        let _ = builder.arg(&self.rows.address);
        let _ = builder.arg(&self.rows.virtual_sequence);
        let _ = builder.arg(&self.pc_map.bucket_offsets);
        let _ = builder.arg(&self.pc_map.sequences);
        let _ = builder.arg(&self.pc_map.values);
        let _ = builder.arg(&buckets);
        let _ = builder.arg(&ram_start);
        let _ = builder.arg(&alignment);
        let _ = builder.arg(&mut out);
        let _ = builder.arg(&mut rejected);
        let _ = builder.arg(&count);
        // SAFETY: thread `i < count` writes only `out[i]` and reads only index
        // `i` of the four row arrays, all of which hold `cycles >= count`
        // elements. Bucket reads are bounds-checked against `buckets` in the
        // kernel, and `bucket_offsets` holds `buckets + 1` entries so
        // `bucket + 1` is in range. `rejected` is a single element mutated only
        // through `atomicMin`. Every buffer is a distinct allocation.
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
