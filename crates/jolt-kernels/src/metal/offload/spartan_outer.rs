//! Metal spartan-outer witness-row production: the uni-skip GPU front, the
//! resident row planes, and the fused Stage-1 owner co-production that fills
//! the InstructionReadRAF/bytecode/RAM owner planes in the same pass.

use jolt_field::AkitaField;
use jolt_poly::EqPolynomial;
use jolt_riscv::InterleavedBitsMarker;
use jolt_sumcheck::SumcheckError;
use jolt_witness::witnesses::SpartanOuterRow;
use jolt_witness::witnesses::{
    Extract, FusedInc, InstructionRafFlag, LookupIndex, MappedPc, RamHammingWeight, RamInc,
    RamReadValue as Stage1RamReadValue, RamWriteValue as Stage1RamWriteValue, RemappedRamAddress,
    TableIndex, WitnessEnv,
};
use jolt_witness::WitnessBundle;
use jolt_witness::{JoltWitnessPlane, WitnessError};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::ram_trace::{
    RamAccessCollection, RamAccessCollectionChunkWriter, RamAccessCollectionStorage,
};
use crate::metal::solinas::bytecode_read_raf_address::{
    BytecodeAddressStage1TopologyChunkWriter, BytecodeAddressStage1TopologyOwner,
    BytecodeAddressStage1TopologyScratch, BytecodeAddressStage1TopologyStorage,
};
use crate::metal::solinas::spartan_shift::{
    SpartanShiftFlagWord, SpartanShiftResidentRows, SPARTAN_SHIFT_FLAG_ROWS_PER_WORD,
};
use crate::metal::solinas::{
    instruction_input_row_bytes, instruction_read_raf_claim_and_count_rank,
    spartan_outer_uniskip_residual_row_bytes, BooleanityRow, InstructionInputRow,
    InstructionInputRows, InstructionReadRafStage1ChunkWriter, InstructionReadRafStage1Owner,
    InstructionReadRafStage1Storage, MetalError, RegistersValInstructionSourceRequest,
    SolinasMetal, SpartanOuterUniskipConfig, SpartanOuterUniskipResidualRow,
    SpartanOuterUniskipRow, SpartanOuterUniskipRows, INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS,
};
use crate::optimized::ram_trace::RamAccessBundle;
use crate::optimized::spartan_outer::{
    extension_coefficients, RowsStore, SpartanOuterCarry, EXTENDED_SIZE,
};
use crate::{KernelError, ProofSession};

impl RowsStore {
    fn production_source_kind(&self) -> &'static str {
        match self {
            Self::Owned(_) => "owned_random_access",
            Self::Retained(_) => "retained_host_repack",
        }
    }

    fn host_repack_rows(&self) -> usize {
        match self {
            Self::Owned(_) => 0,
            Self::Retained(rows) => rows.len(),
        }
    }

    fn explicit_rows(&self) -> usize {
        match self {
            Self::Owned(rows) => rows.physical_rows().min(rows.cycles()),
            Self::Retained(rows) => rows.len(),
        }
    }
}

#[derive(Clone, Copy, Debug, WitnessBundle)]
struct Stage1InstructionFacts {
    lookup_index: LookupIndex,
    table_index: TableIndex,
    raf_flag: InstructionRafFlag,
    mapped_pc: MappedPc,
    remapped_ram_address: RemappedRamAddress,
    fused_inc: FusedInc,
}

#[derive(Clone, Copy, Debug)]
struct Stage1ProjectionRow {
    outer: SpartanOuterRow,
    instruction: Stage1InstructionFacts,
    ram_access: RamAccessBundle,
    register_write: Option<(u8, u64, u64)>,
}

impl WitnessBundle for Stage1ProjectionRow {
    fn from_row(
        row: &jolt_riscv::JoltTraceRow,
        next: Option<&jolt_riscv::JoltTraceRow>,
        env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        let outer = SpartanOuterRow::from_row(row, next, env)?;
        let raf = !row.circuit_flags().is_interleaved_operands();
        let lookup_index = if raf {
            outer.right_lookup_operand.0
        } else {
            jolt_lookup_tables::interleave_bits(
                outer.left_lookup_operand.0,
                outer.right_lookup_operand.0 as u64,
            )
        };
        let remapped_ram_address = RemappedRamAddress::extract(row, next, env)?;
        let ram_inc = RamInc::extract(row, next, env)?;
        let fused_inc = if row.is_store() {
            ram_inc.0
        } else if row.rd_index().is_some() {
            i128::from(row.rd_write_value()) - i128::from(row.rd_pre_value())
        } else {
            0
        };
        Ok(Self {
            outer,
            instruction: Stage1InstructionFacts {
                lookup_index: LookupIndex(lookup_index),
                table_index: TableIndex::extract(row, next, env)?,
                raf_flag: InstructionRafFlag(raf),
                mapped_pc: MappedPc(Some(row.pc() as usize)),
                remapped_ram_address,
                fused_inc: FusedInc(fused_inc),
            },
            ram_access: RamAccessBundle {
                address: remapped_ram_address,
                pre_value: Stage1RamReadValue(row.ram_read_value()),
                post_value: Stage1RamWriteValue(row.ram_write_value()),
                ram_inc,
                ram_hamming_weight: RamHammingWeight::extract(row, next, env)?,
            },
            register_write: row
                .rd_index()
                .map(|register| (register, row.rd_pre_value(), row.rd_write_value())),
        })
    }

    fn annotated_ids() -> Vec<jolt_claims::protocols::jolt::JoltPolynomialId> {
        SpartanOuterRow::annotated_ids()
    }
}

fn pack_stage1_instruction_source(
    facts: Stage1InstructionFacts,
) -> Result<(BooleanityRow, u8, bool), MetalError> {
    let mapped_pc = facts.mapped_pc.0.map(|pc| pc as u64);
    let row = BooleanityRow::new(
        facts.lookup_index.0,
        mapped_pc,
        facts.remapped_ram_address.0,
        facts.fused_inc.0,
    )?;
    let table_plus_one = facts
        .table_index
        .0
        .map_or(Some(0), |table| table.checked_add(1))
        .and_then(|table| u8::try_from(table).ok())
        .ok_or(MetalError::InvalidInstructionReadRafGrouped(
            "lookup table index cannot be encoded by the Stage-1 owner".to_owned(),
        ))?;
    let _ = instruction_read_raf_claim_and_count_rank(table_plus_one, facts.raf_flag.0).ok_or(
        MetalError::InvalidInstructionReadRafGrouped(
            "lookup table index exceeds the InstructionReadRAF table domain".to_owned(),
        ),
    )?;
    Ok((row, table_plus_one, facts.raf_flag.0))
}

#[derive(Clone, Copy)]
struct PackedStage1PaddingRow {
    instruction_input: InstructionInputRow,
    residual: SpartanOuterUniskipResidualRow,
    instruction_source: BooleanityRow,
    table_plus_one: u8,
    raf: bool,
    unexpanded_pc: u64,
    pc: u64,
    shift_flags: SpartanShiftFlagWord,
    ram_access: RamAccessBundle,
}

#[derive(Clone, Copy)]
struct Stage1PaddingRows {
    regular: Option<PackedStage1PaddingRow>,
    terminal: Option<PackedStage1PaddingRow>,
}

impl Stage1PaddingRows {
    fn new(
        access: &jolt_witness::RandomAccessRows<'_>,
        explicit_rows: usize,
        cycles: usize,
    ) -> Result<Self, MetalError> {
        let regular = (explicit_rows + 1 < cycles)
            .then(|| pack_stage1_padding_row(access, explicit_rows))
            .transpose()?;
        let terminal = (explicit_rows < cycles)
            .then(|| pack_stage1_padding_row(access, cycles - 1))
            .transpose()?;
        Ok(Self { regular, terminal })
    }

    const fn source_window_count(self, explicit_rows: usize) -> usize {
        explicit_rows + self.regular.is_some() as usize + self.terminal.is_some() as usize
    }
}

#[derive(Clone, Copy)]
struct Stage1ChunkParts {
    physical: usize,
    regular_padding: usize,
    terminal_padding: usize,
}

fn stage1_chunk_parts(
    chunk_start: usize,
    chunk_len: usize,
    explicit_rows: usize,
    cycles: usize,
) -> Stage1ChunkParts {
    let physical = explicit_rows.saturating_sub(chunk_start).min(chunk_len);
    let terminal_padding = usize::from(
        explicit_rows < cycles && chunk_start + chunk_len == cycles && physical < chunk_len,
    );
    Stage1ChunkParts {
        physical,
        regular_padding: chunk_len - physical - terminal_padding,
        terminal_padding,
    }
}

fn pack_stage1_padding_row(
    access: &jolt_witness::RandomAccessRows<'_>,
    row_index: usize,
) -> Result<PackedStage1PaddingRow, MetalError> {
    let projected: Stage1ProjectionRow =
        access
            .window(row_index)
            .map_err(|error| MetalError::SpartanOuterRowExtraction {
                row: row_index,
                message: error.to_string(),
            })?;
    let packed = SpartanOuterUniskipRow::from_spartan_outer(&projected.outer);
    let (instruction_input, residual) = packed.split();
    let (instruction_source, table_plus_one, raf) =
        pack_stage1_instruction_source(projected.instruction)?;
    let full_mask = |value: bool| if value { u32::MAX } else { 0 };
    Ok(PackedStage1PaddingRow {
        instruction_input,
        residual,
        instruction_source,
        table_plus_one,
        raf,
        unexpanded_pc: projected.outer.unexpanded_pc.0,
        pc: projected.outer.pc.0,
        shift_flags: SpartanShiftFlagWord {
            is_virtual: full_mask(projected.outer.virtual_instruction.0),
            is_first_in_sequence: full_mask(projected.outer.is_first_in_sequence.0),
            is_noop: full_mask(projected.outer.is_noop.0),
        },
        ram_access: projected.ram_access,
    })
}

fn fill_stage1_outer_padding(
    instruction_input: &mut [InstructionInputRow],
    residual: &mut [SpartanOuterUniskipResidualRow],
    start: usize,
    count: usize,
    padding: &PackedStage1PaddingRow,
) {
    let end = start + count;
    instruction_input[start..end].fill(padding.instruction_input);
    residual[start..end].fill(padding.residual);
}

fn fill_stage1_shift_padding(
    unexpanded_pc: &mut [u64],
    pc: &mut [u64],
    flags: &mut [SpartanShiftFlagWord],
    start: usize,
    count: usize,
    padding: &PackedStage1PaddingRow,
) {
    let end = start + count;
    unexpanded_pc[start..end].fill(padding.unexpanded_pc);
    pc[start..end].fill(padding.pc);
    for (word, flag_word) in flags
        .iter_mut()
        .enumerate()
        .take(end.div_ceil(SPARTAN_SHIFT_FLAG_ROWS_PER_WORD))
        .skip(start / SPARTAN_SHIFT_FLAG_ROWS_PER_WORD)
    {
        let low = start.saturating_sub(word * SPARTAN_SHIFT_FLAG_ROWS_PER_WORD);
        let high = end
            .saturating_sub(word * SPARTAN_SHIFT_FLAG_ROWS_PER_WORD)
            .min(SPARTAN_SHIFT_FLAG_ROWS_PER_WORD);
        let low_mask = u32::MAX.checked_shl(low as u32).unwrap_or(0);
        let high_mask = u32::MAX
            .checked_shr((SPARTAN_SHIFT_FLAG_ROWS_PER_WORD - high) as u32)
            .unwrap_or(0);
        let mask = low_mask & high_mask;
        let merge = |current: u32, value: u32| (current & !mask) | (value & mask);
        flag_word.is_virtual = merge(flag_word.is_virtual, padding.shift_flags.is_virtual);
        flag_word.is_first_in_sequence = merge(
            flag_word.is_first_in_sequence,
            padding.shift_flags.is_first_in_sequence,
        );
        flag_word.is_noop = merge(flag_word.is_noop, padding.shift_flags.is_noop);
    }
}

struct Stage1OwnerChunkWriters<'borrow, 'instruction, 'bytecode, 'ram> {
    instruction: &'borrow mut InstructionReadRafStage1ChunkWriter<'instruction>,
    bytecode: Option<&'borrow mut BytecodeAddressStage1TopologyChunkWriter<'bytecode>>,
    ram_access: Option<&'borrow mut RamAccessCollectionChunkWriter<'ram>>,
}

impl Stage1OwnerChunkWriters<'_, '_, '_, '_> {
    fn len(&self) -> usize {
        self.instruction.len()
    }

    fn push(
        &mut self,
        row_index: usize,
        explicit_rows: usize,
        instruction: Stage1InstructionFacts,
        ram_access: RamAccessBundle,
        register_write: Option<(u8, u64, u64)>,
        bytecode_scratch: &mut BytecodeAddressStage1TopologyScratch,
    ) -> Result<(), MetalError> {
        let (row, table_plus_one, raf) = pack_stage1_instruction_source(instruction)?;
        if let Some(topology) = self.bytecode.as_mut() {
            let rank = if row_index < explicit_rows {
                topology.record(bytecode_scratch, instruction.mapped_pc.0.unwrap_or(0))?
            } else {
                0
            };
            self.instruction.push_with_register_write(
                row,
                table_plus_one,
                raf,
                rank,
                register_write,
            )?;
        } else {
            self.instruction.push_with_register_write(
                row,
                table_plus_one,
                raf,
                0,
                register_write,
            )?;
        }
        if let Some(writer) = self.ram_access.as_mut() {
            writer.push(ram_access).map_err(|error| {
                MetalError::InvalidRamAccessCollection(error.reason().to_owned())
            })?;
        }
        Ok(())
    }

    fn fill_padding(
        &mut self,
        padding: &PackedStage1PaddingRow,
        count: usize,
    ) -> Result<(), MetalError> {
        self.instruction.fill_repeated_with_register_write(
            padding.instruction_source,
            padding.table_plus_one,
            padding.raf,
            0,
            None,
            count,
        )?;
        if let Some(writer) = self.ram_access.as_mut() {
            writer
                .fill_repeated(padding.ram_access, count)
                .map_err(|error| {
                    MetalError::InvalidRamAccessCollection(error.reason().to_owned())
                })?;
        }
        Ok(())
    }

    fn finish(
        &mut self,
        bytecode_scratch: &mut BytecodeAddressStage1TopologyScratch,
    ) -> Result<(), MetalError> {
        if let Some(topology) = self.bytecode.as_mut() {
            topology.finish(bytecode_scratch)?;
        }
        if let Some(writer) = self.ram_access.as_mut() {
            writer.finish().map_err(|error| {
                MetalError::InvalidRamAccessCollection(error.reason().to_owned())
            })?;
        }
        Ok(())
    }
}

fn run_stage1_owner_chunks<'instruction, 'bytecode, 'ram, R>(
    instruction: &mut [InstructionReadRafStage1ChunkWriter<'instruction>],
    bytecode: Option<&mut [BytecodeAddressStage1TopologyChunkWriter<'bytecode>]>,
    ram_access: Option<&mut [RamAccessCollectionChunkWriter<'ram>]>,
    fill: impl FnOnce(
        &mut [Stage1OwnerChunkWriters<'_, 'instruction, 'bytecode, 'ram>],
    ) -> Result<R, MetalError>,
) -> Result<R, MetalError> {
    if bytecode
        .as_ref()
        .is_some_and(|writers| writers.len() != instruction.len())
    {
        return Err(MetalError::InvalidInstructionReadRafGrouped(
            "Stage-1 owner chunk counts disagree".to_owned(),
        ));
    }
    if ram_access
        .as_ref()
        .is_some_and(|writers| writers.len() != instruction.len())
    {
        return Err(MetalError::InvalidRamAccessCollection(
            "Stage-1 RAM chunk counts disagree".to_owned(),
        ));
    }
    let mut chunks: Vec<Stage1OwnerChunkWriters<'_, 'instruction, 'bytecode, 'ram>> = match bytecode
    {
        Some(bytecode) => instruction
            .iter_mut()
            .zip(bytecode)
            .map(|(instruction, bytecode)| Stage1OwnerChunkWriters {
                instruction,
                bytecode: Some(bytecode),
                ram_access: None,
            })
            .collect(),
        None => instruction
            .iter_mut()
            .map(|instruction| Stage1OwnerChunkWriters {
                instruction,
                bytecode: None,
                ram_access: None,
            })
            .collect(),
    };
    if let Some(ram_access) = ram_access {
        for (chunk, ram_access) in chunks.iter_mut().zip(ram_access) {
            chunk.ram_access = Some(ram_access);
        }
    }
    fill(&mut chunks)
}

fn with_stage1_owner_chunks<R>(
    instruction: &mut InstructionReadRafStage1Storage,
    bytecode: Option<&mut BytecodeAddressStage1TopologyStorage>,
    ram_access: Option<&mut RamAccessCollectionStorage>,
    fill: impl FnOnce(&mut [Stage1OwnerChunkWriters<'_, '_, '_, '_>]) -> Result<R, MetalError>,
) -> Result<R, MetalError> {
    instruction.with_chunk_writers(|instruction| match ram_access {
        Some(ram_access) => ram_access.with_chunk_writers(|ram_access| match bytecode {
            Some(bytecode) => bytecode.with_chunk_writers(|bytecode| {
                run_stage1_owner_chunks(instruction, Some(bytecode), Some(ram_access), fill)
            }),
            None => run_stage1_owner_chunks(instruction, None, Some(ram_access), fill),
        }),
        None => match bytecode {
            Some(bytecode) => bytecode.with_chunk_writers(|bytecode| {
                run_stage1_owner_chunks(instruction, Some(bytecode), None, fill)
            }),
            None => run_stage1_owner_chunks(instruction, None, None, fill),
        },
    })
}

pub(crate) fn prepare_metal_spartan_outer_uniskip(
    context: &SolinasMetal,
    config: SpartanOuterUniskipConfig,
    session: &mut ProofSession,
    log_t: usize,
    tau: &[AkitaField],
    witness: &dyn JoltWitnessPlane<AkitaField>,
) -> Result<(), KernelError<AkitaField>> {
    if tau.len() != log_t + 2 {
        return Err(KernelError::InvariantViolation {
            reason: "Spartan outer tau must carry log_t + 2 challenges",
        });
    }
    let cycles = 1usize << log_t;
    let rows = RowsStore::resolve(witness, cycles)?;
    let (tau_low, _) = tau.split_at(log_t + 1);
    let split = tau_low.len() / 2;
    let (out_point, in_point) = tau_low.split_at(split);
    let e_out = EqPolynomial::<AkitaField>::evals(out_point, None);
    let e_in = EqPolynomial::<AkitaField>::evals(in_point, None);
    let (extended, resident) = {
        let explicit_rows = rows.explicit_rows();
        let resident = {
            let _span = tracing::info_span!("MetalSpartanOuterUniskip::row_handoff").entered();
            match session.take::<SpartanOuterUniskipRows>() {
                Some(resident)
                    if resident.len() == cycles && resident.explicit_rows() == explicit_rows =>
                {
                    resident
                }
                _ => prepare_metal_spartan_outer_rows(context, &rows, cycles)?,
            }
        };
        let compact_rows_storage_id = resident.instruction_input_allocation_identity();
        let residual_rows_storage_id = resident.allocation_identity();
        let _handoff = tracing::info_span!(
            "MetalInstructionInput::compact_rows_stage1_handoff",
            compact_rows_storage_id,
            residual_rows_storage_id,
            resident_rows = cycles,
            explicit_rows,
            compact_row_bytes = 48,
            residual_row_bytes = 112,
            full_domain_copy_bytes = 0,
            full_domain_copy_dispatches = 0,
            host_repack_rows = 0,
        )
        .entered();
        let invocation = context
            .prepare_spartan_outer_uniskip_with_rows(&resident, &e_in, &e_out, config)
            .map_err(metal_outer_error)?;
        {
            let dispatch_span = tracing::info_span!(
                "MetalSpartanOuterUniskip::dispatch",
                gpu_active_ns = tracing::field::Empty,
            );
            let _dispatch = dispatch_span.enter();
            let gpu_active = invocation.execute_timed().map_err(metal_outer_error)?;
            let gpu_active_ns = u64::try_from(gpu_active.as_nanos()).unwrap_or(u64::MAX);
            let _ = dispatch_span.record("gpu_active_ns", gpu_active_ns);
        }
        let output = invocation.read_output().map_err(metal_outer_error)?;
        drop(invocation);
        (output, resident)
    };
    session.park(resident);
    let mut t1_values = vec![AkitaField::zero(); EXTENDED_SIZE];
    for ((position, _), value) in extension_coefficients().iter().zip(extended) {
        t1_values[*position] = value;
    }
    session.park(SpartanOuterCarry {
        log_t,
        tau: tau.to_vec(),
        rows,
        t1_values,
    });
    Ok(())
}

pub(crate) fn take_metal_spartan_outer_tau(
    session: &mut ProofSession,
    expected_log_t: usize,
) -> Result<Vec<AkitaField>, KernelError<AkitaField>> {
    let carry =
        session
            .take::<SpartanOuterCarry<AkitaField>>()
            .ok_or(KernelError::InvariantViolation {
                reason: "Metal outer remainder found no uni-skip carry",
            })?;
    if carry.log_t != expected_log_t {
        return Err(KernelError::InvariantViolation {
            reason: "Metal outer remainder carry disagrees with relation geometry",
        });
    }
    Ok(carry.tau)
}

pub(crate) fn prepare_metal_spartan_outer_witness_rows(
    context: &SolinasMetal,
    witness: &dyn JoltWitnessPlane<AkitaField>,
    cycles: usize,
) -> Result<SpartanOuterUniskipRows, KernelError<AkitaField>> {
    let rows = RowsStore::resolve(witness, cycles)?;
    prepare_metal_spartan_outer_rows(context, &rows, cycles)
}

#[derive(Debug)]
pub(crate) enum MetalSpartanDenseRowsError {
    Kernel(KernelError<AkitaField>),
    Metal(MetalError),
}

impl MetalSpartanDenseRowsError {
    pub(crate) fn is_capacity_error(&self) -> bool {
        matches!(self, Self::Metal(error) if error.is_capacity_error())
    }

    pub(crate) fn into_kernel_error(self) -> KernelError<AkitaField> {
        match self {
            Self::Kernel(error) => error,
            Self::Metal(error) => metal_outer_error(error),
        }
    }
}

fn stage1_owner_rows_span(
    cycles: usize,
    explicit_rows: usize,
    witness_row_extractions: usize,
) -> tracing::Span {
    tracing::info_span!(
        "MetalInstructionInput::compact_rows_prepare",
        source_kind = "owned_random_access",
        witness_row_extractions,
        padding_rows_bulk_filled = cycles - explicit_rows,
        residual_rows_written = cycles,
        compact_rows_written = cycles,
        compact_row_bytes = 48,
        residual_row_bytes = 112,
        compact_allocations = 1,
        residual_allocations = 1,
        full_row_allocations = 0,
        full_domain_copy_bytes = 0,
        full_domain_copy_dispatches = 0,
        host_repack_rows = 0,
        compact_rows_storage_id = tracing::field::Empty,
        residual_rows_storage_id = tracing::field::Empty,
        resident_rows = cycles,
        explicit_rows,
    )
}

fn bytecode_stage1_topology_span(enabled: bool, physical_rows: usize) -> tracing::Span {
    tracing::info_span!(
        "MetalBytecodeReadRafAddress::fused_topology_prepare",
        enabled,
        physical_rows,
        chunk_rows = INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS,
    )
}

fn bytecode_stage1_topology_publish_span(enabled: bool, physical_rows: usize) -> tracing::Span {
    tracing::info_span!(
        "MetalBytecodeReadRafAddress::fused_topology_publish",
        enabled,
        physical_rows,
        chunk_rows = INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS,
        chunks = tracing::field::Empty,
        descriptors = tracing::field::Empty,
        descriptor_elements = tracing::field::Empty,
        descriptor_bytes = tracing::field::Empty,
        descriptor_storage_id = tracing::field::Empty,
        pivots = tracing::field::Empty,
        pivot_elements = tracing::field::Empty,
        pivot_bytes = tracing::field::Empty,
        pivot_storage_id = tracing::field::Empty,
        chunk_offset_elements = tracing::field::Empty,
        chunk_offset_bytes = tracing::field::Empty,
        chunk_offset_storage_id = tracing::field::Empty,
        work_items = tracing::field::Empty,
        work_item_elements = tracing::field::Empty,
        work_item_bytes = tracing::field::Empty,
        work_item_storage_id = tracing::field::Empty,
        address_offset_elements = tracing::field::Empty,
        address_offset_bytes = tracing::field::Empty,
        address_offset_storage_id = tracing::field::Empty,
        max_descriptors_per_chunk = tracing::field::Empty,
        max_pivots_per_chunk = tracing::field::Empty,
        first_push_pc = tracing::field::Empty,
        source_generation = tracing::field::Empty,
        source_completion_serial = tracing::field::Empty,
        source_rows_storage_id = tracing::field::Empty,
        source_claim_storage_id = tracing::field::Empty,
        topology_completion_serial = tracing::field::Empty,
        shared_source_row_scans = tracing::field::Empty,
        additional_source_row_scans = tracing::field::Empty,
        extra_source_scans = tracing::field::Empty,
        source_windows = tracing::field::Empty,
        member_upload_bytes = tracing::field::Empty,
        complete_overwrite = tracing::field::Empty,
        covered_rows = tracing::field::Empty,
    )
}

fn record_bytecode_stage1_topology_span(
    source: &InstructionReadRafStage1Owner,
    topology: Option<&BytecodeAddressStage1TopologyOwner>,
    physical_rows: usize,
) {
    let span = bytecode_stage1_topology_publish_span(topology.is_some(), physical_rows);
    let source = source.receipt();
    let _ = span.record("source_generation", source.source_generation());
    let _ = span.record("source_completion_serial", source.completion_serial());
    let _ = span.record("source_rows_storage_id", source.row_allocation_identity());
    let _ = span.record(
        "source_claim_storage_id",
        source.claim_allocation_identity(),
    );
    let _ = span.record("source_windows", source.rows());
    let _ = span.record("shared_source_row_scans", 1usize);
    let _ = span.record("additional_source_row_scans", 0usize);
    let _ = span.record("extra_source_scans", 0usize);
    let _ = span.record("member_upload_bytes", 0usize);
    let Some(topology) = topology else {
        for field in [
            "chunks",
            "descriptors",
            "descriptor_elements",
            "descriptor_bytes",
            "descriptor_storage_id",
            "pivots",
            "pivot_elements",
            "pivot_bytes",
            "pivot_storage_id",
            "chunk_offset_elements",
            "chunk_offset_bytes",
            "chunk_offset_storage_id",
            "work_items",
            "work_item_elements",
            "work_item_bytes",
            "work_item_storage_id",
            "address_offset_elements",
            "address_offset_bytes",
            "address_offset_storage_id",
            "max_descriptors_per_chunk",
            "max_pivots_per_chunk",
            "first_push_pc",
            "topology_completion_serial",
            "covered_rows",
        ] {
            let _ = span.record(field, 0usize);
        }
        let _ = span.record("complete_overwrite", false);
        let _entered = span.enter();
        return;
    };
    let receipt = topology.receipt();
    let values = [
        ("chunks", receipt.chunks()),
        ("descriptors", receipt.descriptors()),
        ("descriptor_elements", receipt.descriptor_elements()),
        ("descriptor_bytes", receipt.descriptor_bytes()),
        (
            "descriptor_storage_id",
            receipt.descriptor_allocation_identity(),
        ),
        ("pivots", receipt.pivots()),
        ("pivot_elements", receipt.pivot_elements()),
        ("pivot_bytes", receipt.pivot_bytes()),
        ("pivot_storage_id", receipt.pivot_allocation_identity()),
        ("chunk_offset_elements", receipt.chunk_offset_elements()),
        ("chunk_offset_bytes", receipt.chunk_offset_bytes()),
        (
            "chunk_offset_storage_id",
            receipt.chunk_offset_allocation_identity(),
        ),
        ("work_items", receipt.work_items()),
        ("work_item_elements", receipt.work_items()),
        ("work_item_bytes", receipt.work_item_bytes()),
        (
            "work_item_storage_id",
            receipt.work_item_allocation_identity(),
        ),
        ("address_offset_elements", receipt.address_offset_elements()),
        ("address_offset_bytes", receipt.address_offset_bytes()),
        (
            "address_offset_storage_id",
            receipt.address_offset_allocation_identity(),
        ),
        (
            "max_descriptors_per_chunk",
            receipt.max_descriptors_per_chunk(),
        ),
        ("max_pivots_per_chunk", receipt.max_pivots_per_chunk()),
        ("first_push_pc", receipt.first_push_pc()),
        (
            "topology_completion_serial",
            receipt.completion_serial() as usize,
        ),
        ("covered_rows", receipt.covered_rows()),
    ];
    for (field, value) in values {
        let _ = span.record(field, value);
    }
    let _ = span.record("complete_overwrite", receipt.complete_overwrite());
    let _entered = span.enter();
}

pub(crate) struct InstructionReadRafStage1Ready {
    pub(crate) owner: InstructionReadRafStage1Owner,
    pub(crate) bytecode_topology: Option<BytecodeAddressStage1TopologyOwner>,
    pub(crate) registers_val: Option<RegistersValInstructionSourceRequest>,
    pub(crate) ram_access: Option<RamAccessCollection>,
}

type Stage1OwnerPreparedRows = (SpartanOuterUniskipRows, InstructionReadRafStage1Ready);

type ShiftStage1OwnerPreparedRows = (
    SpartanOuterUniskipRows,
    SpartanShiftResidentRows,
    InstructionReadRafStage1Ready,
);

pub(crate) fn prepare_metal_spartan_outer_stage1_owner_witness_rows(
    context: &SolinasMetal,
    witness: &dyn JoltWitnessPlane<AkitaField>,
    cycles: usize,
    prepare_bytecode_carrier: bool,
    prepare_registers_val: bool,
    prepare_ram_access: bool,
) -> Result<Stage1OwnerPreparedRows, MetalSpartanDenseRowsError> {
    let owned = witness
        .owned_rows()
        .filter(|rows| cycles <= rows.cycles())
        .ok_or(MetalSpartanDenseRowsError::Kernel(
            KernelError::InvariantViolation {
                reason: "InstructionReadRAF Stage-1 ownership requires a random-access witness",
            },
        ))?;
    let explicit_rows = owned.physical_rows().min(cycles);
    let access = owned.view();
    let padding = Stage1PaddingRows::new(&access, explicit_rows, cycles)
        .map_err(MetalSpartanDenseRowsError::Metal)?;
    let span = stage1_owner_rows_span(
        cycles,
        explicit_rows,
        padding.source_window_count(explicit_rows),
    );
    let _entered = span.enter();
    let topology_span = bytecode_stage1_topology_span(prepare_bytecode_carrier, explicit_rows);
    let _topology_entered = topology_span.enter();
    let mut source = context
        .prepare_instruction_read_raf_stage1_storage(cycles)
        .map_err(MetalSpartanDenseRowsError::Metal)?;
    let mut bytecode_topology = prepare_bytecode_carrier
        .then(|| context.prepare_bytecode_address_stage1_topology_storage(cycles, explicit_rows))
        .transpose()
        .map_err(MetalSpartanDenseRowsError::Metal)?;
    let mut ram_access = prepare_ram_access
        .then(|| RamAccessCollectionStorage::new(cycles, INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS))
        .transpose()
        .map_err(|error| MetalSpartanDenseRowsError::Kernel(error.into_kernel_error()))?;
    let outer_rows = context
        .prepare_spartan_outer_uniskip_rows_with_fill(cycles, |instruction_input, residual| {
            with_stage1_owner_chunks(
                &mut source,
                bytecode_topology.as_mut(),
                ram_access.as_mut(),
                |owner_chunks| {
                    let fill_chunk =
                        |chunk: usize,
                         instruction_input: &mut [InstructionInputRow],
                         residual: &mut [SpartanOuterUniskipResidualRow],
                         owner: &mut Stage1OwnerChunkWriters<'_, '_, '_, '_>,
                         bytecode_scratch: &mut BytecodeAddressStage1TopologyScratch|
                         -> Result<(), MetalError> {
                            if instruction_input.len() != owner.len()
                                || residual.len() != owner.len()
                            {
                                return Err(MetalError::InvalidInstructionReadRafGrouped(
                                    "Stage-1 owner chunks disagree on row count".to_owned(),
                                ));
                            }
                            let chunk_start = chunk * INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS;
                            let parts =
                                stage1_chunk_parts(chunk_start, owner.len(), explicit_rows, cycles);
                            for offset in 0..parts.physical {
                                let row_index = chunk_start + offset;
                                let projected: Stage1ProjectionRow = access
                                    .window(row_index)
                                    .map_err(|error| MetalError::SpartanOuterRowExtraction {
                                        row: row_index,
                                        message: error.to_string(),
                                    })?;
                                (instruction_input[offset], residual[offset]) =
                                    SpartanOuterUniskipRow::from_spartan_outer(&projected.outer)
                                        .split();
                                owner.push(
                                    row_index,
                                    explicit_rows,
                                    projected.instruction,
                                    projected.ram_access,
                                    projected.register_write,
                                    bytecode_scratch,
                                )?;
                            }
                            let mut padding_start = parts.physical;
                            if parts.regular_padding != 0 {
                                let regular = padding.regular.ok_or_else(|| {
                                    MetalError::InvalidInstructionReadRafGrouped(
                                        "regular Stage-1 padding template is missing".to_owned(),
                                    )
                                })?;
                                fill_stage1_outer_padding(
                                    instruction_input,
                                    residual,
                                    padding_start,
                                    parts.regular_padding,
                                    &regular,
                                );
                                owner.fill_padding(&regular, parts.regular_padding)?;
                                padding_start += parts.regular_padding;
                            }
                            if parts.terminal_padding != 0 {
                                let terminal = padding.terminal.ok_or_else(|| {
                                    MetalError::InvalidInstructionReadRafGrouped(
                                        "terminal Stage-1 padding template is missing".to_owned(),
                                    )
                                })?;
                                fill_stage1_outer_padding(
                                    instruction_input,
                                    residual,
                                    padding_start,
                                    parts.terminal_padding,
                                    &terminal,
                                );
                                owner.fill_padding(&terminal, parts.terminal_padding)?;
                            }
                            owner.finish(bytecode_scratch)
                        };
                    #[cfg(feature = "parallel")]
                    instruction_input
                        .par_chunks_mut(INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS)
                        .zip(residual.par_chunks_mut(INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS))
                        .zip(owner_chunks.par_iter_mut())
                        .enumerate()
                        .try_for_each_init(
                            BytecodeAddressStage1TopologyScratch::new,
                            |scratch, (chunk, ((instruction_input, residual), owner))| {
                                fill_chunk(chunk, instruction_input, residual, owner, scratch)
                            },
                        )?;
                    #[cfg(not(feature = "parallel"))]
                    {
                        let mut scratch = BytecodeAddressStage1TopologyScratch::new();
                        for (chunk, ((instruction_input, residual), owner)) in instruction_input
                            .chunks_mut(INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS)
                            .zip(residual.chunks_mut(INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS))
                            .zip(owner_chunks.iter_mut())
                            .enumerate()
                        {
                            fill_chunk(chunk, instruction_input, residual, owner, &mut scratch)?;
                        }
                    }
                    Ok(())
                },
            )
        })
        .map_err(MetalSpartanDenseRowsError::Metal)?
        .with_explicit_rows(explicit_rows)
        .map_err(MetalSpartanDenseRowsError::Metal)?;
    let owner = source.seal().map_err(MetalSpartanDenseRowsError::Metal)?;
    let source_storage_ids = [
        outer_rows.instruction_input_allocation_identity(),
        outer_rows.allocation_identity(),
    ];
    let source_storage_bytes = [
        instruction_input_row_bytes(cycles).map_err(MetalSpartanDenseRowsError::Metal)?,
        spartan_outer_uniskip_residual_row_bytes(cycles)
            .map_err(MetalSpartanDenseRowsError::Metal)?,
    ];
    let bytecode_topology = bytecode_topology
        .map(|topology| topology.seal(&owner))
        .transpose()
        .map_err(MetalSpartanDenseRowsError::Metal)?;
    let ram_access = ram_access
        .map(RamAccessCollectionStorage::seal)
        .transpose()
        .map_err(|error| MetalSpartanDenseRowsError::Kernel(error.into_kernel_error()))?;
    let registers_val = prepare_registers_val
        .then(|| {
            context.prepare_registers_val_instruction_source_request(
                cycles,
                explicit_rows,
                source_storage_ids[0],
                source_storage_bytes[0],
                source_storage_ids[1],
                source_storage_bytes[1],
                owner.receipt(),
            )
        })
        .transpose()
        .map_err(MetalSpartanDenseRowsError::Metal)?;
    record_bytecode_stage1_topology_span(&owner, bytecode_topology.as_ref(), explicit_rows);
    let prepared = InstructionReadRafStage1Ready {
        owner,
        bytecode_topology,
        registers_val,
        ram_access,
    };
    let _ = span.record(
        "compact_rows_storage_id",
        outer_rows.instruction_input_allocation_identity(),
    );
    let _ = span.record("residual_rows_storage_id", outer_rows.allocation_identity());
    Ok((outer_rows, prepared))
}

pub(crate) fn prepare_metal_spartan_outer_shift_witness_rows(
    context: &SolinasMetal,
    witness: &dyn JoltWitnessPlane<AkitaField>,
    cycles: usize,
) -> Result<(SpartanOuterUniskipRows, SpartanShiftResidentRows), MetalSpartanDenseRowsError> {
    context
        .validate_spartan_outer_uniskip_shift_rows_capacity(cycles)
        .map_err(MetalSpartanDenseRowsError::Metal)?;
    let rows = RowsStore::resolve(witness, cycles).map_err(MetalSpartanDenseRowsError::Kernel)?;
    let access = rows.access();
    let explicit_rows = rows.explicit_rows();
    let source_kind = rows.production_source_kind();
    let host_repack_rows = rows.host_repack_rows();
    let span = tracing::info_span!(
        "MetalInstructionInput::compact_rows_prepare",
        source_kind,
        witness_row_extractions = cycles,
        residual_rows_written = cycles,
        compact_rows_written = cycles,
        compact_row_bytes = 48,
        residual_row_bytes = 112,
        compact_allocations = 1,
        residual_allocations = 1,
        full_row_allocations = 0,
        full_domain_copy_bytes = 0,
        full_domain_copy_dispatches = 0,
        host_repack_rows,
        compact_rows_storage_id = tracing::field::Empty,
        residual_rows_storage_id = tracing::field::Empty,
        resident_rows = cycles,
        explicit_rows,
    );
    let _entered = span.enter();
    let (outer_rows, shift_rows) = context
        .prepare_spartan_outer_uniskip_rows_with_shift_fill(
            cycles,
            |instruction_input, residual, unexpanded_pc, pc, flags| {
                #[cfg(feature = "parallel")]
                {
                    instruction_input
                        .par_chunks_mut(SPARTAN_SHIFT_FLAG_ROWS_PER_WORD)
                        .zip(residual.par_chunks_mut(SPARTAN_SHIFT_FLAG_ROWS_PER_WORD))
                        .zip(unexpanded_pc.par_chunks_mut(SPARTAN_SHIFT_FLAG_ROWS_PER_WORD))
                        .zip(pc.par_chunks_mut(SPARTAN_SHIFT_FLAG_ROWS_PER_WORD))
                        .zip(flags.par_iter_mut())
                        .enumerate()
                        .try_for_each(
                            |(
                                word_index,
                                ((((instruction_input, residual), unexpanded_pc), pc), flags),
                            )|
                             -> Result<(), MetalError> {
                                let mut packed_flags = SpartanShiftFlagWord::default();
                                for offset in 0..instruction_input.len() {
                                    let row_index =
                                        word_index * SPARTAN_SHIFT_FLAG_ROWS_PER_WORD + offset;
                                    let row = access.row(row_index).map_err(|error| {
                                        MetalError::SpartanOuterRowExtraction {
                                            row: row_index,
                                            message: error.to_string(),
                                        }
                                    })?;
                                    (instruction_input[offset], residual[offset]) =
                                        SpartanOuterUniskipRow::from_spartan_outer(&row).split();
                                    write_metal_spartan_shift_row(
                                        &row,
                                        offset,
                                        &mut unexpanded_pc[offset],
                                        &mut pc[offset],
                                        &mut packed_flags,
                                    );
                                }
                                *flags = packed_flags;
                                Ok(())
                            },
                        )?;
                }
                #[cfg(not(feature = "parallel"))]
                {
                    flags.fill(SpartanShiftFlagWord::default());
                    for row_index in 0..cycles {
                        let row = access.row(row_index).map_err(|error| {
                            MetalError::SpartanOuterRowExtraction {
                                row: row_index,
                                message: error.to_string(),
                            }
                        })?;
                        (instruction_input[row_index], residual[row_index]) =
                            SpartanOuterUniskipRow::from_spartan_outer(&row).split();
                        write_metal_spartan_shift_row(
                            &row,
                            row_index % SPARTAN_SHIFT_FLAG_ROWS_PER_WORD,
                            &mut unexpanded_pc[row_index],
                            &mut pc[row_index],
                            &mut flags[row_index / SPARTAN_SHIFT_FLAG_ROWS_PER_WORD],
                        );
                    }
                }
                Ok(())
            },
        )
        .map_err(MetalSpartanDenseRowsError::Metal)?;
    let prepared = (
        outer_rows
            .with_explicit_rows(explicit_rows)
            .map_err(MetalSpartanDenseRowsError::Metal)?,
        shift_rows,
    );
    let _ = span.record(
        "compact_rows_storage_id",
        prepared.0.instruction_input_allocation_identity(),
    );
    let _ = span.record("residual_rows_storage_id", prepared.0.allocation_identity());
    Ok(prepared)
}

pub(crate) fn prepare_metal_spartan_outer_shift_stage1_owner_witness_rows(
    context: &SolinasMetal,
    witness: &dyn JoltWitnessPlane<AkitaField>,
    cycles: usize,
    prepare_bytecode_carrier: bool,
    prepare_registers_val: bool,
    prepare_ram_access: bool,
) -> Result<ShiftStage1OwnerPreparedRows, MetalSpartanDenseRowsError> {
    context
        .validate_spartan_outer_uniskip_shift_rows_capacity(cycles)
        .map_err(MetalSpartanDenseRowsError::Metal)?;
    let owned = witness
        .owned_rows()
        .filter(|rows| cycles <= rows.cycles())
        .ok_or(MetalSpartanDenseRowsError::Kernel(
            KernelError::InvariantViolation {
                reason: "InstructionReadRAF Stage-1 ownership requires a random-access witness",
            },
        ))?;
    let explicit_rows = owned.physical_rows().min(cycles);
    let access = owned.view();
    let padding = Stage1PaddingRows::new(&access, explicit_rows, cycles)
        .map_err(MetalSpartanDenseRowsError::Metal)?;
    let span = stage1_owner_rows_span(
        cycles,
        explicit_rows,
        padding.source_window_count(explicit_rows),
    );
    let _entered = span.enter();
    let topology_span = bytecode_stage1_topology_span(prepare_bytecode_carrier, explicit_rows);
    let _topology_entered = topology_span.enter();
    let mut source = context
        .prepare_instruction_read_raf_stage1_storage(cycles)
        .map_err(MetalSpartanDenseRowsError::Metal)?;
    let mut bytecode_topology = prepare_bytecode_carrier
        .then(|| context.prepare_bytecode_address_stage1_topology_storage(cycles, explicit_rows))
        .transpose()
        .map_err(MetalSpartanDenseRowsError::Metal)?;
    let mut ram_access = prepare_ram_access
        .then(|| RamAccessCollectionStorage::new(cycles, INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS))
        .transpose()
        .map_err(|error| MetalSpartanDenseRowsError::Kernel(error.into_kernel_error()))?;
    let (outer_rows, shift_rows) = context
        .prepare_spartan_outer_uniskip_rows_with_shift_fill(
            cycles,
            |instruction_input, residual, unexpanded_pc, pc, flags| {
                with_stage1_owner_chunks(
                    &mut source,
                    bytecode_topology.as_mut(),
                    ram_access.as_mut(),
                    |owner_chunks| {
                        let fill_chunk =
                            |chunk: usize,
                             instruction_input: &mut [InstructionInputRow],
                             residual: &mut [SpartanOuterUniskipResidualRow],
                             unexpanded_pc: &mut [u64],
                             pc: &mut [u64],
                             flags: &mut [SpartanShiftFlagWord],
                             owner: &mut Stage1OwnerChunkWriters<'_, '_, '_, '_>,
                             bytecode_scratch: &mut BytecodeAddressStage1TopologyScratch|
                             -> Result<(), MetalError> {
                                if instruction_input.len() != owner.len()
                                    || residual.len() != owner.len()
                                    || unexpanded_pc.len() != owner.len()
                                    || pc.len() != owner.len()
                                    || flags.len() != owner.len() / SPARTAN_SHIFT_FLAG_ROWS_PER_WORD
                                {
                                    return Err(MetalError::InvalidInstructionReadRafGrouped(
                                        "Stage-1 owner/Shift chunks disagree on row count"
                                            .to_owned(),
                                    ));
                                }
                                flags.fill(SpartanShiftFlagWord::default());
                                let chunk_start = chunk * INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS;
                                let parts = stage1_chunk_parts(
                                    chunk_start,
                                    owner.len(),
                                    explicit_rows,
                                    cycles,
                                );
                                for offset in 0..parts.physical {
                                    let row_index = chunk_start + offset;
                                    let projected: Stage1ProjectionRow =
                                        access.window(row_index).map_err(|error| {
                                            MetalError::SpartanOuterRowExtraction {
                                                row: row_index,
                                                message: error.to_string(),
                                            }
                                        })?;
                                    (instruction_input[offset], residual[offset]) =
                                        SpartanOuterUniskipRow::from_spartan_outer(
                                            &projected.outer,
                                        )
                                        .split();
                                    write_metal_spartan_shift_row(
                                        &projected.outer,
                                        offset % SPARTAN_SHIFT_FLAG_ROWS_PER_WORD,
                                        &mut unexpanded_pc[offset],
                                        &mut pc[offset],
                                        &mut flags[offset / SPARTAN_SHIFT_FLAG_ROWS_PER_WORD],
                                    );
                                    owner.push(
                                        row_index,
                                        explicit_rows,
                                        projected.instruction,
                                        projected.ram_access,
                                        projected.register_write,
                                        bytecode_scratch,
                                    )?;
                                }
                                let mut padding_start = parts.physical;
                                if parts.regular_padding != 0 {
                                    let regular = padding.regular.ok_or_else(|| {
                                        MetalError::InvalidInstructionReadRafGrouped(
                                            "regular Stage-1 padding template is missing"
                                                .to_owned(),
                                        )
                                    })?;
                                    fill_stage1_outer_padding(
                                        instruction_input,
                                        residual,
                                        padding_start,
                                        parts.regular_padding,
                                        &regular,
                                    );
                                    fill_stage1_shift_padding(
                                        unexpanded_pc,
                                        pc,
                                        flags,
                                        padding_start,
                                        parts.regular_padding,
                                        &regular,
                                    );
                                    owner.fill_padding(&regular, parts.regular_padding)?;
                                    padding_start += parts.regular_padding;
                                }
                                if parts.terminal_padding != 0 {
                                    let terminal = padding.terminal.ok_or_else(|| {
                                        MetalError::InvalidInstructionReadRafGrouped(
                                            "terminal Stage-1 padding template is missing"
                                                .to_owned(),
                                        )
                                    })?;
                                    fill_stage1_outer_padding(
                                        instruction_input,
                                        residual,
                                        padding_start,
                                        parts.terminal_padding,
                                        &terminal,
                                    );
                                    fill_stage1_shift_padding(
                                        unexpanded_pc,
                                        pc,
                                        flags,
                                        padding_start,
                                        parts.terminal_padding,
                                        &terminal,
                                    );
                                    owner.fill_padding(&terminal, parts.terminal_padding)?;
                                }
                                owner.finish(bytecode_scratch)
                            };
                        let flags_per_chunk = INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS
                            / SPARTAN_SHIFT_FLAG_ROWS_PER_WORD;
                        #[cfg(feature = "parallel")]
                        instruction_input
                            .par_chunks_mut(INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS)
                            .zip(residual.par_chunks_mut(INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS))
                            .zip(
                                unexpanded_pc
                                    .par_chunks_mut(INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS),
                            )
                            .zip(pc.par_chunks_mut(INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS))
                            .zip(flags.par_chunks_mut(flags_per_chunk))
                            .zip(owner_chunks.par_iter_mut())
                            .enumerate()
                            .try_for_each_init(
                                BytecodeAddressStage1TopologyScratch::new,
                                |bytecode_scratch,
                                 (
                                    chunk,
                                    (
                                        (
                                            (((instruction_input, residual), unexpanded_pc), pc),
                                            flags,
                                        ),
                                        owner,
                                    ),
                                )| {
                                    fill_chunk(
                                        chunk,
                                        instruction_input,
                                        residual,
                                        unexpanded_pc,
                                        pc,
                                        flags,
                                        owner,
                                        bytecode_scratch,
                                    )
                                },
                            )?;
                        #[cfg(not(feature = "parallel"))]
                        {
                            let mut bytecode_scratch = BytecodeAddressStage1TopologyScratch::new();
                            for (
                                chunk,
                                (
                                    ((((instruction_input, residual), unexpanded_pc), pc), flags),
                                    owner,
                                ),
                            ) in instruction_input
                                .chunks_mut(INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS)
                                .zip(residual.chunks_mut(INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS))
                                .zip(
                                    unexpanded_pc
                                        .chunks_mut(INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS),
                                )
                                .zip(pc.chunks_mut(INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS))
                                .zip(flags.chunks_mut(flags_per_chunk))
                                .zip(owner_chunks.iter_mut())
                                .enumerate()
                            {
                                fill_chunk(
                                    chunk,
                                    instruction_input,
                                    residual,
                                    unexpanded_pc,
                                    pc,
                                    flags,
                                    owner,
                                    &mut bytecode_scratch,
                                )?;
                            }
                        }
                        Ok(())
                    },
                )
            },
        )
        .map_err(MetalSpartanDenseRowsError::Metal)?;
    let outer_rows = outer_rows
        .with_explicit_rows(explicit_rows)
        .map_err(MetalSpartanDenseRowsError::Metal)?;
    let owner = source.seal().map_err(MetalSpartanDenseRowsError::Metal)?;
    let source_storage_ids = [
        outer_rows.instruction_input_allocation_identity(),
        outer_rows.allocation_identity(),
    ];
    let source_storage_bytes = [
        instruction_input_row_bytes(cycles).map_err(MetalSpartanDenseRowsError::Metal)?,
        spartan_outer_uniskip_residual_row_bytes(cycles)
            .map_err(MetalSpartanDenseRowsError::Metal)?,
    ];
    let bytecode_topology = bytecode_topology
        .map(|topology| topology.seal(&owner))
        .transpose()
        .map_err(MetalSpartanDenseRowsError::Metal)?;
    let ram_access = ram_access
        .map(RamAccessCollectionStorage::seal)
        .transpose()
        .map_err(|error| MetalSpartanDenseRowsError::Kernel(error.into_kernel_error()))?;
    let registers_val = prepare_registers_val
        .then(|| {
            context.prepare_registers_val_instruction_source_request(
                cycles,
                explicit_rows,
                source_storage_ids[0],
                source_storage_bytes[0],
                source_storage_ids[1],
                source_storage_bytes[1],
                owner.receipt(),
            )
        })
        .transpose()
        .map_err(MetalSpartanDenseRowsError::Metal)?;
    record_bytecode_stage1_topology_span(&owner, bytecode_topology.as_ref(), explicit_rows);
    let prepared = InstructionReadRafStage1Ready {
        owner,
        bytecode_topology,
        registers_val,
        ram_access,
    };
    let _ = span.record(
        "compact_rows_storage_id",
        outer_rows.instruction_input_allocation_identity(),
    );
    let _ = span.record("residual_rows_storage_id", outer_rows.allocation_identity());
    Ok((outer_rows, shift_rows, prepared))
}

fn write_metal_spartan_shift_row(
    row: &SpartanOuterRow,
    bit: usize,
    unexpanded_pc: &mut u64,
    pc: &mut u64,
    flags: &mut SpartanShiftFlagWord,
) {
    *unexpanded_pc = row.unexpanded_pc.0;
    *pc = row.pc.0;
    let mask = 1u32 << bit;
    flags.is_virtual |= u32::from(row.virtual_instruction.0) * mask;
    flags.is_first_in_sequence |= u32::from(row.is_first_in_sequence.0) * mask;
    flags.is_noop |= u32::from(row.is_noop.0) * mask;
}

pub(crate) fn prepare_metal_instruction_input_witness_rows(
    context: &SolinasMetal,
    witness: &dyn JoltWitnessPlane<AkitaField>,
    cycles: usize,
) -> Result<InstructionInputRows, KernelError<AkitaField>> {
    let rows = RowsStore::resolve(witness, cycles)?;
    let access = rows.access();
    let explicit_rows = rows.explicit_rows();
    let source_kind = rows.production_source_kind();
    let host_repack_rows = rows.host_repack_rows();
    let span = tracing::info_span!(
        "MetalInstructionInput::compact_rows_prepare",
        source_kind,
        witness_row_extractions = cycles,
        residual_rows_written = 0,
        compact_rows_written = cycles,
        compact_row_bytes = 48,
        residual_row_bytes = 0,
        compact_allocations = 1,
        residual_allocations = 0,
        full_row_allocations = 0,
        full_domain_copy_bytes = 0,
        full_domain_copy_dispatches = 0,
        host_repack_rows,
        compact_rows_storage_id = tracing::field::Empty,
        residual_rows_storage_id = 0,
        resident_rows = cycles,
        explicit_rows,
    );
    let _entered = span.enter();
    let prepared = context
        .prepare_instruction_input_rows_with_fill(cycles, |destination| {
            #[cfg(feature = "parallel")]
            {
                destination.par_iter_mut().enumerate().try_for_each(
                    |(row_index, destination)| -> Result<(), MetalError> {
                        let row = access.row(row_index).map_err(|error| {
                            MetalError::SpartanOuterRowExtraction {
                                row: row_index,
                                message: error.to_string(),
                            }
                        })?;
                        *destination = InstructionInputRow::from_spartan_outer(&row);
                        Ok(())
                    },
                )?;
            }
            #[cfg(not(feature = "parallel"))]
            {
                for (row_index, destination) in destination.iter_mut().enumerate() {
                    let row = access.row(row_index).map_err(|error| {
                        MetalError::SpartanOuterRowExtraction {
                            row: row_index,
                            message: error.to_string(),
                        }
                    })?;
                    *destination = InstructionInputRow::from_spartan_outer(&row);
                }
            }
            Ok(())
        })
        .map_err(metal_outer_error)?;
    let _ = span.record("compact_rows_storage_id", prepared.allocation_identity());
    Ok(prepared)
}

fn prepare_metal_spartan_outer_rows(
    context: &SolinasMetal,
    rows: &RowsStore,
    cycles: usize,
) -> Result<SpartanOuterUniskipRows, KernelError<AkitaField>> {
    let access = rows.access();
    let explicit_rows = rows.explicit_rows();
    let source_kind = rows.production_source_kind();
    let host_repack_rows = rows.host_repack_rows();
    let span = tracing::info_span!(
        "MetalInstructionInput::compact_rows_prepare",
        source_kind,
        witness_row_extractions = cycles,
        residual_rows_written = cycles,
        compact_rows_written = cycles,
        compact_row_bytes = 48,
        residual_row_bytes = 112,
        compact_allocations = 1,
        residual_allocations = 1,
        full_row_allocations = 0,
        full_domain_copy_bytes = 0,
        full_domain_copy_dispatches = 0,
        host_repack_rows,
        compact_rows_storage_id = tracing::field::Empty,
        residual_rows_storage_id = tracing::field::Empty,
        resident_rows = cycles,
        explicit_rows,
    );
    let _entered = span.enter();
    let prepared = context
        .prepare_spartan_outer_uniskip_rows_with_fill(cycles, |instruction_input, residual| {
            #[cfg(feature = "parallel")]
            {
                instruction_input
                    .par_iter_mut()
                    .zip(residual.par_iter_mut())
                    .enumerate()
                    .try_for_each(
                        |(row_index, (instruction_input, residual))| -> Result<(), MetalError> {
                            let row = access.row(row_index).map_err(|error| {
                                MetalError::SpartanOuterRowExtraction {
                                    row: row_index,
                                    message: error.to_string(),
                                }
                            })?;
                            (*instruction_input, *residual) =
                                SpartanOuterUniskipRow::from_spartan_outer(&row).split();
                            Ok(())
                        },
                    )?;
            }
            #[cfg(not(feature = "parallel"))]
            {
                for (row_index, (instruction_input, residual)) in
                    instruction_input.iter_mut().zip(residual).enumerate()
                {
                    let row = access.row(row_index).map_err(|error| {
                        MetalError::SpartanOuterRowExtraction {
                            row: row_index,
                            message: error.to_string(),
                        }
                    })?;
                    (*instruction_input, *residual) =
                        SpartanOuterUniskipRow::from_spartan_outer(&row).split();
                }
            }
            Ok(())
        })
        .map_err(metal_outer_error)?
        .with_explicit_rows(explicit_rows)
        .map_err(metal_outer_error)?;
    let _ = span.record(
        "compact_rows_storage_id",
        prepared.instruction_input_allocation_identity(),
    );
    let _ = span.record("residual_rows_storage_id", prepared.allocation_identity());
    Ok(prepared)
}

fn metal_outer_error(error: MetalError) -> KernelError<AkitaField> {
    SumcheckError::ComputeBackend {
        backend: "metal",
        message: error.to_string(),
    }
    .into()
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_poly::EqPolynomial;
    use jolt_witness::testing::with_sample_backend;
    use jolt_witness::witnesses::SpartanOuterRow;
    use jolt_witness::BundleSource;

    use super::*;

    #[test]
    fn instruction_source_lookup_and_increment_are_reconstructible() {
        use jolt_lookup_tables::interleave_bits;

        with_sample_backend(|backend| {
            let rows: Vec<Stage1ProjectionRow> = backend.bundles().unwrap();
            let independently_extracted: Vec<Stage1InstructionFacts> = backend.bundles().unwrap();
            for (cycle, (row, expected)) in
                rows.into_iter().zip(independently_extracted).enumerate()
            {
                assert_eq!(row.instruction.lookup_index.0, expected.lookup_index.0);
                assert_eq!(row.instruction.table_index.0, expected.table_index.0);
                assert_eq!(row.instruction.raf_flag.0, expected.raf_flag.0);
                assert_eq!(row.instruction.mapped_pc.0, expected.mapped_pc.0);
                assert_eq!(
                    row.instruction.remapped_ram_address.0,
                    expected.remapped_ram_address.0
                );
                assert_eq!(row.instruction.fused_inc.0, expected.fused_inc.0);
                let right = row.outer.right_lookup_operand.0;
                let reconstructed_lookup = if row.instruction.raf_flag.0 {
                    right
                } else {
                    assert_eq!(right >> 64, 0, "cycle {cycle}");
                    interleave_bits(row.outer.left_lookup_operand.0, right as u64)
                };
                assert_eq!(
                    reconstructed_lookup, row.instruction.lookup_index.0,
                    "lookup index at cycle {cycle}"
                );

                let reconstructed_increment = if row.outer.store.0 {
                    assert!(row.register_write.is_none(), "store cycle {cycle}");
                    i128::from(row.outer.ram_write_value.0) - i128::from(row.outer.ram_read_value.0)
                } else {
                    row.register_write
                        .map_or(0, |(_, pre, post)| i128::from(post) - i128::from(pre))
                };
                assert_eq!(
                    reconstructed_increment, row.instruction.fused_inc.0,
                    "fused increment at cycle {cycle}"
                );
            }
        });
    }

    #[test]
    fn metal_co_produced_projection_matches_independent_producers() {
        use jolt_claims::protocols::jolt::JoltOneHotConfig;
        use jolt_program::execution::{JoltProgram, OwnedTrace, TraceOutput, TraceRow};
        use jolt_program::preprocess::{
            BytecodePreprocessing, JoltProgramPreprocessing, RAMPreprocessing,
        };
        use jolt_riscv::{
            JoltInstructionKind, JoltInstructionRow, NormalizedOperands, RV64IMAC_JOLT,
        };
        use jolt_witness::{
            JoltVmWitnessConfig, JoltVmWitnessInputs, JoltWitnessPlane, TraceBackend,
        };

        use crate::metal::solinas::spartan_shift::{
            SpartanShiftGeometry, SpartanShiftKernelConfig,
        };
        use crate::metal::solinas::SolinasMetal;
        use crate::optimized::spartan_shift::prepare_metal_spartan_shift_witness_rows;

        fn instruction(
            address: usize,
            virtual_sequence_remaining: Option<u16>,
            first: bool,
        ) -> JoltInstructionRow {
            JoltInstructionRow {
                instruction_kind: JoltInstructionKind::ADDI,
                address,
                operands: NormalizedOperands {
                    rd: Some(1),
                    rs1: Some(2),
                    rs2: None,
                    imm: 3,
                },
                virtual_sequence_remaining,
                is_first_in_sequence: first,
                is_compressed: false,
            }
        }

        let log_t = 4usize;
        let cycles = 1usize << log_t;
        let plain_a = instruction(0x8000_0000, None, false);
        let virtual_first = instruction(0x8000_0004, Some(1), true);
        let virtual_last = instruction(0x8000_0004, Some(0), false);
        let plain_b = instruction(0x8000_0008, None, false);
        let noop = JoltInstructionRow {
            instruction_kind: JoltInstructionKind::NoOp,
            ..plain_a
        };
        let bytecode = vec![plain_a, virtual_first, virtual_last, plain_b];
        let rows: Vec<TraceRow> = [plain_a, virtual_first, virtual_last, noop, plain_b, plain_a]
            .into_iter()
            .map(|instruction| TraceRow {
                instruction,
                ..TraceRow::default()
            })
            .collect();
        let preprocessing = JoltProgramPreprocessing {
            bytecode: BytecodePreprocessing::preprocess(
                bytecode,
                plain_a.address as u64,
                RV64IMAC_JOLT,
            )
            .unwrap(),
            ram: RAMPreprocessing::default(),
            memory_layout: Default::default(),
            max_padded_trace_length: cycles,
        };
        let program = JoltProgram::default();
        let config = JoltVmWitnessConfig::new(
            log_t,
            64,
            JoltOneHotConfig {
                log_k_chunk: 4,
                lookups_ra_virtual_log_k_chunk: 16,
            },
        );
        let inputs = JoltVmWitnessInputs::new(
            &program,
            &preprocessing,
            TraceOutput::new(OwnedTrace::new(rows), Default::default(), None, None),
        );
        let backend = TraceBackend::new(config, inputs);
        let witness = &backend as &dyn JoltWitnessPlane<jolt_field::AkitaField>;
        let projected: Vec<SpartanOuterRow> = backend.bundles().unwrap();
        assert!(projected.iter().any(|row| row.virtual_instruction.0));
        assert!(projected.iter().any(|row| row.is_first_in_sequence.0));
        assert!(projected.iter().any(|row| row.is_noop.0));

        let context = SolinasMetal::for_akita().unwrap();
        let independent_outer =
            prepare_metal_spartan_outer_witness_rows(&context, witness, cycles).unwrap();
        let independent_shift =
            prepare_metal_spartan_shift_witness_rows(&context, witness, cycles).unwrap();
        let (combined_outer, combined_shift) =
            prepare_metal_spartan_outer_shift_witness_rows(&context, witness, cycles).unwrap();

        let e_out = EqPolynomial::<jolt_field::AkitaField>::evals(
            &[
                jolt_field::AkitaField::from_u64(3),
                jolt_field::AkitaField::from_u64(5),
            ],
            None,
        );
        let e_in = EqPolynomial::<jolt_field::AkitaField>::evals(
            &[
                jolt_field::AkitaField::from_u64(7),
                jolt_field::AkitaField::from_u64(11),
                jolt_field::AkitaField::from_u64(13),
            ],
            None,
        );
        let outer_config = SpartanOuterUniskipConfig {
            threads_per_threadgroup: Some(32),
        };
        let independent_outer_invocation = context
            .prepare_spartan_outer_uniskip_with_rows(
                &independent_outer,
                &e_in,
                &e_out,
                outer_config,
            )
            .unwrap();
        independent_outer_invocation.execute().unwrap();
        let independent_outer_output = independent_outer_invocation.read_output().unwrap();
        let combined_outer_invocation = context
            .prepare_spartan_outer_uniskip_with_rows(&combined_outer, &e_in, &e_out, outer_config)
            .unwrap();
        combined_outer_invocation.execute().unwrap();
        assert_eq!(
            combined_outer_invocation.read_output().unwrap(),
            independent_outer_output
        );

        let geometry = SpartanShiftGeometry::new(cycles).unwrap();
        let point = |seed: u64| {
            (0..log_t)
                .scan(seed, |state, _| {
                    *state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    Some(jolt_field::AkitaField::from_u64(*state | 1))
                })
                .collect::<Vec<_>>()
        };
        let r_outer = point(0xA11C_E001);
        let r_product = point(0xB22D_F002);
        let gamma = jolt_field::AkitaField::from_u64(0xC33E_1003);
        let shift_config = SpartanShiftKernelConfig {
            build_threads_per_threadgroup: 32,
            high_tile_elements: geometry.suffix_elements(),
            fold_threads_per_threadgroup: 32,
        };
        let independent_shift_output = context
            .prepare_spartan_shift_prefix(
                &independent_shift,
                &r_outer,
                &r_product,
                gamma,
                shift_config,
            )
            .unwrap()
            .execute()
            .unwrap();
        let combined_shift_output = context
            .prepare_spartan_shift_prefix(
                &combined_shift,
                &r_outer,
                &r_product,
                gamma,
                shift_config,
            )
            .unwrap()
            .execute()
            .unwrap();
        assert_eq!(combined_shift_output.q, independent_shift_output.q);
    }
}
