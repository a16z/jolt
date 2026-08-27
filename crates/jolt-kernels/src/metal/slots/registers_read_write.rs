//! Metal registers read/write-checking sparse cycle rounds.

use std::sync::Arc;

use jolt_claims::protocols::jolt::geometry::registers::rd_inc_read_write;
use jolt_field::{Fr, Ring};
use jolt_poly::{BindingOrder, GruenSplitEqPolynomial, Polynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage4::registers_read_write_checking::{
    RegistersReadWriteChecking, RegistersReadWriteOutputClaims,
};
use jolt_witness::JoltWitnessPlane;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::{num_threadgroups, Partials, RoundTable};
use crate::metal::buffers::{OwnedDeviceBuffer, PageAlignedVec};
use crate::metal::field::{fr_as_u32s, fr_to_u32_limbs};
use crate::metal::runtime::{ComputePass, DetachedPass, KernelId, MetalContext};
use crate::metal::{metal_gate, testing, MetalError};
use crate::mmap_vec::MmapVec;
#[cfg(feature = "parallel")]
use crate::optimized::registers_read_write::register_build_chunk_size;
use crate::optimized::registers_read_write::{
    BoundRegistersRwEntry, OptimizedRegistersReadWrite, ReadWriteKernel, RegistersRwCycleEntry,
    SharedRdIndices,
};
use crate::optimized::trace_record::{RegisterLanes, TraceRecord, NO_REGISTER};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

const KIND: &str = "registers_read_write";
const RA_TABLE_DEREF_LEN: usize = 256;

fn fused_rounds_enabled() -> bool {
    std::env::var("JOLT_REGRW_FUSED").is_ok_and(|value| matches!(value.trim(), "1" | "on" | "ON"))
}

fn gpu_prepare_enabled() -> bool {
    std::env::var("JOLT_REGRW_GPU_PREPARE")
        .map_or(true, |value| !matches!(value.trim(), "0" | "off" | "OFF"))
        && std::env::var_os("JOLT_REGISTERS_PREPARE_SERIAL").is_none()
}

/// Bind outputs as kernel-zeroed mmap buffers instead of host-memset
/// `PageAlignedVec`s: the eager serial zero-fill of the multi-GiB early-round
/// CSRs was 2.84 s of st4 host time @2^27. `JOLT_REGRW_MMAP_BIND=0` restores
/// the eager allocation.
fn mmap_bind_buffers_enabled() -> bool {
    std::env::var("JOLT_REGRW_MMAP_BIND")
        .map_or(true, |value| !matches!(value.trim(), "0" | "off" | "OFF"))
}

/// Recycle bind output buffers round-over-round (ping-pong through the
/// retired input slab) instead of mapping a fresh region every round: the
/// lazy-zero mmap buffers pay first-touch fault cost inside the bind CBs
/// (the W11 31% clawback); a reused slab's pages are already resident.
/// Sound because every readable slot `[0, new_count)` is fully written by
/// the bind kernel (out_offsets are exact prefix sums of the merge counts).
/// `JOLT_REGRW_ARENA=0` restores per-round allocation.
fn arena_reuse_enabled() -> bool {
    std::env::var("JOLT_REGRW_ARENA")
        .map_or(true, |value| !matches!(value.trim(), "0" | "off" | "OFF"))
}

/// Launch each device cycle round as a detached fused bind+message command
/// buffer through [`ProveRounds::begin_round`], so the batch engine overlaps
/// the CB with RamValCheck's synchronous CPU rounds instead of serializing
/// them behind blocking waits. `JOLT_REGRW_OVERLAP=0` restores the
/// synchronous schedule.
fn overlap_rounds_enabled() -> bool {
    std::env::var("JOLT_REGRW_OVERLAP")
        .map_or(true, |value| !matches!(value.trim(), "0" | "off" | "OFF"))
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct RawRwEntryIdx {
    val: Fr,
    prev_val: u64,
    next_val: u64,
    ra: u16,
    wa: u16,
    col: u8,
    pad: [u8; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct RawRwEntryF {
    val: Fr,
    ra: Fr,
    wa: Fr,
    prev_val: u64,
    next_val: u64,
    col: u8,
    pad: [u8; 7],
}

const _: () = {
    assert!(std::mem::size_of::<RawRwEntryIdx>() == 56);
    assert!(std::mem::size_of::<RawRwEntryF>() == 120);
};

struct MetalRegisterTables {
    entries: Vec<RawRwEntryIdx>,
    offsets: Vec<u32>,
    inc: Vec<Fr>,
    rs1_indices: Vec<Option<u8>>,
    rs2_indices: Vec<Option<u8>>,
    rd_indices: Vec<Option<u8>>,
}

struct MetalRegisterMetadata {
    offsets: Vec<u32>,
    inc: Vec<Fr>,
    rs1_indices: Vec<Option<u8>>,
    rs2_indices: Vec<Option<u8>>,
    rd_indices: Vec<Option<u8>>,
}

#[derive(Clone, Copy)]
struct RegisterBuildInputs<'a> {
    rs1_value: &'a [u64],
    rs2_value: &'a [u64],
    rd_pre_value: &'a [u64],
    rd_post_value: &'a [u64],
    rs1_index: &'a [u8],
    rs2_index: &'a [u8],
    rd_index: &'a [u8],
}

impl<'a> From<&'a RegisterLanes> for RegisterBuildInputs<'a> {
    fn from(registers: &'a RegisterLanes) -> Self {
        Self {
            rs1_value: registers.rs1_value.as_slice(),
            rs2_value: registers.rs2_value.as_slice(),
            rd_pre_value: registers.rd_pre_value.as_slice(),
            rd_post_value: registers.rd_post_value.as_slice(),
            rs1_index: registers.rs1_index.as_slice(),
            rs2_index: registers.rs2_index.as_slice(),
            rd_index: registers.rd_index.as_slice(),
        }
    }
}

impl RegisterBuildInputs<'_> {
    fn len(self) -> usize {
        self.rd_index.len()
    }
}

#[inline]
fn raw_cycle_entries(registers: RegisterBuildInputs<'_>, t: usize) -> ([RawRwEntryIdx; 3], usize) {
    let mut row = [RawRwEntryIdx::default(); 3];
    let mut len = 0;
    let rs1 = registers.rs1_index[t];
    let rs2 = registers.rs2_index[t];
    let rd = registers.rd_index[t];
    if rs1 != NO_REGISTER {
        row[len] = RawRwEntryIdx {
            val: Fr::from_u64(registers.rs1_value[t]),
            prev_val: registers.rs1_value[t],
            next_val: registers.rs1_value[t],
            ra: 1,
            wa: 0,
            col: rs1,
            pad: [0; 3],
        };
        len += 1;
    }
    if rs2 != NO_REGISTER {
        if let Some(entry) = row[..len].iter_mut().find(|entry| entry.col == rs2) {
            entry.ra = 3;
        } else {
            row[len] = RawRwEntryIdx {
                val: Fr::from_u64(registers.rs2_value[t]),
                prev_val: registers.rs2_value[t],
                next_val: registers.rs2_value[t],
                ra: 2,
                wa: 0,
                col: rs2,
                pad: [0; 3],
            };
            len += 1;
        }
    }
    if rd != NO_REGISTER {
        if let Some(entry) = row[..len].iter_mut().find(|entry| entry.col == rd) {
            entry.wa = 1;
            entry.next_val = registers.rd_post_value[t];
        } else {
            row[len] = RawRwEntryIdx {
                val: Fr::from_u64(registers.rd_pre_value[t]),
                prev_val: registers.rd_pre_value[t],
                next_val: registers.rd_post_value[t],
                ra: 0,
                wa: 1,
                col: rd,
                pad: [0; 3],
            };
            len += 1;
        }
    }
    row[..len].sort_unstable_by_key(|entry| entry.col);
    (row, len)
}

#[inline]
fn raw_cycle_entry_count(registers: RegisterBuildInputs<'_>, t: usize) -> usize {
    let rs1 = registers.rs1_index[t];
    let rs2 = registers.rs2_index[t];
    let rd = registers.rd_index[t];
    usize::from(rs1 != NO_REGISTER)
        + usize::from(rs2 != NO_REGISTER && rs2 != rs1)
        + usize::from(rd != NO_REGISTER && rd != rs1 && rd != rs2)
}

fn build_metal_register_tables_serial(registers: RegisterBuildInputs<'_>) -> MetalRegisterTables {
    let cycles = registers.len();
    let mut tables = MetalRegisterTables {
        entries: Vec::with_capacity(cycles * 3),
        offsets: Vec::with_capacity(cycles + 1),
        inc: Vec::with_capacity(cycles),
        rs1_indices: Vec::with_capacity(cycles),
        rs2_indices: Vec::with_capacity(cycles),
        rd_indices: Vec::with_capacity(cycles),
    };
    tables.offsets.push(0);
    for t in 0..cycles {
        let (row, len) = raw_cycle_entries(registers, t);
        tables.entries.extend_from_slice(&row[..len]);
        tables.offsets.push(tables.entries.len() as u32);
        tables.inc.push(Fr::from_i128(
            i128::from(registers.rd_post_value[t]) - i128::from(registers.rd_pre_value[t]),
        ));
        let rs1 = registers.rs1_index[t];
        let rs2 = registers.rs2_index[t];
        let rd = registers.rd_index[t];
        tables.rs1_indices.push((rs1 != NO_REGISTER).then_some(rs1));
        tables.rs2_indices.push((rs2 != NO_REGISTER).then_some(rs2));
        tables.rd_indices.push((rd != NO_REGISTER).then_some(rd));
    }
    tables
}

#[cfg(any(test, not(feature = "parallel")))]
fn build_metal_register_metadata_serial(
    registers: RegisterBuildInputs<'_>,
) -> MetalRegisterMetadata {
    let cycles = registers.len();
    let mut metadata = MetalRegisterMetadata {
        offsets: Vec::with_capacity(cycles + 1),
        inc: Vec::with_capacity(cycles),
        rs1_indices: Vec::with_capacity(cycles),
        rs2_indices: Vec::with_capacity(cycles),
        rd_indices: Vec::with_capacity(cycles),
    };
    metadata.offsets.push(0);
    for t in 0..cycles {
        metadata
            .offsets
            .push(metadata.offsets[t] + raw_cycle_entry_count(registers, t) as u32);
        metadata.inc.push(Fr::from_i128(
            i128::from(registers.rd_post_value[t]) - i128::from(registers.rd_pre_value[t]),
        ));
        let rs1 = registers.rs1_index[t];
        let rs2 = registers.rs2_index[t];
        let rd = registers.rd_index[t];
        metadata
            .rs1_indices
            .push((rs1 != NO_REGISTER).then_some(rs1));
        metadata
            .rs2_indices
            .push((rs2 != NO_REGISTER).then_some(rs2));
        metadata.rd_indices.push((rd != NO_REGISTER).then_some(rd));
    }
    metadata
}

#[cfg(feature = "parallel")]
fn build_metal_register_metadata_parallel(
    registers: RegisterBuildInputs<'_>,
    chunk_size: usize,
) -> MetalRegisterMetadata {
    let cycles = registers.len();
    let mut offsets = vec![0u32; cycles + 1];
    offsets[1..]
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk, counts)| {
            let start = chunk * chunk_size;
            for (local_t, count) in counts.iter_mut().enumerate() {
                *count = raw_cycle_entry_count(registers, start + local_t) as u32;
            }
        });
    for t in 0..cycles {
        offsets[t + 1] += offsets[t];
    }

    let mut metadata = MetalRegisterMetadata {
        offsets,
        inc: Vec::with_capacity(cycles),
        rs1_indices: Vec::with_capacity(cycles),
        rs2_indices: Vec::with_capacity(cycles),
        rd_indices: Vec::with_capacity(cycles),
    };
    metadata
        .inc
        .spare_capacity_mut()
        .par_chunks_mut(chunk_size)
        .zip(
            metadata
                .rs1_indices
                .spare_capacity_mut()
                .par_chunks_mut(chunk_size),
        )
        .zip(
            metadata
                .rs2_indices
                .spare_capacity_mut()
                .par_chunks_mut(chunk_size),
        )
        .zip(
            metadata
                .rd_indices
                .spare_capacity_mut()
                .par_chunks_mut(chunk_size),
        )
        .enumerate()
        .for_each(|(chunk, (((inc, rs1_indices), rs2_indices), rd_indices))| {
            let start = chunk * chunk_size;
            for local_t in 0..inc.len() {
                let t = start + local_t;
                let _ = inc[local_t].write(Fr::from_i128(
                    i128::from(registers.rd_post_value[t]) - i128::from(registers.rd_pre_value[t]),
                ));
                let rs1 = registers.rs1_index[t];
                let rs2 = registers.rs2_index[t];
                let rd = registers.rd_index[t];
                let _ = rs1_indices[local_t].write((rs1 != NO_REGISTER).then_some(rs1));
                let _ = rs2_indices[local_t].write((rs2 != NO_REGISTER).then_some(rs2));
                let _ = rd_indices[local_t].write((rd != NO_REGISTER).then_some(rd));
            }
        });

    // SAFETY: each spare-capacity slot is in exactly one parallel chunk and
    // initialized once above.
    unsafe {
        metadata.inc.set_len(cycles);
        metadata.rs1_indices.set_len(cycles);
        metadata.rs2_indices.set_len(cycles);
        metadata.rd_indices.set_len(cycles);
    }
    metadata
}

fn build_metal_register_metadata(registers: RegisterBuildInputs<'_>) -> MetalRegisterMetadata {
    #[cfg(feature = "parallel")]
    {
        build_metal_register_metadata_parallel(
            registers,
            register_build_chunk_size(registers.len()),
        )
    }
    #[cfg(not(feature = "parallel"))]
    {
        build_metal_register_metadata_serial(registers)
    }
}

fn expand_lookup_table(table: &[Fr], challenge: Fr) -> Vec<Fr> {
    let mut next = Vec::with_capacity(table.len() * table.len());
    for odd in table {
        for even in table {
            next.push(*even + challenge * (*odd - *even));
        }
    }
    next
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EntryKind {
    Indexed,
    Direct,
}

impl EntryKind {
    const fn entry_size(self) -> usize {
        match self {
            Self::Indexed => std::mem::size_of::<RawRwEntryIdx>(),
            Self::Direct => std::mem::size_of::<RawRwEntryF>(),
        }
    }
}

/// The entry CSR as an untyped byte slab plus its representation tag: one
/// storage currency for both entry layouts, so a retired slab is reusable as
/// a later round's output across the Indexed→Direct deref transition.
struct EntrySlab {
    buffer: OwnedDeviceBuffer<u8>,
    kind: EntryKind,
}

impl EntrySlab {
    /// The slab's first `count` entries as `T` (callers pass the type
    /// matching `kind`); `None` when the slab is too short.
    fn typed<T: Copy>(&self, count: usize) -> Option<&[T]> {
        let bytes = count.checked_mul(std::mem::size_of::<T>())?;
        let slice = self.buffer.as_slice().get(..bytes)?;
        // SAFETY: the slab base is page-aligned (mmap / page-aligned
        // backings only), covering `T`'s alignment; both entry structs are
        // repr(C) with explicit pad fields (the size asserts pin the
        // no-implicit-padding layout), so any byte pattern is a valid `T`.
        Some(unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<T>(), count) })
    }
}

/// A raw-entry `Vec`'s bytes (for slab fills). Both entry structs carry
/// explicit, always-initialized pad fields, so every byte is defined.
fn entry_bytes(entries: &[RawRwEntryIdx]) -> &[u8] {
    // SAFETY: repr(C) POD with explicit pad fields; length is exact.
    unsafe {
        std::slice::from_raw_parts(
            entries.as_ptr().cast::<u8>(),
            std::mem::size_of_val(entries),
        )
    }
}

struct DeviceBindPlan {
    out_offsets: OwnedDeviceBuffer<u32>,
    new_entries: OwnedDeviceBuffer<u8>,
    new_kind: EntryKind,
    new_count: usize,
    next_ra_table: Vec<Fr>,
    next_wa_table: Vec<Fr>,
}

/// One detached round in flight (committed, not yet waited). `pass` is
/// declared first so a drop without a wait settles the GPU before the plan's
/// buffers free.
struct DeviceFlight {
    pass: DetachedPass,
    /// `Some` for a fused bind+message round (installed on collect); `None`
    /// for a message-only round.
    plan: Option<DeviceBindPlan>,
    num_tgs: usize,
}

/// The message dispatch's input tables: the pre-bind state for a standalone
/// message, the just-encoded bind plan's outputs for a fused round.
struct MessageSource<'b> {
    entries: &'b OwnedDeviceBuffer<u8>,
    kind: EntryKind,
    ra_table: &'b [Fr],
    wa_table: &'b [Fr],
    offsets: &'b OwnedDeviceBuffer<u32>,
    inc: &'b OwnedDeviceBuffer<Fr>,
    pairs: usize,
}

struct DeviceRegistersRwState {
    context: &'static MetalContext,
    entries: EntrySlab,
    entry_count: usize,
    /// The previous round's retired input slab, recycled as a later round's
    /// output when arena reuse is on (never the current round's input, so no
    /// aliasing with in-flight reads).
    spare: Option<OwnedDeviceBuffer<u8>>,
    /// `arena_reuse_enabled() && mmap_bind_buffers_enabled()`, cached.
    arena: bool,
    /// `mmap_bind_buffers_enabled()`, cached.
    mmap_bind: bool,
    row_offsets: OwnedDeviceBuffer<u32>,
    rows: usize,
    counts: OwnedDeviceBuffer<u32>,
    counts_valid: bool,
    inc: RoundTable,
    partials: Partials,
    ra_table: Vec<Fr>,
    wa_table: Vec<Fr>,
}

impl DeviceRegistersRwState {
    fn new(
        context: &'static MetalContext,
        entries: Vec<RawRwEntryIdx>,
        row_offsets: Vec<u32>,
        inc: Vec<Fr>,
        gamma: Fr,
    ) -> Result<Self, MetalError> {
        let rows = inc.len();
        let mut slab = MmapVec::<u8>::zeroed(std::mem::size_of_val(entries.as_slice()));
        slab.copy_from_slice(entry_bytes(&entries));
        let mmap_bind = mmap_bind_buffers_enabled();
        Ok(Self {
            context,
            entry_count: entries.len(),
            entries: EntrySlab {
                buffer: context.own_mmap(slab)?,
                kind: EntryKind::Indexed,
            },
            spare: None,
            arena: mmap_bind && arena_reuse_enabled(),
            mmap_bind,
            row_offsets: context.own_vec(row_offsets)?,
            rows,
            counts: context
                .own_page_aligned(PageAlignedVec::from_elem(0_u32, (rows / 2).max(1)))?,
            counts_valid: false,
            inc: RoundTable::new(context, inc)?,
            partials: Partials::new(context, 2, (rows / 2).max(1))?,
            ra_table: vec![Fr::from_u64(0), gamma, gamma * gamma, gamma + gamma * gamma],
            wa_table: vec![Fr::from_u64(0), Fr::from_u64(1)],
        })
    }

    fn new_from_registers(
        context: &'static MetalContext,
        registers: RegisterBuildInputs<'_>,
        row_offsets: Vec<u32>,
        inc: Vec<Fr>,
        gamma: Fr,
    ) -> Result<Self, MetalError> {
        let rows = inc.len();
        let entry_count = row_offsets.last().copied().unwrap_or_default() as usize;
        let row_offsets = context.own_vec(row_offsets)?;
        let entries = context.own_mmap(MmapVec::<u8>::zeroed(
            entry_count * EntryKind::Indexed.entry_size(),
        ))?;
        let rs1_index = context.wrap_slice(registers.rs1_index)?;
        let rs2_index = context.wrap_slice(registers.rs2_index)?;
        let rd_index = context.wrap_slice(registers.rd_index)?;
        let rs1_value = context.wrap_slice(registers.rs1_value)?;
        let rs2_value = context.wrap_slice(registers.rs2_value)?;
        let rd_pre_value = context.wrap_slice(registers.rd_pre_value)?;
        let rd_post_value = context.wrap_slice(registers.rd_post_value)?;
        {
            let _span = tracing::info_span!("RegRw::prepare_gpu").entered();
            let offset_buffer = row_offsets.device_buffer();
            let entry_buffer = entries.device_buffer();
            let mut pass = context.begin_pass()?;
            pass.dispatch(
                KernelId::RegRwBuild,
                &[rows as u32],
                &[
                    &rs1_index,
                    &rs2_index,
                    &rd_index,
                    &rs1_value,
                    &rs2_value,
                    &rd_pre_value,
                    &rd_post_value,
                    &offset_buffer,
                    &entry_buffer,
                ],
                rows,
            );
            pass.run()?;
        }
        let mmap_bind = mmap_bind_buffers_enabled();
        Ok(Self {
            context,
            entries: EntrySlab {
                buffer: entries,
                kind: EntryKind::Indexed,
            },
            entry_count,
            spare: None,
            arena: mmap_bind && arena_reuse_enabled(),
            mmap_bind,
            row_offsets,
            rows,
            counts: context
                .own_page_aligned(PageAlignedVec::from_elem(0_u32, (rows / 2).max(1)))?,
            counts_valid: false,
            inc: RoundTable::new(context, inc)?,
            partials: Partials::new(context, 2, (rows / 2).max(1))?,
            ra_table: vec![Fr::from_u64(0), gamma, gamma * gamma, gamma + gamma * gamma],
            wa_table: vec![Fr::from_u64(0), Fr::from_u64(1)],
        })
    }

    /// Encode one message dispatch over `source` into `pass`. Returns the
    /// dispatch's threadgroup count (the partials rows to sum).
    fn encode_message<'b>(
        &'b self,
        pass: &mut ComputePass<'_, 'b>,
        source: MessageSource<'b>,
        gruen: &'b GruenSplitEqPolynomial<Fr>,
    ) -> Result<usize, MetalError> {
        let num_tgs = num_threadgroups(source.pairs);
        let e_in = gruen.e_in_current();
        let e_out = gruen.e_out_current();
        let e_in_buffer = self.context.wrap_slice(fr_as_u32s(e_in))?;
        let e_out_buffer = self.context.wrap_slice(fr_as_u32s(e_out))?;
        let offset_buffer = source.offsets.device_buffer();
        let inc_buffer = source.inc.device_buffer();
        let partial_buffer = self.partials.buffer().device_buffer();
        let count_buffer = self.counts.device_buffer();
        let params = [
            source.pairs as u32,
            num_tgs as u32,
            e_in.len().trailing_zeros(),
            e_in.len() as u32,
        ];
        let entry_buffer = source.entries.device_buffer();
        match source.kind {
            EntryKind::Indexed => {
                let ra_buffer = self.context.wrap_slice(fr_as_u32s(source.ra_table))?;
                let wa_buffer = self.context.wrap_slice(fr_as_u32s(source.wa_table))?;
                pass.dispatch(
                    KernelId::RegRwMessageIdx,
                    &params,
                    &[
                        &entry_buffer,
                        &offset_buffer,
                        &ra_buffer,
                        &wa_buffer,
                        &inc_buffer,
                        &e_out_buffer,
                        &e_in_buffer,
                        &partial_buffer,
                        &count_buffer,
                    ],
                    source.pairs,
                );
            }
            EntryKind::Direct => {
                pass.dispatch(
                    KernelId::RegRwMessageF,
                    &params,
                    &[
                        &entry_buffer,
                        &offset_buffer,
                        &inc_buffer,
                        &e_out_buffer,
                        &e_in_buffer,
                        &partial_buffer,
                        &count_buffer,
                    ],
                    source.pairs,
                );
            }
        }
        Ok(num_tgs)
    }

    fn message(&mut self, gruen: &GruenSplitEqPolynomial<Fr>) -> Result<[Fr; 2], MetalError> {
        let _span = tracing::info_span!("RegRw::message").entered();
        let pairs = self.rows / 2;
        let num_tgs;
        {
            let mut pass = self.context.begin_pass()?;
            num_tgs = self.encode_message(
                &mut pass,
                MessageSource {
                    entries: &self.entries.buffer,
                    kind: self.entries.kind,
                    ra_table: &self.ra_table,
                    wa_table: &self.wa_table,
                    offsets: &self.row_offsets,
                    inc: self.inc.cur(),
                    pairs,
                },
                gruen,
            )?;
            tracing::info_span!("RegRw::msg_run").in_scope(|| pass.run())?;
        }
        testing::note_device_round();
        self.counts_valid = true;
        let sums = tracing::info_span!("RegRw::msg_sums").in_scope(|| self.partials.sums(num_tgs));
        Ok([sums[0], sums[1]])
    }

    /// [`message`](Self::message) committed as a detached pass: the CB
    /// executes while the caller's batch runs its synchronous members.
    fn launch_message(
        &self,
        gruen: &GruenSplitEqPolynomial<Fr>,
    ) -> Result<DeviceFlight, MetalError> {
        let pairs = self.rows / 2;
        let mut pass = self.context.begin_pass()?;
        let num_tgs = self.encode_message(
            &mut pass,
            MessageSource {
                entries: &self.entries.buffer,
                kind: self.entries.kind,
                ra_table: &self.ra_table,
                wa_table: &self.wa_table,
                offsets: &self.row_offsets,
                inc: self.inc.cur(),
                pairs,
            },
            gruen,
        )?;
        // SAFETY: every no-copy-wrapped backing outlives the wait — state-
        // owned (entries slab, row_offsets, inc, partials, counts, ra/wa
        // tables) or caller-owned (`gruen`, alive and unmutated until the
        // flight collects); copied wraps are retained by the command buffer.
        // The host touches none of them before `collect_flight`.
        let pass = unsafe { pass.commit().detach() };
        Ok(DeviceFlight {
            pass,
            plan: None,
            num_tgs,
        })
    }

    /// Wait on a detached round, install its plan (fused rounds), and read
    /// back the two message sums.
    fn collect_flight(&mut self, flight: DeviceFlight) -> Result<[Fr; 2], MetalError> {
        let DeviceFlight {
            pass,
            plan,
            num_tgs,
        } = flight;
        if plan.is_some() {
            tracing::info_span!("RegRw::bind_msg_run").in_scope(|| pass.wait())?;
        } else {
            tracing::info_span!("RegRw::msg_run").in_scope(|| pass.wait())?;
        }
        testing::note_device_round();
        if let Some(plan) = plan {
            tracing::info_span!("RegRw::install").in_scope(|| self.install_bind(plan));
        }
        self.counts_valid = true;
        let sums = tracing::info_span!("RegRw::msg_sums").in_scope(|| self.partials.sums(num_tgs));
        Ok([sums[0], sums[1]])
    }

    fn scanned_offsets(&self) -> Result<(Vec<u32>, usize), MetalError> {
        let _span = tracing::info_span!("RegRw::scan_offsets").entered();
        if !self.counts_valid {
            return Err(MetalError::Execution(
                "registers read/write bind without a preceding message".to_string(),
            ));
        }
        let pairs = self.rows / 2;
        let mut offsets = Vec::with_capacity(pairs + 1);
        offsets.push(0_u32);
        let mut total = 0_u32;
        for &count in &self.counts.as_slice()[..pairs] {
            total = total.checked_add(count).ok_or_else(|| {
                MetalError::Execution("registers read/write entry count overflow".to_string())
            })?;
            offsets.push(total);
        }
        Ok((offsets, total as usize))
    }

    /// A fresh (kernel-zeroed) bind output slab of `needed` bytes. With
    /// arena reuse on, new mmap slabs take Direct-width headroom — virtual
    /// pages are free until touched, and the deref round then fits in the
    /// recycled slab without a growth remap.
    fn alloc_entries(
        &self,
        needed: usize,
        new_count: usize,
    ) -> Result<OwnedDeviceBuffer<u8>, MetalError> {
        if !self.mmap_bind {
            return self
                .context
                .own_page_aligned(PageAlignedVec::from_elem(0u8, needed));
        }
        if self.arena {
            let headroom = new_count * EntryKind::Direct.entry_size();
            if headroom > needed {
                // An oversized wrap can exceed the device's buffer cap —
                // fall through to the exact size.
                if let Ok(buffer) = self.context.own_mmap(MmapVec::<u8>::zeroed(headroom)) {
                    return Ok(buffer);
                }
            }
        }
        self.context.own_mmap(MmapVec::<u8>::zeroed(needed))
    }

    fn plan_bind(&mut self, challenge: Fr, final_bind: bool) -> Result<DeviceBindPlan, MetalError> {
        let (offsets, new_count) = self.scanned_offsets()?;
        let out_offsets = self.context.own_vec(offsets)?;
        let deref = self.entries.kind == EntryKind::Indexed
            && (self.ra_table.len() >= RA_TABLE_DEREF_LEN || final_bind);
        let new_kind = if self.entries.kind == EntryKind::Indexed && !deref {
            EntryKind::Indexed
        } else {
            EntryKind::Direct
        };
        let alloc_span = tracing::info_span!("RegRw::alloc_entries", new_count).entered();
        let needed = new_count * new_kind.entry_size();
        let new_entries = match self.spare.take() {
            Some(spare) if spare.len() >= needed => spare,
            _ => self.alloc_entries(needed, new_count)?,
        };
        drop(alloc_span);
        let (next_ra_table, next_wa_table) = if new_kind == EntryKind::Indexed {
            (
                expand_lookup_table(&self.ra_table, challenge),
                expand_lookup_table(&self.wa_table, challenge),
            )
        } else {
            (Vec::new(), Vec::new())
        };
        Ok(DeviceBindPlan {
            out_offsets,
            new_entries,
            new_kind,
            new_count,
            next_ra_table,
            next_wa_table,
        })
    }

    fn encode_bind<'b>(
        &'b self,
        pass: &mut ComputePass<'_, 'b>,
        plan: &'b DeviceBindPlan,
        challenge: Fr,
    ) -> Result<(), MetalError> {
        let pairs = self.rows / 2;
        let offset_buffer = self.row_offsets.device_buffer();
        let out_offset_buffer = plan.out_offsets.device_buffer();
        let entry_buffer = self.entries.buffer.device_buffer();
        let out_buffer = plan.new_entries.device_buffer();
        match (self.entries.kind, plan.new_kind) {
            (EntryKind::Indexed, EntryKind::Indexed) => {
                let mut params = vec![
                    pairs as u32,
                    self.ra_table.len().trailing_zeros(),
                    self.wa_table.len().trailing_zeros(),
                ];
                params.extend_from_slice(&fr_to_u32_limbs(challenge));
                pass.dispatch(
                    KernelId::RegRwBindIdx,
                    &params,
                    &[
                        &entry_buffer,
                        &offset_buffer,
                        &out_offset_buffer,
                        &out_buffer,
                    ],
                    pairs,
                );
            }
            (EntryKind::Indexed, EntryKind::Direct) => {
                let ra_buffer = self.context.wrap_slice(fr_as_u32s(&self.ra_table))?;
                let wa_buffer = self.context.wrap_slice(fr_as_u32s(&self.wa_table))?;
                let mut params = vec![pairs as u32];
                params.extend_from_slice(&fr_to_u32_limbs(challenge));
                pass.dispatch(
                    KernelId::RegRwBindIdxToF,
                    &params,
                    &[
                        &entry_buffer,
                        &offset_buffer,
                        &out_offset_buffer,
                        &out_buffer,
                        &ra_buffer,
                        &wa_buffer,
                    ],
                    pairs,
                );
            }
            (EntryKind::Direct, EntryKind::Direct) => {
                let mut params = vec![pairs as u32];
                params.extend_from_slice(&fr_to_u32_limbs(challenge));
                pass.dispatch(
                    KernelId::RegRwBindF,
                    &params,
                    &[
                        &entry_buffer,
                        &offset_buffer,
                        &out_offset_buffer,
                        &out_buffer,
                    ],
                    pairs,
                );
            }
            (EntryKind::Direct, EntryKind::Indexed) => {
                return Err(MetalError::Execution(
                    "registers read/write coefficient representation regressed".to_string(),
                ));
            }
        }
        let inc_buffer = self.inc.cur().device_buffer();
        let out_inc_buffer = self.inc.nxt().device_buffer();
        let mut bind_params = vec![pairs as u32];
        bind_params.extend_from_slice(&fr_to_u32_limbs(challenge));
        pass.dispatch(
            KernelId::FrBind,
            &bind_params,
            &[&inc_buffer, &out_inc_buffer],
            pairs,
        );
        Ok(())
    }

    fn install_bind(&mut self, plan: DeviceBindPlan) {
        let retired = std::mem::replace(&mut self.entries.buffer, plan.new_entries);
        self.entries.kind = plan.new_kind;
        if self.arena {
            // The retired input becomes a later round's output slab; it is
            // never the NEXT round's input, so recycling cannot alias a
            // concurrent read.
            self.spare = Some(retired);
        }
        self.entry_count = plan.new_count;
        self.row_offsets = plan.out_offsets;
        self.ra_table = plan.next_ra_table;
        self.wa_table = plan.next_wa_table;
        self.inc.swap();
        self.rows /= 2;
    }

    fn bind(&mut self, challenge: Fr, final_bind: bool) -> Result<(), MetalError> {
        let plan = self.plan_bind(challenge, final_bind)?;
        {
            let mut pass = self.context.begin_pass()?;
            self.encode_bind(&mut pass, &plan, challenge)?;
            tracing::info_span!("RegRw::bind_run").in_scope(|| pass.run())?;
        }
        testing::note_device_round();
        tracing::info_span!("RegRw::install").in_scope(|| self.install_bind(plan));
        self.counts_valid = false;
        Ok(())
    }

    /// Encode + commit one fused bind+message round without blocking. The
    /// plan rides in the returned flight and installs on collect, so a
    /// failed round leaves the pre-bind state intact for the host recompute.
    fn launch_bind_and_message(
        &mut self,
        challenge: Fr,
        gruen: &GruenSplitEqPolynomial<Fr>,
    ) -> Result<DeviceFlight, MetalError> {
        let plan = self.plan_bind(challenge, false)?;
        let pairs = (self.rows / 2) / 2;
        let mut pass = self.context.begin_pass()?;
        self.encode_bind(&mut pass, &plan, challenge)?;
        pass.buffer_barrier();
        let num_tgs = self.encode_message(
            &mut pass,
            MessageSource {
                entries: &plan.new_entries,
                kind: plan.new_kind,
                ra_table: &plan.next_ra_table,
                wa_table: &plan.next_wa_table,
                offsets: &plan.out_offsets,
                inc: self.inc.nxt(),
                pairs,
            },
            gruen,
        )?;
        // SAFETY: every no-copy-wrapped backing outlives the wait — state-
        // owned (entries slab, row_offsets, inc ping-pong, partials, counts,
        // ra/wa tables), flight-owned (the plan's out_offsets, output slab,
        // and next ra/wa tables: heap/mmap allocations are address-stable
        // across the plan's move into the flight), or caller-owned (`gruen`,
        // alive and unmutated until the flight collects); copied wraps are
        // retained by the command buffer. The host touches none of them
        // before `collect_flight`.
        let pass = unsafe { pass.commit().detach() };
        Ok(DeviceFlight {
            pass,
            plan: Some(plan),
            num_tgs,
        })
    }

    fn bind_and_message(
        &mut self,
        challenge: Fr,
        gruen: &GruenSplitEqPolynomial<Fr>,
    ) -> Result<[Fr; 2], MetalError> {
        let flight = self.launch_bind_and_message(challenge, gruen)?;
        self.collect_flight(flight)
    }

    fn into_cycle_state(self) -> Result<(Vec<BoundRegistersRwEntry<Fr>>, Fr), ()> {
        if self.rows != 1 || self.inc.cur_slice(1).is_empty() {
            return Err(());
        }
        if self.entries.kind != EntryKind::Direct {
            return Err(());
        }
        let entries = self
            .entries
            .typed::<RawRwEntryF>(self.entry_count)
            .ok_or(())?;
        Ok((
            entries
                .iter()
                .map(|entry| BoundRegistersRwEntry {
                    col: entry.col,
                    val: entry.val,
                    ra: entry.ra,
                    wa: entry.wa,
                })
                .collect(),
            self.inc.cur_slice(1)[0],
        ))
    }

    fn into_partial_state(self) -> Result<(Vec<RegistersRwCycleEntry<Fr>>, Polynomial<Fr>), ()> {
        let offsets = self.row_offsets.as_slice().get(..=self.rows).ok_or(())?;
        if offsets[self.rows] as usize != self.entry_count {
            return Err(());
        }
        let mut out = Vec::with_capacity(self.entry_count);
        match self.entries.kind {
            EntryKind::Indexed => {
                let entries = self
                    .entries
                    .typed::<RawRwEntryIdx>(self.entry_count)
                    .ok_or(())?;
                for row in 0..self.rows {
                    let start = offsets[row] as usize;
                    let end = offsets[row + 1] as usize;
                    for entry in entries.get(start..end).ok_or(())? {
                        out.push(RegistersRwCycleEntry {
                            row,
                            col: entry.col,
                            prev_val: entry.prev_val,
                            next_val: entry.next_val,
                            val: entry.val,
                            ra: *self.ra_table.get(entry.ra as usize).ok_or(())?,
                            wa: *self.wa_table.get(entry.wa as usize).ok_or(())?,
                        });
                    }
                }
            }
            EntryKind::Direct => {
                let entries = self
                    .entries
                    .typed::<RawRwEntryF>(self.entry_count)
                    .ok_or(())?;
                for row in 0..self.rows {
                    let start = offsets[row] as usize;
                    let end = offsets[row + 1] as usize;
                    for entry in entries.get(start..end).ok_or(())? {
                        out.push(RegistersRwCycleEntry {
                            row,
                            col: entry.col,
                            prev_val: entry.prev_val,
                            next_val: entry.next_val,
                            val: entry.val,
                            ra: entry.ra,
                            wa: entry.wa,
                        });
                    }
                }
            }
        }
        let inc = Polynomial::new(self.inc.cur_slice(self.rows).to_vec());
        Ok((out, inc))
    }
}

pub struct MetalRegistersReadWriteChecking {
    fallback: OptimizedRegistersReadWrite,
}

impl MetalRegistersReadWriteChecking {
    pub fn new() -> Self {
        Self {
            fallback: OptimizedRegistersReadWrite,
        }
    }
}

impl Default for MetalRegistersReadWriteChecking {
    fn default() -> Self {
        Self::new()
    }
}

impl PrepareKernel<Fr, RegistersReadWriteChecking<Fr>> for MetalRegistersReadWriteChecking {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<Fr>,
        inputs: ProverInputs<'_, Fr, RegistersReadWriteChecking<Fr>>,
    ) -> Result<
        Box<dyn SumcheckKernel<Fr, Relation = RegistersReadWriteChecking<Fr>>>,
        KernelError<Fr>,
    > {
        let dimensions = inputs.relation.register_dimensions();
        if dimensions.phase1_num_rounds() != dimensions.log_t() {
            return Err(KernelError::Unsupported {
                reason: "Metal registers read-write checking supports only the default read-write config",
            });
        }
        let log_t = dimensions.log_t();
        let log_k = dimensions.log_k();
        if log_t == 0 {
            return Err(KernelError::Unsupported {
                reason: "Metal registers read-write checking requires at least one cycle round",
            });
        }
        let r_cycle = &inputs.points.rd_write_value;
        if r_cycle.len() != log_t {
            return Err(KernelError::InvariantViolation {
                reason: "registers read-write input point has the wrong variable count",
            });
        }
        let cycles = 1usize << log_t;
        if !metal_gate(KIND, cycles) || log_t >= 32 {
            return self.fallback.prepare(session, witness, inputs);
        }
        let context = match MetalContext::global() {
            Ok(context) => context,
            Err(error) => {
                tracing::warn!(slot = KIND, %error, "no device context; using optimized fallback");
                return self.fallback.prepare(session, witness, inputs);
            }
        };

        let record = TraceRecord::shared(session, witness, log_t)?;
        if record.len() != cycles {
            return Err(KernelError::TableSizeMismatch {
                table: format!("{:?}", rd_inc_read_write()),
                expected: cycles,
                got: record.len(),
            });
        }
        let registers = Arc::clone(&record.registers);
        let ram = Arc::clone(&record.ram_values);
        drop(record);
        TraceRecord::release(session);
        crate::optimized::opening::park_opening_increments(session, &registers, &ram);

        let register_inputs = RegisterBuildInputs::from(registers.as_ref());
        let prepared = if gpu_prepare_enabled() {
            let metadata = tracing::info_span!("RegRw::prepare_meta")
                .in_scope(|| build_metal_register_metadata(register_inputs));
            if metadata.offsets.last().copied().unwrap_or_default() == 0 {
                return self.fallback.prepare(session, witness, inputs);
            }
            DeviceRegistersRwState::new_from_registers(
                context,
                register_inputs,
                metadata.offsets,
                metadata.inc,
                inputs.challenges.gamma,
            )
            .map(|device| {
                (
                    device,
                    metadata.rs1_indices,
                    metadata.rs2_indices,
                    metadata.rd_indices,
                )
            })
        } else {
            let tables = build_metal_register_tables_serial(register_inputs);
            if tables.entries.is_empty() {
                return self.fallback.prepare(session, witness, inputs);
            }
            DeviceRegistersRwState::new(
                context,
                tables.entries,
                tables.offsets,
                tables.inc,
                inputs.challenges.gamma,
            )
            .map(|device| {
                (
                    device,
                    tables.rs1_indices,
                    tables.rs2_indices,
                    tables.rd_indices,
                )
            })
        };
        drop(registers);
        let (device, rs1_indices, rs2_indices, rd_indices) = match prepared {
            Ok(prepared) => prepared,
            Err(error) => {
                tracing::warn!(slot = KIND, %error, "device preparation failed; using optimized fallback");
                return self.fallback.prepare(session, witness, inputs);
            }
        };
        session.park(SharedRdIndices(rd_indices));
        Ok(Box::new(MetalRegistersRwKernel {
            log_t,
            log_k,
            fused_rounds: fused_rounds_enabled(),
            overlap: overlap_rounds_enabled(),
            in_flight: None,
            device: Some(device),
            gruen: Some(GruenSplitEqPolynomial::new(
                r_cycle,
                BindingOrder::LowToHigh,
            )),
            host: None,
            rs1_indices: Some(rs1_indices),
            rs2_indices: Some(rs2_indices),
            bound_challenges: Some(Vec::with_capacity(log_t + log_k)),
            rounds_bound: 0,
        }))
    }
}

fn missing_device_state() -> SumcheckError<Fr> {
    SumcheckError::MissingEvaluationSource {
        kind: "Metal registers read/write device state",
    }
}

/// One overlapped round in flight: the detached device pass plus the host
/// state to install on a successful collect. Declared before `device`/`gruen`
/// in the kernel so a drop without a collect settles the GPU first.
struct RwFlight {
    flight: DeviceFlight,
    /// The eq tables after this round's fold; installed on collect so a
    /// failed round recomputes host-side from the intact pre-bind state.
    /// `None` for a message-only (first active) round.
    next_gruen: Option<GruenSplitEqPolynomial<Fr>>,
    /// The challenge this round bound (`None` for message-only rounds).
    challenge: Option<Fr>,
}

struct MetalRegistersRwKernel {
    log_t: usize,
    log_k: usize,
    fused_rounds: bool,
    overlap: bool,
    in_flight: Option<RwFlight>,
    device: Option<DeviceRegistersRwState>,
    gruen: Option<GruenSplitEqPolynomial<Fr>>,
    host: Option<ReadWriteKernel<Fr>>,
    rs1_indices: Option<Vec<Option<u8>>>,
    rs2_indices: Option<Vec<Option<u8>>>,
    bound_challenges: Option<Vec<Fr>>,
    rounds_bound: usize,
}

impl MetalRegistersRwKernel {
    fn fallback_to_host(&mut self) -> Result<(), SumcheckError<Fr>> {
        let device = self.device.take().ok_or_else(missing_device_state)?;
        let (entries, inc) = device
            .into_partial_state()
            .map_err(|()| missing_device_state())?;
        self.host = Some(ReadWriteKernel::from_partial_cycle_state(
            self.log_t,
            self.log_k,
            entries,
            self.gruen.take().ok_or_else(missing_device_state)?,
            inc,
            self.rs1_indices.take().ok_or_else(missing_device_state)?,
            self.rs2_indices.take().ok_or_else(missing_device_state)?,
            self.bound_challenges
                .take()
                .ok_or_else(missing_device_state)?,
            self.rounds_bound,
        ));
        Ok(())
    }

    fn transition(&mut self) -> Result<(), SumcheckError<Fr>> {
        let device = self.device.take().ok_or_else(missing_device_state)?;
        let (entries, inc_scalar) = device
            .into_cycle_state()
            .map_err(|()| missing_device_state())?;
        self.host = Some(ReadWriteKernel::from_cycle_state(
            self.log_t,
            self.log_k,
            entries,
            self.gruen.take().ok_or_else(missing_device_state)?,
            inc_scalar,
            self.rs1_indices.take().ok_or_else(missing_device_state)?,
            self.rs2_indices.take().ok_or_else(missing_device_state)?,
            self.bound_challenges
                .take()
                .ok_or_else(missing_device_state)?,
        ));
        Ok(())
    }
}

impl ProveRounds<Fr> for MetalRegistersRwKernel {
    fn num_rounds(&self) -> usize {
        self.log_t + self.log_k
    }

    fn prove_round(
        &mut self,
        bind: Option<Fr>,
        round: usize,
        previous_claim: Fr,
    ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
        if let Some(host) = &mut self.host {
            return host.prove_round(bind, round, previous_claim);
        }
        if let Some(challenge) = bind {
            let final_bind = self.rounds_bound + 1 == self.log_t;
            if self.fused_rounds && !final_bind {
                let mut next_gruen = self
                    .gruen
                    .as_ref()
                    .ok_or_else(missing_device_state)?
                    .clone();
                next_gruen.bind(challenge);
                let result = self
                    .device
                    .as_mut()
                    .ok_or_else(missing_device_state)?
                    .bind_and_message(challenge, &next_gruen);
                let [q_0, q_inf] = match result {
                    Ok(values) => values,
                    Err(error) => {
                        tracing::warn!(slot = KIND, %error, "fused device round failed; finishing on CPU");
                        self.fallback_to_host()?;
                        return self
                            .host
                            .as_mut()
                            .ok_or_else(missing_device_state)?
                            .prove_round(Some(challenge), round, previous_claim);
                    }
                };
                self.gruen = Some(next_gruen);
                self.bound_challenges
                    .as_mut()
                    .ok_or_else(missing_device_state)?
                    .push(challenge);
                self.rounds_bound += 1;
                return Ok(self
                    .gruen
                    .as_ref()
                    .ok_or_else(missing_device_state)?
                    .gruen_poly_deg_3(q_0, q_inf, previous_claim));
            }
            let result = self
                .device
                .as_mut()
                .ok_or_else(missing_device_state)?
                .bind(challenge, final_bind);
            if let Err(error) = result {
                tracing::warn!(slot = KIND, %error, "device bind failed; finishing on CPU");
                self.fallback_to_host()?;
                return self
                    .host
                    .as_mut()
                    .ok_or_else(missing_device_state)?
                    .prove_round(Some(challenge), round, previous_claim);
            }
            self.gruen
                .as_mut()
                .ok_or_else(missing_device_state)?
                .bind(challenge);
            self.bound_challenges
                .as_mut()
                .ok_or_else(missing_device_state)?
                .push(challenge);
            self.rounds_bound += 1;
            if self.rounds_bound == self.log_t {
                self.transition()?;
                return self
                    .host
                    .as_mut()
                    .ok_or_else(missing_device_state)?
                    .prove_round(None, round, previous_claim);
            }
        }
        let result = self
            .device
            .as_mut()
            .ok_or_else(missing_device_state)?
            .message(self.gruen.as_ref().ok_or_else(missing_device_state)?);
        let [q_0, q_inf] = match result {
            Ok(values) => values,
            Err(error) => {
                tracing::warn!(slot = KIND, %error, "device message failed; finishing on CPU");
                self.fallback_to_host()?;
                return self
                    .host
                    .as_mut()
                    .ok_or_else(missing_device_state)?
                    .prove_round(None, round, previous_claim);
            }
        };
        Ok(self
            .gruen
            .as_ref()
            .ok_or_else(missing_device_state)?
            .gruen_poly_deg_3(q_0, q_inf, previous_claim))
    }

    fn begin_round(
        &mut self,
        bind: Option<Fr>,
        _round: usize,
        _previous_claim: Fr,
    ) -> Result<bool, SumcheckError<Fr>> {
        if !self.overlap || self.host.is_some() || self.device.is_none() {
            return Ok(false);
        }
        // The final cycle bind transitions to the host tail — synchronous.
        if bind.is_some() && self.rounds_bound + 1 == self.log_t {
            return Ok(false);
        }
        let launched = match bind {
            Some(challenge) => {
                let mut next_gruen = self
                    .gruen
                    .as_ref()
                    .ok_or_else(missing_device_state)?
                    .clone();
                next_gruen.bind(challenge);
                self.device
                    .as_mut()
                    .ok_or_else(missing_device_state)?
                    .launch_bind_and_message(challenge, &next_gruen)
                    .map(|flight| RwFlight {
                        flight,
                        next_gruen: Some(next_gruen),
                        challenge: Some(challenge),
                    })
            }
            None => self
                .device
                .as_ref()
                .ok_or_else(missing_device_state)?
                .launch_message(self.gruen.as_ref().ok_or_else(missing_device_state)?)
                .map(|flight| RwFlight {
                    flight,
                    next_gruen: None,
                    challenge: None,
                }),
        };
        match launched {
            Ok(flight) => {
                self.in_flight = Some(flight);
                Ok(true)
            }
            Err(error) => {
                // Nothing committed: the round runs synchronously instead.
                tracing::warn!(slot = KIND, %error, "detached launch failed; round runs synchronously");
                Ok(false)
            }
        }
    }

    fn collect_round(
        &mut self,
        bind: Option<Fr>,
        round: usize,
        previous_claim: Fr,
    ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
        let Some(flight) = self.in_flight.take() else {
            return self.prove_round(bind, round, previous_claim);
        };
        let collected = self
            .device
            .as_mut()
            .ok_or_else(missing_device_state)?
            .collect_flight(flight.flight);
        match collected {
            Ok([q_0, q_inf]) => {
                if let Some(next_gruen) = flight.next_gruen {
                    self.gruen = Some(next_gruen);
                }
                if let Some(challenge) = flight.challenge {
                    self.bound_challenges
                        .as_mut()
                        .ok_or_else(missing_device_state)?
                        .push(challenge);
                    self.rounds_bound += 1;
                }
                Ok(self
                    .gruen
                    .as_ref()
                    .ok_or_else(missing_device_state)?
                    .gruen_poly_deg_3(q_0, q_inf, previous_claim))
            }
            Err(error) => {
                // The failed round's plan never installed — the pre-bind
                // state is intact and the host redoes the whole round.
                tracing::warn!(slot = KIND, %error, "detached device round failed; finishing on CPU");
                self.fallback_to_host()?;
                self.host
                    .as_mut()
                    .ok_or_else(missing_device_state)?
                    .prove_round(bind, round, previous_claim)
            }
        }
    }

    fn finish_rounds(&mut self, bind: Fr) -> Result<(), SumcheckError<Fr>> {
        if let Some(host) = &mut self.host {
            return host.finish_rounds(bind);
        }
        let result = self
            .device
            .as_mut()
            .ok_or_else(missing_device_state)?
            .bind(bind, true);
        if let Err(error) = result {
            tracing::warn!(slot = KIND, %error, "final device bind failed; finishing on CPU");
            self.fallback_to_host()?;
            return self
                .host
                .as_mut()
                .ok_or_else(missing_device_state)?
                .finish_rounds(bind);
        }
        self.gruen
            .as_mut()
            .ok_or_else(missing_device_state)?
            .bind(bind);
        self.bound_challenges
            .as_mut()
            .ok_or_else(missing_device_state)?
            .push(bind);
        self.rounds_bound += 1;
        self.transition()
    }
}

impl SumcheckKernel<Fr> for MetalRegistersRwKernel {
    type Relation = RegistersReadWriteChecking<Fr>;

    fn output_claims(
        &mut self,
        inputs: &SumcheckInputClaims<Fr, Self::Relation>,
    ) -> Result<RegistersReadWriteOutputClaims<Fr>, SumcheckKernelError<Fr>> {
        let remaining = self.num_rounds().saturating_sub(self.rounds_bound);
        self.host
            .as_mut()
            .ok_or(SumcheckKernelError::NotFullyBound { remaining })?
            .output_claims(inputs)
    }

    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<Fr, Self::Relation>,
        output_points: &SumcheckOutputPoints<Fr, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<Fr, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<Fr>> {
        self.host
            .as_ref()
            .ok_or(SumcheckKernelError::NotFullyBound {
                remaining: self.num_rounds().saturating_sub(self.rounds_bound),
            })?
            .validate_derived_tables(relation, input_points, output_points, challenges)
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module must fail loudly")]
mod tests {
    use jolt_claims::protocols::jolt::geometry::dimensions::{
        ReadWriteDimensions, REGISTER_ADDRESS_BITS,
    };
    use jolt_claims::protocols::jolt::{JoltPolynomialId, JoltVirtualPolynomial};
    use jolt_poly::Polynomial;
    use jolt_verifier::stages::stage4::registers_read_write_checking::{
        RegistersReadWriteChallenges, RegistersReadWriteInputClaims,
    };
    use jolt_witness::JoltWitnessOracle;

    use super::*;
    use crate::metal::testing::{device_dispatch_count, device_probe_count, gpu_lock};
    use crate::optimized::registers_read_write::test_support::{
        assert_kernel_parity, assert_nontrivial, challenge_sequence, structured_fixture,
    };

    #[test]
    fn registers_rw_gpu_prepare_matches_serial() {
        let _lock = gpu_lock();
        structured_fixture(257).with_plane(9, |backend| {
            let mut session = ProofSession::default();
            let record = TraceRecord::shared::<Fr>(&mut session, backend, 9).unwrap();
            let inputs = RegisterBuildInputs::from(record.registers.as_ref());
            let serial = build_metal_register_tables_serial(inputs);
            let metadata = build_metal_register_metadata_serial(inputs);
            let context = MetalContext::global().unwrap();
            let device = DeviceRegistersRwState::new_from_registers(
                context,
                inputs,
                metadata.offsets.clone(),
                metadata.inc.clone(),
                Fr::from_u64(0x5EED_1234_5678_9ABC),
            )
            .unwrap();

            assert_eq!(
                device.entries.kind,
                EntryKind::Indexed,
                "prepare must start in the indexed CSR representation"
            );
            let entries = device
                .entries
                .typed::<RawRwEntryIdx>(device.entry_count)
                .unwrap();
            assert_eq!(entries, serial.entries);
            assert_eq!(metadata.offsets, serial.offsets);
            assert_eq!(metadata.inc, serial.inc);
            assert_eq!(metadata.rs1_indices, serial.rs1_indices);
            assert_eq!(metadata.rs2_indices, serial.rs2_indices);
            assert_eq!(metadata.rd_indices, serial.rd_indices);
        });
    }

    fn run_parity(log_t: usize, seed: u64, fused: bool, arena: bool, expected_device_rounds: u64) {
        let _lock = gpu_lock();
        std::env::remove_var("JOLT_METAL_DISABLE");
        std::env::set_var("JOLT_METAL_MIN_TERMS_REGISTERS_READ_WRITE", "0");
        std::env::set_var("JOLT_REGRW_FUSED", if fused { "1" } else { "0" });
        std::env::set_var("JOLT_REGRW_ARENA", if arena { "1" } else { "0" });
        // The parity harness drives `prove_round` (the synchronous path);
        // pin the overlap switch off so the CB-count assertions stay exact.
        std::env::set_var("JOLT_REGRW_OVERLAP", "0");
        std::env::set_var("JOLT_REGRW_GPU_PREPARE", "1");
        std::env::remove_var("JOLT_REGISTERS_PREPARE_SERIAL");
        structured_fixture(1usize << log_t).with_plane(log_t, |backend| {
            let relation = RegistersReadWriteChecking::<Fr>::new(ReadWriteDimensions::new(
                log_t,
                REGISTER_ADDRESS_BITS,
                log_t,
                0,
            ));
            let r_cycle = challenge_sequence(log_t, seed ^ 0xA5A5);
            let evaluate = |polynomial: JoltVirtualPolynomial| {
                let table = JoltWitnessOracle::<Fr>::oracle_table(
                    backend,
                    JoltPolynomialId::Virtual(polynomial),
                )
                .unwrap();
                Polynomial::new(table).evaluate(&r_cycle)
            };
            let gamma = Fr::from_u64(0x5EED_1234_5678_9ABC);
            let claims = RegistersReadWriteInputClaims {
                rd_write_value: evaluate(JoltVirtualPolynomial::RdWriteValue),
                rs1_value: evaluate(JoltVirtualPolynomial::Rs1Value),
                rs2_value: evaluate(JoltVirtualPolynomial::Rs2Value),
            };
            let points = RegistersReadWriteInputClaims {
                rd_write_value: r_cycle.clone(),
                rs1_value: r_cycle.clone(),
                rs2_value: r_cycle,
            };
            let challenges = RegistersReadWriteChallenges { gamma };
            let input_claim =
                claims.rd_write_value + gamma * claims.rs1_value + gamma * gamma * claims.rs2_value;
            assert_nontrivial(input_claim);
            let round_challenges = challenge_sequence(log_t + REGISTER_ADDRESS_BITS, seed);
            let before = device_probe_count();
            let dispatches_before = device_dispatch_count();
            assert_kernel_parity(
                &MetalRegistersReadWriteChecking::new(),
                backend,
                &relation,
                &claims,
                &points,
                &challenges,
                input_claim,
                &round_challenges,
            );
            assert_eq!(
                device_probe_count() - before,
                expected_device_rounds,
                "registers read/write command-buffer schedule drifted"
            );
            assert_eq!(
                device_dispatch_count() - dispatches_before,
                3 * log_t as u64 + 1,
                "registers read/write dispatch schedule drifted"
            );
        });
    }

    #[test]
    fn registers_rw_matches_reference_index_handoff() {
        run_parity(4, 23, true, true, 5);
    }

    #[test]
    fn registers_rw_matches_reference_field_rounds() {
        run_parity(6, 47, true, true, 7);
    }

    #[test]
    fn registers_rw_legacy_schedule_matches_reference() {
        run_parity(6, 71, false, false, 12);
    }

    /// The overlap path (detached `begin_round`/`collect_round` fused
    /// rounds, arena reuse on) against the reference kernel's synchronous
    /// rounds — the schedule the batch engine actually drives in production.
    #[test]
    fn registers_rw_overlapped_matches_reference() {
        use crate::reference::ReferenceBackend;

        let log_t = 6;
        let seed = 89u64;
        let _lock = gpu_lock();
        std::env::remove_var("JOLT_METAL_DISABLE");
        std::env::set_var("JOLT_METAL_MIN_TERMS_REGISTERS_READ_WRITE", "0");
        std::env::set_var("JOLT_REGRW_FUSED", "0");
        std::env::set_var("JOLT_REGRW_ARENA", "1");
        std::env::set_var("JOLT_REGRW_OVERLAP", "1");
        std::env::set_var("JOLT_REGRW_GPU_PREPARE", "1");
        std::env::remove_var("JOLT_REGISTERS_PREPARE_SERIAL");
        structured_fixture(1usize << log_t).with_plane(log_t, |backend| {
            let relation = RegistersReadWriteChecking::<Fr>::new(ReadWriteDimensions::new(
                log_t,
                REGISTER_ADDRESS_BITS,
                log_t,
                0,
            ));
            let r_cycle = challenge_sequence(log_t, seed ^ 0xA5A5);
            let evaluate = |polynomial: JoltVirtualPolynomial| {
                let table = JoltWitnessOracle::<Fr>::oracle_table(
                    backend,
                    JoltPolynomialId::Virtual(polynomial),
                )
                .unwrap();
                Polynomial::new(table).evaluate(&r_cycle)
            };
            let gamma = Fr::from_u64(0x5EED_1234_5678_9ABC);
            let claims = RegistersReadWriteInputClaims {
                rd_write_value: evaluate(JoltVirtualPolynomial::RdWriteValue),
                rs1_value: evaluate(JoltVirtualPolynomial::Rs1Value),
                rs2_value: evaluate(JoltVirtualPolynomial::Rs2Value),
            };
            let points = RegistersReadWriteInputClaims {
                rd_write_value: r_cycle.clone(),
                rs1_value: r_cycle.clone(),
                rs2_value: r_cycle,
            };
            let challenges = RegistersReadWriteChallenges { gamma };
            let input_claim =
                claims.rd_write_value + gamma * claims.rs1_value + gamma * gamma * claims.rs2_value;
            assert_nontrivial(input_claim);
            let round_challenges = challenge_sequence(log_t + REGISTER_ADDRESS_BITS, seed);

            let inputs = || ProverInputs {
                relation: &relation,
                claims: &claims,
                points: &points,
                challenges: &challenges,
            };
            let mut reference = ReferenceBackend
                .prepare(&mut ProofSession::default(), backend, inputs())
                .unwrap();
            let mut optimized = MetalRegistersReadWriteChecking::new()
                .prepare(&mut ProofSession::default(), backend, inputs())
                .unwrap();

            let rounds = log_t + REGISTER_ADDRESS_BITS;
            let before = device_probe_count();
            let mut claim = input_claim;
            let mut launched_rounds = 0u64;
            for round in 0..rounds {
                let bind = (round > 0).then(|| round_challenges[round - 1]);
                let reference_poly = reference.prove_round(bind, round, claim).unwrap();
                launched_rounds += u64::from(optimized.begin_round(bind, round, claim).unwrap());
                let optimized_poly = optimized.collect_round(bind, round, claim).unwrap();
                assert_eq!(
                    reference_poly, optimized_poly,
                    "round {round} polynomial mismatch"
                );
                claim = reference_poly.evaluate(round_challenges[round]);
            }
            reference
                .finish_rounds(round_challenges[rounds - 1])
                .unwrap();
            optimized
                .finish_rounds(round_challenges[rounds - 1])
                .unwrap();
            // Message-only round 0 plus fused rounds 1..log_t-1 detach; the
            // final cycle bind (host transition) and the address tail stay
            // synchronous, adding one more CB.
            assert_eq!(launched_rounds, log_t as u64, "overlap did not engage");
            assert_eq!(
                device_probe_count() - before,
                log_t as u64 + 1,
                "overlapped command-buffer schedule drifted"
            );

            let reference_outputs = reference.output_claims(&claims).unwrap();
            let optimized_outputs = optimized.output_claims(&claims).unwrap();
            assert_eq!(
                reference_outputs, optimized_outputs,
                "output claims mismatch"
            );
        });
    }
}
