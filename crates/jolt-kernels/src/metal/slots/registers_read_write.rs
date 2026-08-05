//! Metal registers read/write-checking sparse cycle rounds.

use std::sync::Arc;

use jolt_claims::protocols::jolt::geometry::registers::rd_inc_read_write;
use jolt_field::{Fr, FromPrimitiveInt};
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
use crate::metal::runtime::{ComputePass, KernelId, MetalContext};
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

#[cfg(any(test, feature = "bench-utils", not(feature = "parallel")))]
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

enum DeviceEntries {
    Indexed(OwnedDeviceBuffer<RawRwEntryIdx>),
    Direct(OwnedDeviceBuffer<RawRwEntryF>),
}

struct DeviceBindPlan {
    out_offsets: OwnedDeviceBuffer<u32>,
    new_entries: DeviceEntries,
    new_count: usize,
    next_ra_table: Vec<Fr>,
    next_wa_table: Vec<Fr>,
}

struct DeviceRegistersRwState {
    context: &'static MetalContext,
    entries: DeviceEntries,
    entry_count: usize,
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
        Ok(Self {
            context,
            entry_count: entries.len(),
            entries: DeviceEntries::Indexed(context.own_vec(entries)?),
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
        let entries = context.own_mmap(MmapVec::zeroed(entry_count))?;
        let rs1_index = context.wrap_slice(registers.rs1_index)?;
        let rs2_index = context.wrap_slice(registers.rs2_index)?;
        let rd_index = context.wrap_slice(registers.rd_index)?;
        let rs1_value = context.wrap_slice(registers.rs1_value)?;
        let rs2_value = context.wrap_slice(registers.rs2_value)?;
        let rd_pre_value = context.wrap_slice(registers.rd_pre_value)?;
        let rd_post_value = context.wrap_slice(registers.rd_post_value)?;
        {
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
        Ok(Self {
            context,
            entries: DeviceEntries::Indexed(entries),
            entry_count,
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

    fn message(&mut self, gruen: &GruenSplitEqPolynomial<Fr>) -> Result<[Fr; 2], MetalError> {
        let pairs = self.rows / 2;
        let num_tgs = num_threadgroups(pairs);
        let e_in = gruen.e_in_current();
        let e_out = gruen.e_out_current();
        let e_in_buffer = self.context.wrap_slice(fr_as_u32s(e_in))?;
        let e_out_buffer = self.context.wrap_slice(fr_as_u32s(e_out))?;
        let offset_buffer = self.row_offsets.device_buffer();
        let inc_buffer = self.inc.cur().device_buffer();
        let partial_buffer = self.partials.buffer().device_buffer();
        let count_buffer = self.counts.device_buffer();
        let params = [
            pairs as u32,
            num_tgs as u32,
            e_in.len().trailing_zeros(),
            e_in.len() as u32,
        ];
        let mut pass = self.context.begin_pass()?;
        match &self.entries {
            DeviceEntries::Indexed(entries) => {
                let entry_buffer = entries.device_buffer();
                let ra_buffer = self.context.wrap_slice(fr_as_u32s(&self.ra_table))?;
                let wa_buffer = self.context.wrap_slice(fr_as_u32s(&self.wa_table))?;
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
                    pairs,
                );
            }
            DeviceEntries::Direct(entries) => {
                let entry_buffer = entries.device_buffer();
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
                    pairs,
                );
            }
        }
        pass.run()?;
        testing::note_device_round();
        self.counts_valid = true;
        let sums = self.partials.sums(num_tgs);
        Ok([sums[0], sums[1]])
    }

    fn scanned_offsets(&self) -> Result<(Vec<u32>, usize), MetalError> {
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

    fn plan_bind(&self, challenge: Fr, final_bind: bool) -> Result<DeviceBindPlan, MetalError> {
        let (offsets, new_count) = self.scanned_offsets()?;
        let out_offsets = self.context.own_vec(offsets)?;
        let deref = matches!(self.entries, DeviceEntries::Indexed(_))
            && (self.ra_table.len() >= RA_TABLE_DEREF_LEN || final_bind);
        let new_entries =
            match (&self.entries, deref) {
                (DeviceEntries::Indexed(_), false) => {
                    DeviceEntries::Indexed(self.context.own_page_aligned(
                        PageAlignedVec::from_elem(RawRwEntryIdx::default(), new_count),
                    )?)
                }
                _ => DeviceEntries::Direct(self.context.own_page_aligned(
                    PageAlignedVec::from_elem(RawRwEntryF::default(), new_count),
                )?),
            };
        let (next_ra_table, next_wa_table) = if deref {
            (Vec::new(), Vec::new())
        } else if matches!(new_entries, DeviceEntries::Indexed(_)) {
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
        match (&self.entries, &plan.new_entries) {
            (DeviceEntries::Indexed(entries), DeviceEntries::Indexed(out)) => {
                let entry_buffer = entries.device_buffer();
                let out_buffer = out.device_buffer();
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
            (DeviceEntries::Indexed(entries), DeviceEntries::Direct(out)) => {
                let entry_buffer = entries.device_buffer();
                let out_buffer = out.device_buffer();
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
            (DeviceEntries::Direct(entries), DeviceEntries::Direct(out)) => {
                let entry_buffer = entries.device_buffer();
                let out_buffer = out.device_buffer();
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
            (DeviceEntries::Direct(_), DeviceEntries::Indexed(_)) => {
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
        self.entries = plan.new_entries;
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
            pass.run()?;
        }
        testing::note_device_round();
        self.install_bind(plan);
        self.counts_valid = false;
        Ok(())
    }

    fn bind_and_message(
        &mut self,
        challenge: Fr,
        gruen: &GruenSplitEqPolynomial<Fr>,
    ) -> Result<[Fr; 2], MetalError> {
        let plan = self.plan_bind(challenge, false)?;
        let rows = self.rows / 2;
        let pairs = rows / 2;
        let num_tgs = num_threadgroups(pairs);
        let e_in = gruen.e_in_current();
        let e_out = gruen.e_out_current();
        let e_in_buffer = self.context.wrap_slice(fr_as_u32s(e_in))?;
        let e_out_buffer = self.context.wrap_slice(fr_as_u32s(e_out))?;
        let partial_buffer = self.partials.buffer().device_buffer();
        let count_buffer = self.counts.device_buffer();
        let params = [
            pairs as u32,
            num_tgs as u32,
            e_in.len().trailing_zeros(),
            e_in.len() as u32,
        ];
        {
            let mut pass = self.context.begin_pass()?;
            self.encode_bind(&mut pass, &plan, challenge)?;
            pass.buffer_barrier();
            let offset_buffer = plan.out_offsets.device_buffer();
            let inc_buffer = self.inc.nxt().device_buffer();
            match &plan.new_entries {
                DeviceEntries::Indexed(entries) => {
                    let entry_buffer = entries.device_buffer();
                    let ra_buffer = self.context.wrap_slice(fr_as_u32s(&plan.next_ra_table))?;
                    let wa_buffer = self.context.wrap_slice(fr_as_u32s(&plan.next_wa_table))?;
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
                        pairs,
                    );
                }
                DeviceEntries::Direct(entries) => {
                    let entry_buffer = entries.device_buffer();
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
                        pairs,
                    );
                }
            }
            pass.run()?;
        }
        testing::note_device_round();
        self.install_bind(plan);
        self.counts_valid = true;
        let sums = self.partials.sums(num_tgs);
        Ok([sums[0], sums[1]])
    }

    fn into_cycle_state(self) -> Result<(Vec<BoundRegistersRwEntry<Fr>>, Fr), ()> {
        if self.rows != 1 || self.inc.cur_slice(1).is_empty() {
            return Err(());
        }
        let DeviceEntries::Direct(entries) = self.entries else {
            return Err(());
        };
        let entries = entries.as_slice().get(..self.entry_count).ok_or(())?;
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
        match self.entries {
            DeviceEntries::Indexed(entries) => {
                let entries = entries.as_slice().get(..self.entry_count).ok_or(())?;
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
            DeviceEntries::Direct(entries) => {
                let entries = entries.as_slice().get(..self.entry_count).ok_or(())?;
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
        let ram = Arc::clone(&record.ram);
        drop(record);
        TraceRecord::release(session);
        crate::optimized::opening::park_opening_increments(session, &registers, &ram);

        let register_inputs = RegisterBuildInputs::from(registers.as_ref());
        let prepared = if gpu_prepare_enabled() {
            let metadata = build_metal_register_metadata(register_inputs);
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

struct MetalRegistersRwKernel {
    log_t: usize,
    log_k: usize,
    fused_rounds: bool,
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

#[cfg(feature = "bench-utils")]
pub mod bench {
    #![expect(
        clippy::expect_used,
        clippy::panic,
        reason = "benchmark fixtures must fail loudly"
    )]

    use std::time::{Duration, Instant};

    use jolt_sumcheck::ProveRounds;

    use super::*;

    const PREFIX_ROUNDS: usize = 7;

    #[derive(Clone, Copy)]
    pub struct BenchConfig {
        pub log_t: usize,
        pub seed: u64,
    }

    impl BenchConfig {
        pub fn production(log_t: usize) -> Self {
            assert!(log_t >= PREFIX_ROUNDS);
            Self {
                log_t,
                seed: 0x5743_5352_5750_5246 ^ log_t as u64,
            }
        }
    }

    pub struct BenchFixture {
        config: BenchConfig,
        entries: Vec<RawRwEntryIdx>,
        row_offsets: Vec<u32>,
        inc: Vec<Fr>,
        r_cycle: Vec<Fr>,
        gamma: Fr,
    }

    struct SyntheticRegisterLanes {
        rs1_value: MmapVec<u64>,
        rs2_value: MmapVec<u64>,
        rd_pre_value: MmapVec<u64>,
        rd_post_value: MmapVec<u64>,
        rs1_index: MmapVec<u8>,
        rs2_index: MmapVec<u8>,
        rd_index: MmapVec<u8>,
    }

    impl SyntheticRegisterLanes {
        fn inputs(&self) -> RegisterBuildInputs<'_> {
            RegisterBuildInputs {
                rs1_value: self.rs1_value.as_slice(),
                rs2_value: self.rs2_value.as_slice(),
                rd_pre_value: self.rd_pre_value.as_slice(),
                rd_post_value: self.rd_post_value.as_slice(),
                rs1_index: self.rs1_index.as_slice(),
                rs2_index: self.rs2_index.as_slice(),
                rd_index: self.rd_index.as_slice(),
            }
        }
    }

    pub struct PrepareBenchFixture {
        registers: SyntheticRegisterLanes,
        gamma: Fr,
    }

    pub struct PrepareTiming {
        pub total: Duration,
        pub command_buffers: u64,
        pub kernel_dispatches: u64,
        pub entry_bytes: usize,
    }

    pub struct PassTiming {
        pub total: Duration,
        pub command_buffers: u64,
        pub kernel_dispatches: u64,
    }

    pub struct PreparedPass {
        kernel: MetalRegistersRwKernel,
        fused_rounds: bool,
    }

    impl PreparedPass {
        pub fn run(&mut self) -> PassTiming {
            let probes_before = testing::device_probe_count();
            let dispatches_before = testing::device_dispatch_count();
            let started = Instant::now();
            let _ = drive_prefix(&mut self.kernel);
            let total = started.elapsed();
            let command_buffers = testing::device_probe_count() - probes_before;
            let kernel_dispatches = testing::device_dispatch_count() - dispatches_before;
            let expected_command_buffers = if self.fused_rounds { 7 } else { 13 };
            assert_eq!(
                command_buffers, expected_command_buffers,
                "registers read/write prefix command-buffer schedule drifted"
            );
            assert_eq!(
                kernel_dispatches, 19,
                "registers read/write prefix dispatch schedule drifted"
            );
            PassTiming {
                total,
                command_buffers,
                kernel_dispatches,
            }
        }
    }

    fn splitmix(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn synthetic_register_lanes(config: BenchConfig) -> SyntheticRegisterLanes {
        let rows = 1usize << config.log_t;
        let mut lanes = SyntheticRegisterLanes {
            rs1_value: MmapVec::zeroed(rows),
            rs2_value: MmapVec::zeroed(rows),
            rd_pre_value: MmapVec::zeroed(rows),
            rd_post_value: MmapVec::zeroed(rows),
            rs1_index: MmapVec::filled(rows, NO_REGISTER),
            rs2_index: MmapVec::filled(rows, NO_REGISTER),
            rd_index: MmapVec::filled(rows, NO_REGISTER),
        };
        let mut state = config.seed;
        for row in 0..rows {
            let base = (splitmix(&mut state) & 127) as u8;
            let other = base.wrapping_add(43) & 127;
            let third = base.wrapping_add(89) & 127;
            let (rs1, rs2, rd) = match row & 7 {
                0 => (Some(base), None, Some(other)),
                1 => (Some(base), Some(base), Some(third)),
                2 => (Some(base), Some(other), Some(third)),
                3 => (None, None, None),
                4 => (Some(base), Some(other), None),
                5 => (Some(base), None, Some(base)),
                6 => (None, Some(other), Some(third)),
                _ => (Some(base), Some(other), Some(other)),
            };
            let rs1_value = splitmix(&mut state);
            let rs2_value = splitmix(&mut state);
            let rd_pre_value = splitmix(&mut state);
            let rd_post_value = splitmix(&mut state);
            lanes.rs1_value[row] = rs1_value;
            lanes.rs2_value[row] = rs2_value;
            lanes.rd_pre_value[row] = rd_pre_value;
            lanes.rd_post_value[row] = rd_post_value;
            lanes.rs1_index[row] = rs1.unwrap_or(NO_REGISTER);
            lanes.rs2_index[row] = rs2.unwrap_or(NO_REGISTER);
            lanes.rd_index[row] = rd.unwrap_or(NO_REGISTER);
        }
        lanes
    }

    fn challenge(round: usize) -> Fr {
        Fr::from_u64(
            0xC0FF_EE11_D00D_F00D ^ (round as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x2A,
        )
    }

    fn synthetic_tables(
        config: BenchConfig,
        include_host: bool,
    ) -> (
        Vec<RawRwEntryIdx>,
        Vec<RegistersRwCycleEntry<Fr>>,
        Vec<u32>,
        Vec<Fr>,
    ) {
        let rows = 1usize << config.log_t;
        let mut entries = Vec::with_capacity(rows * 2);
        let mut host_entries = include_host.then(|| Vec::with_capacity(rows * 2));
        let mut offsets = Vec::with_capacity(rows + 1);
        let mut inc = Vec::with_capacity(rows);
        let gamma = Fr::from_u64(0x5EED_1234_5678_9ABC);
        let ra_table = [Fr::from_u64(0), gamma, gamma * gamma, gamma + gamma * gamma];
        let wa_table = [Fr::from_u64(0), Fr::from_u64(1)];
        let mut state = config.seed;
        offsets.push(0);
        for row in 0..rows {
            let len = [1usize, 2, 2, 3][row & 3];
            let base = (splitmix(&mut state) & 127) as u8;
            let mut cols = [
                base,
                base.wrapping_add(43) & 127,
                base.wrapping_add(89) & 127,
            ];
            cols[..len].sort_unstable();
            for (slot, &col) in cols[..len].iter().enumerate() {
                let prev_val = splitmix(&mut state);
                let next_val = prev_val.wrapping_add((row as u64) | 1);
                let ra = 1 + ((row + slot) % 3) as u16;
                let wa = u16::from(slot + 1 == len && row & 1 == 0);
                let val = Fr::from_u64(prev_val);
                entries.push(RawRwEntryIdx {
                    val,
                    prev_val,
                    next_val,
                    ra,
                    wa,
                    col,
                    pad: [0; 3],
                });
                if let Some(host_entries) = &mut host_entries {
                    host_entries.push(RegistersRwCycleEntry {
                        row,
                        col,
                        prev_val,
                        next_val,
                        val,
                        ra: ra_table[ra as usize],
                        wa: wa_table[wa as usize],
                    });
                }
            }
            offsets.push(entries.len() as u32);
            inc.push(Fr::from_u64(splitmix(&mut state)));
        }
        (entries, host_entries.unwrap_or_default(), offsets, inc)
    }

    impl BenchFixture {
        pub fn synthetic(config: BenchConfig) -> Self {
            let (entries, _, row_offsets, inc) = synthetic_tables(config, false);
            let mut state = config.seed ^ 0x4551_4359_434C_4553;
            let r_cycle = (0..config.log_t)
                .map(|_| Fr::from_u64(splitmix(&mut state)))
                .collect();
            Self {
                config,
                entries,
                row_offsets,
                inc,
                r_cycle,
                gamma: Fr::from_u64(0x5EED_1234_5678_9ABC),
            }
        }

        pub fn cycles(&self) -> usize {
            1usize << self.config.log_t
        }

        fn device_kernel(&self, fused_rounds: bool) -> MetalRegistersRwKernel {
            let context = MetalContext::global().expect("Metal context");
            let device = DeviceRegistersRwState::new(
                context,
                self.entries.clone(),
                self.row_offsets.clone(),
                self.inc.clone(),
                self.gamma,
            )
            .expect("registers read/write device fixture");
            MetalRegistersRwKernel {
                log_t: self.config.log_t,
                log_k: 7,
                fused_rounds,
                device: Some(device),
                gruen: Some(GruenSplitEqPolynomial::new(
                    &self.r_cycle,
                    BindingOrder::LowToHigh,
                )),
                host: None,
                rs1_indices: Some(Vec::new()),
                rs2_indices: Some(Vec::new()),
                bound_challenges: Some(Vec::with_capacity(PREFIX_ROUNDS)),
                rounds_bound: 0,
            }
        }

        pub fn prepare_device_pass(&self, fused_rounds: bool) -> PreparedPass {
            PreparedPass {
                kernel: self.device_kernel(fused_rounds),
                fused_rounds,
            }
        }

        pub fn run_device_pass(&self, fused_rounds: bool) -> PassTiming {
            self.prepare_device_pass(fused_rounds).run()
        }
    }

    impl PrepareBenchFixture {
        pub fn synthetic(config: BenchConfig) -> Self {
            Self {
                registers: synthetic_register_lanes(config),
                gamma: Fr::from_u64(0x5EED_1234_5678_9ABC),
            }
        }

        pub fn cycles(&self) -> usize {
            self.registers.rd_index.len()
        }

        pub fn run(&self, gpu: bool) -> PrepareTiming {
            let context = MetalContext::global().expect("Metal context");
            let inputs = self.registers.inputs();
            let dispatches_before = testing::device_dispatch_count();
            let started = Instant::now();
            let device = if gpu {
                let metadata = build_metal_register_metadata(inputs);
                DeviceRegistersRwState::new_from_registers(
                    context,
                    inputs,
                    metadata.offsets,
                    metadata.inc,
                    self.gamma,
                )
                .expect("GPU registers CSR prepare")
            } else {
                let tables = build_metal_register_tables_serial(inputs);
                DeviceRegistersRwState::new(
                    context,
                    tables.entries,
                    tables.offsets,
                    tables.inc,
                    self.gamma,
                )
                .expect("serial registers CSR prepare")
            };
            let total = started.elapsed();
            let kernel_dispatches = testing::device_dispatch_count() - dispatches_before;
            assert_eq!(kernel_dispatches, u64::from(gpu));
            let entry_bytes = device.entry_count * std::mem::size_of::<RawRwEntryIdx>();
            let _ = std::hint::black_box(&device);
            PrepareTiming {
                total,
                command_buffers: u64::from(gpu),
                kernel_dispatches,
                entry_bytes,
            }
        }
    }

    fn drive_prefix(kernel: &mut dyn ProveRounds<Fr>) -> Vec<Vec<Fr>> {
        let mut claim = Fr::from_u64(0xBEEF);
        let mut messages = Vec::with_capacity(PREFIX_ROUNDS);
        for round in 0..PREFIX_ROUNDS {
            let bind = round.checked_sub(1).map(challenge);
            let poly = kernel
                .prove_round(bind, round, claim)
                .expect("registers read/write prefix round");
            claim = poly.evaluate(challenge(round));
            messages.push(poly.coefficients().to_vec());
        }
        messages
    }

    pub fn assert_small_scale_parity() {
        let config = BenchConfig::production(12);
        let fixture = BenchFixture::synthetic(config);
        let (_, host_entries, _, _) = synthetic_tables(config, true);
        let mut host = ReadWriteKernel::from_partial_cycle_state(
            config.log_t,
            7,
            host_entries,
            GruenSplitEqPolynomial::new(&fixture.r_cycle, BindingOrder::LowToHigh),
            Polynomial::new(fixture.inc.clone()),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            0,
        );
        let expected = drive_prefix(&mut host);
        for fused_rounds in [false, true] {
            let mut device = fixture.device_kernel(fused_rounds);
            assert_eq!(
                drive_prefix(&mut device),
                expected,
                "registers read/write prefix wire parity failed (fused={fused_rounds})"
            );
        }
    }

    pub fn assert_small_scale_prepare_parity() {
        let fixture = PrepareBenchFixture::synthetic(BenchConfig::production(12));
        let inputs = fixture.registers.inputs();
        let serial = build_metal_register_tables_serial(inputs);
        let metadata = build_metal_register_metadata_serial(inputs);
        let context = MetalContext::global().expect("Metal context");
        let device = DeviceRegistersRwState::new_from_registers(
            context,
            inputs,
            metadata.offsets.clone(),
            metadata.inc.clone(),
            fixture.gamma,
        )
        .expect("GPU registers CSR prepare oracle");
        let DeviceEntries::Indexed(entries) = &device.entries else {
            panic!("prepare must start in the indexed CSR representation");
        };
        assert_eq!(entries.as_slice(), serial.entries);
        assert_eq!(metadata.offsets, serial.offsets);
        assert_eq!(metadata.inc, serial.inc);
        assert_eq!(metadata.rs1_indices, serial.rs1_indices);
        assert_eq!(metadata.rs2_indices, serial.rs2_indices);
        assert_eq!(metadata.rd_indices, serial.rd_indices);
    }
}

#[cfg(test)]
#[expect(
    clippy::panic,
    clippy::unwrap_used,
    reason = "test module must fail loudly"
)]
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

            let DeviceEntries::Indexed(entries) = &device.entries else {
                panic!("prepare must start in the indexed CSR representation");
            };
            assert_eq!(entries.as_slice(), serial.entries);
            assert_eq!(metadata.offsets, serial.offsets);
            assert_eq!(metadata.inc, serial.inc);
            assert_eq!(metadata.rs1_indices, serial.rs1_indices);
            assert_eq!(metadata.rs2_indices, serial.rs2_indices);
            assert_eq!(metadata.rd_indices, serial.rd_indices);
        });
    }

    fn run_parity(log_t: usize, seed: u64, fused: bool, expected_device_rounds: u64) {
        let _lock = gpu_lock();
        std::env::remove_var("JOLT_METAL_DISABLE");
        std::env::set_var("JOLT_METAL_MIN_TERMS_REGISTERS_READ_WRITE", "0");
        std::env::set_var("JOLT_REGRW_FUSED", if fused { "1" } else { "0" });
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
        run_parity(4, 23, true, 5);
    }

    #[test]
    fn registers_rw_matches_reference_field_rounds() {
        run_parity(6, 47, true, 7);
    }

    #[test]
    fn registers_rw_legacy_schedule_matches_reference() {
        run_parity(6, 71, false, 12);
    }
}
