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
use crate::metal::runtime::{KernelId, MetalContext};
use crate::metal::{metal_gate, testing, MetalError};
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

#[inline]
fn raw_cycle_entries(registers: &RegisterLanes, t: usize) -> ([RawRwEntryIdx; 3], usize) {
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
fn raw_cycle_entry_count(registers: &RegisterLanes, t: usize) -> usize {
    let rs1 = registers.rs1_index[t];
    let rs2 = registers.rs2_index[t];
    let rd = registers.rd_index[t];
    usize::from(rs1 != NO_REGISTER)
        + usize::from(rs2 != NO_REGISTER && rs2 != rs1)
        + usize::from(rd != NO_REGISTER && rd != rs1 && rd != rs2)
}

fn build_metal_register_tables_serial(registers: &RegisterLanes) -> MetalRegisterTables {
    let cycles = registers.rd_index.len();
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

#[cfg(feature = "parallel")]
fn build_metal_register_tables_parallel(
    registers: &RegisterLanes,
    chunk_size: usize,
) -> MetalRegisterTables {
    let cycles = registers.rd_index.len();
    let num_chunks = cycles.div_ceil(chunk_size);
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
    let entries_len = offsets[cycles] as usize;

    let mut tables = MetalRegisterTables {
        entries: Vec::with_capacity(cycles * 3),
        offsets,
        inc: Vec::with_capacity(cycles),
        rs1_indices: Vec::with_capacity(cycles),
        rs2_indices: Vec::with_capacity(cycles),
        rd_indices: Vec::with_capacity(cycles),
    };
    let mut entry_chunks = Vec::with_capacity(num_chunks);
    let mut entries_rest = tables.entries.spare_capacity_mut();
    for chunk in 0..num_chunks {
        let start = chunk * chunk_size;
        let end = (start + chunk_size).min(cycles);
        let len = (tables.offsets[end] - tables.offsets[start]) as usize;
        let (entries, rest) = entries_rest.split_at_mut(len);
        entry_chunks.push(entries);
        entries_rest = rest;
    }

    entry_chunks
        .into_par_iter()
        .zip(tables.inc.spare_capacity_mut().par_chunks_mut(chunk_size))
        .zip(
            tables
                .rs1_indices
                .spare_capacity_mut()
                .par_chunks_mut(chunk_size),
        )
        .zip(
            tables
                .rs2_indices
                .spare_capacity_mut()
                .par_chunks_mut(chunk_size),
        )
        .zip(
            tables
                .rd_indices
                .spare_capacity_mut()
                .par_chunks_mut(chunk_size),
        )
        .enumerate()
        .for_each(
            |(chunk_index, ((((entries, inc), rs1_indices), rs2_indices), rd_indices))| {
                let start = chunk_index * chunk_size;
                let mut entry_index = 0;
                for local_t in 0..inc.len() {
                    let t = start + local_t;
                    let (row, len) = raw_cycle_entries(registers, t);
                    debug_assert_eq!(len, raw_cycle_entry_count(registers, t));
                    for entry in &row[..len] {
                        let _ = entries[entry_index].write(*entry);
                        entry_index += 1;
                    }
                    let _ = inc[local_t].write(Fr::from_i128(
                        i128::from(registers.rd_post_value[t])
                            - i128::from(registers.rd_pre_value[t]),
                    ));
                    let rs1 = registers.rs1_index[t];
                    let rs2 = registers.rs2_index[t];
                    let rd = registers.rd_index[t];
                    let _ = rs1_indices[local_t].write((rs1 != NO_REGISTER).then_some(rs1));
                    let _ = rs2_indices[local_t].write((rs2 != NO_REGISTER).then_some(rs2));
                    let _ = rd_indices[local_t].write((rd != NO_REGISTER).then_some(rd));
                }
                debug_assert_eq!(entry_index, entries.len());
            },
        );

    // SAFETY: every spare-capacity slot below the new lengths is partitioned
    // into one parallel chunk and initialized exactly once above.
    unsafe {
        tables.entries.set_len(entries_len);
        tables.inc.set_len(cycles);
        tables.rs1_indices.set_len(cycles);
        tables.rs2_indices.set_len(cycles);
        tables.rd_indices.set_len(cycles);
    }
    tables
}

fn build_metal_register_tables(registers: &RegisterLanes) -> MetalRegisterTables {
    #[cfg(feature = "parallel")]
    {
        if std::env::var_os("JOLT_REGISTERS_PREPARE_SERIAL").is_some() {
            build_metal_register_tables_serial(registers)
        } else {
            build_metal_register_tables_parallel(
                registers,
                register_build_chunk_size(registers.rd_index.len()),
            )
        }
    }
    #[cfg(not(feature = "parallel"))]
    {
        build_metal_register_tables_serial(registers)
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

    fn bind(&mut self, challenge: Fr, final_bind: bool) -> Result<(), MetalError> {
        let pairs = self.rows / 2;
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
        {
            let offset_buffer = self.row_offsets.device_buffer();
            let out_offset_buffer = out_offsets.device_buffer();
            let mut pass = self.context.begin_pass()?;
            match (&self.entries, &new_entries) {
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
            pass.run()?;
        }
        testing::note_device_round();
        if deref {
            self.ra_table.clear();
            self.wa_table.clear();
        } else if matches!(new_entries, DeviceEntries::Indexed(_)) {
            self.ra_table = expand_lookup_table(&self.ra_table, challenge);
            self.wa_table = expand_lookup_table(&self.wa_table, challenge);
        }
        self.entries = new_entries;
        self.entry_count = new_count;
        self.row_offsets = out_offsets;
        self.inc.swap();
        self.rows = pairs;
        self.counts_valid = false;
        Ok(())
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

        let tables = build_metal_register_tables(&registers);
        drop(registers);
        if tables.entries.is_empty() {
            return self.fallback.prepare(session, witness, inputs);
        }
        session.park(SharedRdIndices(tables.rd_indices));
        let device = match DeviceRegistersRwState::new(
            context,
            tables.entries,
            tables.offsets,
            tables.inc,
            inputs.challenges.gamma,
        ) {
            Ok(device) => device,
            Err(error) => {
                tracing::warn!(slot = KIND, %error, "device preparation failed; using optimized fallback");
                return self.fallback.prepare(session, witness, inputs);
            }
        };
        Ok(Box::new(MetalRegistersRwKernel {
            log_t,
            log_k,
            device: Some(device),
            gruen: Some(GruenSplitEqPolynomial::new(
                r_cycle,
                BindingOrder::LowToHigh,
            )),
            host: None,
            rs1_indices: Some(tables.rs1_indices),
            rs2_indices: Some(tables.rs2_indices),
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

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
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
    use crate::metal::testing::{device_probe_count, gpu_lock};
    use crate::optimized::registers_read_write::test_support::{
        assert_kernel_parity, assert_nontrivial, challenge_sequence, structured_fixture,
    };

    #[cfg(feature = "parallel")]
    #[test]
    fn registers_rw_parallel_prepare_matches_serial() {
        structured_fixture(257).with_plane(9, |backend| {
            let mut session = ProofSession::default();
            let record = TraceRecord::shared::<Fr>(&mut session, backend, 9).unwrap();
            let serial = build_metal_register_tables_serial(&record.registers);
            let parallel = build_metal_register_tables_parallel(&record.registers, 17);

            assert_eq!(parallel.entries, serial.entries);
            assert_eq!(parallel.offsets, serial.offsets);
            assert_eq!(parallel.inc, serial.inc);
            assert_eq!(parallel.rs1_indices, serial.rs1_indices);
            assert_eq!(parallel.rs2_indices, serial.rs2_indices);
            assert_eq!(parallel.rd_indices, serial.rd_indices);
        });
    }

    fn run_parity(log_t: usize, seed: u64) {
        let _lock = gpu_lock();
        std::env::remove_var("JOLT_METAL_DISABLE");
        std::env::set_var("JOLT_METAL_MIN_TERMS_REGISTERS_READ_WRITE", "0");
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
            assert!(device_probe_count() > before, "device path did not engage");
        });
    }

    #[test]
    fn registers_rw_matches_reference_index_handoff() {
        run_parity(4, 23);
    }

    #[test]
    fn registers_rw_matches_reference_field_rounds() {
        run_parity(6, 47);
    }
}
