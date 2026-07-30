//! Metal lazy-RA drivers (stage 6b): booleanity-cycle and instruction RA
//! virtualization behind the [`LazyRaDevice`] seam.
//!
//! Both consumers keep their optimized kernels as THE kernels — binds, the
//! Gruen factor, claim assembly and every CPU path stay host-side and
//! byte-identical; the driver substitutes the round-message mass:
//!
//! - **Lazy rounds** (widths 1/2/4): one dispatch gathers the branch-table
//!   pairs straight off the shared packed rows (zero-copy — the same buffer
//!   the stage-5 slot scans) and reduces the summand lanes; a decline or
//!   failure costs nothing (reads only) and the CPU recomputes the round.
//! - **The third bind**: `jk_ra_materialize` gathers every polynomial dense
//!   at `cycles/8` into ONE flat device-owned ping-pong pair — the gather
//!   happens once, device-resident, instead of a host materialization the
//!   dense rounds then re-walk.
//! - **Dense rounds**: the st5 fused fold+eval shape (compact strides
//!   `len → len/2`, one command buffer, one wait) with each consumer's
//!   summand. Below the gate or on failure the driver steps aside pre-round
//!   and [`LazyFoldedRa::ensure_host`] reclaims the live tables (small at
//!   the shrunken length) plus any pending fold.
//!
//! Parity is by construction: eq weights multiply each row's lane sums
//! where the CPU folds `e_in` per block and `e_out` per block-end
//! (distributivity, exact), gathers add the same hot branches in the same
//! order, and the host assembles round polynomials through the consumers'
//! own recipes.

use std::sync::Arc;

use jolt_field::{Fr, FromPrimitiveInt};
use jolt_verifier::stages::stage6b::booleanity::Booleanity;
use jolt_verifier::stages::stage6b::instruction_ra_virtualization::InstructionRaVirtualization;
use jolt_verifier::stages::stage6b::ram_ra_virtualization::RamRaVirtualization;
use jolt_witness::JoltWitnessPlane;

use super::{num_threadgroups, own_uninit_frs, DeviceRound, Partials};
use crate::metal::buffers::{DeviceBuffer, OwnedDeviceBuffer};
use crate::metal::field::{fr_as_u32s, fr_to_u32_limbs};
use crate::metal::runtime::{KernelId, MetalContext};
use crate::metal::{metal_gate, testing, MetalError};
use crate::optimized::booleanity::{prepare_booleanity_cycle, BooleanityDeviceInputs};
use crate::optimized::instruction_ra_virtualization::OptimizedInstructionRaVirtualizationKernel;
use crate::optimized::instruction_read_raf::{shared_instruction_rows, InstructionCycleRow};
use crate::optimized::lazy_ra::LazyRaDevice;
use crate::optimized::ram_ra_virtualization::{
    prepare_ram_ra_virtualization, RamRaVirtualizationDeviceInputs,
};
use crate::{KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel};

const BOOL_KIND: &str = "booleanity_cycle";
const RAV_KIND: &str = "instruction_ra_virtualization";
const RAM_RAV_KIND: &str = "ram_ra_virtualization";

/// The packed-row gather family of the +1-sentinel remapped RAM address
/// (`jk_ra_hot_index` kind 2 in `ra_lazy.metal`).
const RAM_ADDRESS_KIND: u32 = 2;

/// Per-virtual batch capacity baked into the shaders (`JK_RAV_MAX_BATCH`).
const MAX_BATCH: usize = 8;
/// Branch-table width cap: tables ride to the device per round, so keep the
/// upload bounded (production chunk bits are 4 or 8).
const MAX_CHUNK_BITS: usize = 16;

/// Slot front: the optimized booleanity-cycle prepare with the device
/// driver factory installed — the optimized kernel IS the fallback (a
/// `None` driver), so every bind, message recipe and claim stays shared.
pub struct MetalBooleanityCycle;

impl PrepareKernel<Fr, Booleanity<Fr>> for MetalBooleanityCycle {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<Fr>,
        inputs: ProverInputs<'_, Fr, Booleanity<Fr>>,
    ) -> Result<Box<dyn SumcheckKernel<Fr, Relation = Booleanity<Fr>>>, KernelError<Fr>> {
        prepare_booleanity_cycle(session, witness, inputs, |device| {
            build_bool_driver(&device).map(|driver| Box::new(driver) as Box<dyn LazyRaDevice<Fr>>)
        })
    }
}

fn build_bool_driver(inputs: &BooleanityDeviceInputs<'_, Fr>) -> Option<BoolDriver> {
    if !metal_gate(BOOL_KIND, inputs.rows.len()) || inputs.log_k_chunk > MAX_CHUNK_BITS {
        return None;
    }
    let context = match MetalContext::global() {
        Ok(context) => context,
        Err(error) => {
            tracing::warn!(slot = BOOL_KIND, %error, "no device context; staying on the CPU");
            return None;
        }
    };
    let build = || -> Result<BoolDriver, MetalError> {
        let rho = context.own_vec(inputs.gamma_powers.to_vec())?;
        testing::note_copied_buffers(u64::from(rho.was_copied()));
        Ok(BoolDriver {
            core: DeviceLazyRa::build(
                context,
                BOOL_KIND,
                Arc::clone(inputs.rows),
                &inputs.poly_meta,
                inputs.log_k_chunk,
                2,
            )?,
            rho,
        })
    };
    match build() {
        Ok(driver) => Some(driver),
        Err(error) => {
            tracing::warn!(slot = BOOL_KIND, %error, "driver build failed; staying on the CPU");
            None
        }
    }
}

/// Slot front for instruction RA virtualization: the optimized kernel with
/// the device driver threaded through `new_with_driver`.
pub struct MetalInstructionRaVirtualization;

impl PrepareKernel<Fr, InstructionRaVirtualization<Fr>> for MetalInstructionRaVirtualization {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<Fr>,
        inputs: ProverInputs<'_, Fr, InstructionRaVirtualization<Fr>>,
    ) -> Result<
        Box<dyn SumcheckKernel<Fr, Relation = InstructionRaVirtualization<Fr>>>,
        KernelError<Fr>,
    > {
        let relation = inputs.relation;
        let dimensions = relation.dimensions();
        let cycles = 1usize << dimensions.log_t();
        let rows = shared_instruction_rows(session, witness, cycles)?;
        let driver = build_rav_driver(
            &rows,
            dimensions.num_virtual_ra_polys() * dimensions.num_committed_per_virtual(),
            dimensions.num_committed_per_virtual(),
            relation.committed_chunk_bits(),
        )
        .map(|driver| Box::new(driver) as Box<dyn LazyRaDevice<Fr>>);
        Ok(Box::new(
            OptimizedInstructionRaVirtualizationKernel::new_with_driver(
                dimensions.log_t(),
                dimensions.num_virtual_ra_polys(),
                dimensions.num_committed_per_virtual(),
                relation.instruction_address(),
                relation.instruction_read_raf_cycle(),
                relation.committed_chunk_bits(),
                rows,
                inputs.challenges.gamma,
                driver,
            )?,
        ))
    }
}

/// Slot front for RAM RA virtualization: the optimized kernel with a device
/// driver injected through the prepare seam. The summand is a SINGLE
/// product group over all committed chunks, so the instruction-RAV kernels
/// serve it at `num_polys == batch == num_committed` — same lane grid, RAM
/// address gathers via the packed rows' sentinel family.
pub struct MetalRamRaVirtualization;

impl PrepareKernel<Fr, RamRaVirtualization<Fr>> for MetalRamRaVirtualization {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<Fr>,
        inputs: ProverInputs<'_, Fr, RamRaVirtualization<Fr>>,
    ) -> Result<Box<dyn SumcheckKernel<Fr, Relation = RamRaVirtualization<Fr>>>, KernelError<Fr>>
    {
        prepare_ram_ra_virtualization(session, witness, inputs, |device| {
            build_ram_rav_driver(device).map(|driver| Box::new(driver) as Box<dyn LazyRaDevice<Fr>>)
        })
    }
}

fn build_ram_rav_driver(inputs: RamRaVirtualizationDeviceInputs<'_, Fr>) -> Option<RavDriver> {
    let batch = inputs.num_committed;
    let cycles = 1usize << inputs.log_t;
    if !metal_gate(RAM_RAV_KIND, cycles)
        || !(2..=MAX_BATCH).contains(&batch)
        || inputs.committed_chunk_bits == 0
        || inputs.committed_chunk_bits > MAX_CHUNK_BITS
    {
        return None;
    }
    let context = match MetalContext::global() {
        Ok(context) => context,
        Err(error) => {
            tracing::warn!(slot = RAM_RAV_KIND, %error, "no device context; staying on the CPU");
            return None;
        }
    };
    // The device gathers off the stage-6b packed rows (session-shared with
    // the booleanity and instruction-RAV slots); the CPU kernel keeps its
    // RamAccessColumns source — same per-cycle remapped addresses.
    let rows = match shared_instruction_rows(inputs.session, inputs.witness, cycles) {
        Ok(rows) => rows,
        Err(error) => {
            tracing::warn!(slot = RAM_RAV_KIND, %error, "no packed rows; staying on the CPU");
            return None;
        }
    };
    // Committed chunk `i` selects address bits at RamAddressChunks::index's
    // shift order.
    let meta: Vec<(u32, u32)> = (0..batch)
        .map(|i| {
            (
                RAM_ADDRESS_KIND,
                ((batch - 1 - i) * inputs.committed_chunk_bits) as u32,
            )
        })
        .collect();
    match DeviceLazyRa::build(
        context,
        RAM_RAV_KIND,
        rows,
        &meta,
        inputs.committed_chunk_bits,
        batch,
    ) {
        Ok(core) => Some(RavDriver { core, batch }),
        Err(error) => {
            tracing::warn!(slot = RAM_RAV_KIND, %error, "driver build failed; staying on the CPU");
            None
        }
    }
}

fn build_rav_driver(
    rows: &Arc<Vec<InstructionCycleRow>>,
    num_committed: usize,
    batch: usize,
    chunk_bits: usize,
) -> Option<RavDriver> {
    if !metal_gate(RAV_KIND, rows.len())
        || !(2..=MAX_BATCH).contains(&batch)
        || chunk_bits > MAX_CHUNK_BITS
        || chunk_bits == 0
    {
        return None;
    }
    let context = match MetalContext::global() {
        Ok(context) => context,
        Err(error) => {
            tracing::warn!(slot = RAV_KIND, %error, "no device context; staying on the CPU");
            return None;
        }
    };
    // Committed chunk `i` selects lookup-index bits at the read-raf shift
    // order (LookupIndexChunks::index).
    let meta: Vec<(u32, u32)> = (0..num_committed)
        .map(|i| (0, ((num_committed - 1 - i) * chunk_bits) as u32))
        .collect();
    match DeviceLazyRa::build(
        context,
        RAV_KIND,
        Arc::clone(rows),
        &meta,
        chunk_bits,
        batch,
    ) {
        Ok(core) => Some(RavDriver { core, batch }),
        Err(error) => {
            tracing::warn!(slot = RAV_KIND, %error, "driver build failed; staying on the CPU");
            None
        }
    }
}

/// Device state shared by the two drivers: the packed rows, the per-poly
/// selector metas, the lane partials and the dense ping-pong. The
/// summand-specific dispatch encodings live on the driver wrappers.
struct DeviceLazyRa {
    device: DeviceRound,
    rows: Arc<Vec<InstructionCycleRow>>,
    meta: OwnedDeviceBuffer<u32>,
    num_polys: usize,
    k_entries: usize,
    mask: u32,
    partials: Partials,
    dense: Option<DenseTables>,
}

/// Flat poly-major dense tables, compact at stride `len` in `cur` and
/// `len / 2` in `nxt` (the st5 ping-pong shape).
struct DenseTables {
    cur: OwnedDeviceBuffer<Fr>,
    nxt: OwnedDeviceBuffer<Fr>,
    len: usize,
}

/// One admitted round's dispatch geometry.
struct RoundGeometry {
    context: &'static MetalContext,
    groups: usize,
    num_tgs: usize,
    /// Dense rounds only: the pre-bind per-poly length (`cur`'s stride).
    len: usize,
}

impl DeviceLazyRa {
    fn build(
        context: &'static MetalContext,
        kind: &'static str,
        rows: Arc<Vec<InstructionCycleRow>>,
        meta_pairs: &[(u32, u32)],
        chunk_bits: usize,
        lanes: usize,
    ) -> Result<Self, MetalError> {
        let mut meta_words = Vec::with_capacity(meta_pairs.len() * 2);
        for (kind_word, shift) in meta_pairs {
            meta_words.push(*kind_word);
            meta_words.push(*shift);
        }
        let meta = context.own_vec(meta_words)?;
        testing::note_copied_buffers(u64::from(meta.was_copied()));
        Ok(Self {
            device: DeviceRound::new(context, kind),
            partials: Partials::new(context, lanes, rows.len() / 2)?,
            rows,
            meta,
            num_polys: meta_pairs.len(),
            k_entries: 1 << chunk_bits,
            mask: ((1u64 << chunk_bits) - 1) as u32,
            dense: None,
        })
    }

    /// Gate + tiling for a lazy round at branch width `width`; the geometry
    /// the dispatch encodes.
    fn lazy_geometry(&self, width: usize, e_in: &[Fr], e_out: &[Fr]) -> Option<RoundGeometry> {
        let domain = self.rows.len() / width;
        let context = self.device.gated(domain)?;
        let groups = domain / 2;
        if groups == 0 || !e_in.len().is_power_of_two() || e_in.len() * e_out.len() != groups {
            return None;
        }
        Some(RoundGeometry {
            context,
            groups,
            num_tgs: num_threadgroups(groups),
            len: 0,
        })
    }

    /// Gate + tiling for a dense round.
    fn dense_geometry(&self, binding: bool, e_in: &[Fr], e_out: &[Fr]) -> Option<RoundGeometry> {
        let len = self.dense.as_ref()?.len;
        let context = self.device.gated(len)?;
        let groups = if binding { len / 4 } else { len / 2 };
        if groups == 0 || !e_in.len().is_power_of_two() || e_in.len() * e_out.len() != groups {
            return None;
        }
        Some(RoundGeometry {
            context,
            groups,
            num_tgs: num_threadgroups(groups),
            len,
        })
    }

    /// The width-8 materialization into a fresh flat ping-pong pair.
    /// `Ok(false)` = ineligible buffers, nothing dispatched.
    fn dispatch_materialize(
        &mut self,
        context: &'static MetalContext,
        tables: &[Vec<Fr>],
        new_len: usize,
    ) -> Result<bool, MetalError> {
        let Some(rows_buffer) = context.wrap_slice_nocopy(self.rows.as_slice()) else {
            return Ok(false);
        };
        let flat = concat_tables(tables, 8 * self.k_entries);
        let tables_buffer = context.wrap_slice(fr_as_u32s(&flat))?;
        testing::note_copied_buffers(u64::from(tables_buffer.was_copied()));
        let Some(cur) = own_uninit_frs(context, self.num_polys * new_len)? else {
            return Ok(false);
        };
        let Some(nxt) = own_uninit_frs(context, self.num_polys * (new_len / 2))? else {
            return Ok(false);
        };
        let params = [
            new_len.trailing_zeros(),
            self.num_polys as u32,
            self.k_entries as u32,
            self.mask,
        ];
        let cur_buffer = cur.device_buffer();
        let meta = self.meta.device_buffer();
        let mut pass = context.begin_pass()?;
        pass.dispatch(
            KernelId::RaMaterialize,
            &params,
            &[&rows_buffer, &meta, &tables_buffer, &cur_buffer],
            self.num_polys * new_len,
        );
        pass.run()?;
        testing::note_device_round();
        drop(cur_buffer);
        self.dense = Some(DenseTables {
            cur,
            nxt,
            len: new_len,
        });
        Ok(true)
    }

    fn adopt(&mut self, tables: &[Vec<Fr>]) -> bool {
        let cycles = self.rows.len();
        let new_len = cycles / 8;
        let Some(context) = self.device.gated(new_len) else {
            return false;
        };
        match self.dispatch_materialize(context, tables, new_len) {
            Ok(adopted) => adopted,
            Err(error) => {
                // Materialization writes only the dropped buffers.
                self.device.failed(&error);
                false
            }
        }
    }

    /// After a successful fused bind: swap the ping-pong and halve.
    fn dense_advance(&mut self, bound: bool) {
        if !bound {
            return;
        }
        if let Some(dense) = self.dense.as_mut() {
            std::mem::swap(&mut dense.cur, &mut dense.nxt);
            dense.len /= 2;
        }
    }

    fn take_dense(&mut self) -> Vec<Vec<Fr>> {
        let Some(dense) = self.dense.take() else {
            return Vec::new();
        };
        let len = dense.len;
        let flat = dense.cur.as_slice();
        (0..self.num_polys)
            .map(|i| flat[i * len..(i + 1) * len].to_vec())
            .collect()
    }
}

/// Flatten per-poly branch tables for the device (uniform stride —
/// asserted, since the kernels index `poly * stride`).
fn concat_tables(tables: &[Vec<Fr>], stride: usize) -> Vec<Fr> {
    let mut flat = Vec::with_capacity(tables.len() * stride);
    for table in tables {
        debug_assert_eq!(table.len(), stride);
        flat.extend_from_slice(table);
    }
    flat
}

/// Wrap the current gruen levels (tiny copies below the no-copy floor are
/// expected and counted).
fn wrap_eq<'a>(
    context: &'static MetalContext,
    e_in: &'a [Fr],
    e_out: &'a [Fr],
) -> Result<(DeviceBuffer<'a>, DeviceBuffer<'a>), MetalError> {
    let e_in_buffer = context.wrap_slice(fr_as_u32s(e_in))?;
    let e_out_buffer = context.wrap_slice(fr_as_u32s(e_out))?;
    testing::note_copied_buffers(
        u64::from(e_in_buffer.was_copied()) + u64::from(e_out_buffer.was_copied()),
    );
    Ok((e_in_buffer, e_out_buffer))
}

/// Booleanity-cycle driver: lanes `[q_constant, q_leading]`.
struct BoolDriver {
    core: DeviceLazyRa,
    /// γ^i per polynomial (the summand's `H(H − γ^i)`).
    rho: OwnedDeviceBuffer<Fr>,
}

// SAFETY: [`LazyRaDevice`] exposes no `&self` operations and the drivers no
// other API, so a shared reference crossing threads (the consumers' rayon
// round loops capture the enclosing `LazyFoldedRa`, driver field unused)
// permits no access at all; every `&mut` use stays on the single prove
// thread. The Metal handles inside are `MTLResource`s, which Metal
// documents as thread-safe (only command encoders are not), and every pass
// blocks to completion before any host access.
unsafe impl Send for BoolDriver {}
// SAFETY: as for `Send` — no shared-reference operations exist.
unsafe impl Sync for BoolDriver {}

impl BoolDriver {
    fn dispatch_lazy(
        &mut self,
        geometry: &RoundGeometry,
        tables: &[Vec<Fr>],
        width: usize,
        e_in: &[Fr],
        e_out: &[Fr],
    ) -> Result<Option<Vec<Fr>>, MetalError> {
        let context = geometry.context;
        let core = &self.core;
        // An ineligible wrap declines softly — nothing ran, the device
        // stays healthy for later phases.
        let Some(rows_buffer) = context.wrap_slice_nocopy(core.rows.as_slice()) else {
            return Ok(None);
        };
        let flat = concat_tables(tables, width * core.k_entries);
        let tables_buffer = context.wrap_slice(fr_as_u32s(&flat))?;
        testing::note_copied_buffers(u64::from(tables_buffer.was_copied()));
        let (e_in_buffer, e_out_buffer) = wrap_eq(context, e_in, e_out)?;
        let params = [
            geometry.groups as u32,
            geometry.num_tgs as u32,
            e_in.len().trailing_zeros(),
            width as u32,
            core.num_polys as u32,
            core.k_entries as u32,
            core.mask,
        ];
        let meta = core.meta.device_buffer();
        let rho = self.rho.device_buffer();
        let partials = core.partials.buffer().device_buffer();
        let mut pass = context.begin_pass()?;
        pass.dispatch(
            KernelId::BoolLazyRound,
            &params,
            &[
                &rows_buffer,
                &meta,
                &tables_buffer,
                &rho,
                &e_in_buffer,
                &e_out_buffer,
                &partials,
            ],
            geometry.groups,
        );
        pass.run()?;
        testing::note_device_round();
        Ok(Some(core.partials.sums(geometry.num_tgs)))
    }

    fn dispatch_dense(
        &mut self,
        geometry: &RoundGeometry,
        bind: Option<Fr>,
        e_in: &[Fr],
        e_out: &[Fr],
    ) -> Result<Vec<Fr>, MetalError> {
        let context = geometry.context;
        let core = &self.core;
        let dense = core
            .dense
            .as_ref()
            .ok_or(MetalError::UnsupportedShape("dense round before adoption"))?;
        let (e_in_buffer, e_out_buffer) = wrap_eq(context, e_in, e_out)?;
        let mut params = vec![
            geometry.groups as u32,
            u32::from(bind.is_some()),
            geometry.num_tgs as u32,
            e_in.len().trailing_zeros(),
            core.num_polys as u32,
            geometry.len as u32,
        ];
        params.extend_from_slice(&fr_to_u32_limbs(bind.unwrap_or_else(|| Fr::from_u64(0))));
        let cur = dense.cur.device_buffer();
        let nxt = dense.nxt.device_buffer();
        let rho = self.rho.device_buffer();
        let partials = core.partials.buffer().device_buffer();
        let mut pass = context.begin_pass()?;
        pass.dispatch(
            KernelId::BoolDenseRound,
            &params,
            &[&cur, &nxt, &rho, &e_in_buffer, &e_out_buffer, &partials],
            geometry.groups,
        );
        pass.run()?;
        testing::note_device_round();
        Ok(core.partials.sums(geometry.num_tgs))
    }
}

impl LazyRaDevice<Fr> for BoolDriver {
    fn lazy_lanes(
        &mut self,
        tables: &[Vec<Fr>],
        width: usize,
        e_in: &[Fr],
        e_out: &[Fr],
    ) -> Option<Vec<Fr>> {
        let geometry = self.core.lazy_geometry(width, e_in, e_out)?;
        match self.dispatch_lazy(&geometry, tables, width, e_in, e_out) {
            Ok(lanes) => lanes,
            Err(error) => {
                // Lazy rounds only read — the CPU recomputes this round.
                self.core.device.failed(&error);
                None
            }
        }
    }

    fn adopt_dense(&mut self, tables: &[Vec<Fr>]) -> bool {
        self.core.adopt(tables)
    }

    fn dense_round(&mut self, bind: Option<Fr>, e_in: &[Fr], e_out: &[Fr]) -> Option<Vec<Fr>> {
        let geometry = self.core.dense_geometry(bind.is_some(), e_in, e_out)?;
        match self.dispatch_dense(&geometry, bind, e_in, e_out) {
            Ok(lanes) => {
                self.core.dense_advance(bind.is_some());
                Some(lanes)
            }
            Err(error) => {
                // The fused kernel writes only `nxt` and the partials —
                // `cur` (the pre-bind tables) is intact for the handoff.
                self.core.device.failed(&error);
                None
            }
        }
    }

    fn take_dense(&mut self) -> Vec<Vec<Fr>> {
        self.core.take_dense()
    }
}

/// RA-virtualization driver: lanes `[q(1), …, q(batch−1), q(∞)]`.
struct RavDriver {
    core: DeviceLazyRa,
    batch: usize,
}

// SAFETY: see [`BoolDriver`] — identical argument.
unsafe impl Send for RavDriver {}
// SAFETY: see [`BoolDriver`] — identical argument.
unsafe impl Sync for RavDriver {}

impl RavDriver {
    fn dispatch_lazy(
        &mut self,
        geometry: &RoundGeometry,
        tables: &[Vec<Fr>],
        width: usize,
        e_in: &[Fr],
        e_out: &[Fr],
    ) -> Result<Option<Vec<Fr>>, MetalError> {
        let context = geometry.context;
        let core = &self.core;
        // An ineligible wrap declines softly — nothing ran, the device
        // stays healthy for later phases.
        let Some(rows_buffer) = context.wrap_slice_nocopy(core.rows.as_slice()) else {
            return Ok(None);
        };
        let flat = concat_tables(tables, width * core.k_entries);
        let tables_buffer = context.wrap_slice(fr_as_u32s(&flat))?;
        testing::note_copied_buffers(u64::from(tables_buffer.was_copied()));
        let (e_in_buffer, e_out_buffer) = wrap_eq(context, e_in, e_out)?;
        let params = [
            geometry.groups as u32,
            geometry.num_tgs as u32,
            e_in.len().trailing_zeros(),
            width as u32,
            core.num_polys as u32,
            self.batch as u32,
            core.k_entries as u32,
            core.mask,
        ];
        let meta = core.meta.device_buffer();
        let partials = core.partials.buffer().device_buffer();
        let mut pass = context.begin_pass()?;
        pass.dispatch(
            KernelId::RavLazyRound,
            &params,
            &[
                &rows_buffer,
                &meta,
                &tables_buffer,
                &e_in_buffer,
                &e_out_buffer,
                &partials,
            ],
            geometry.groups,
        );
        pass.run()?;
        testing::note_device_round();
        Ok(Some(core.partials.sums(geometry.num_tgs)))
    }

    fn dispatch_dense(
        &mut self,
        geometry: &RoundGeometry,
        bind: Option<Fr>,
        e_in: &[Fr],
        e_out: &[Fr],
    ) -> Result<Vec<Fr>, MetalError> {
        let context = geometry.context;
        let core = &self.core;
        let dense = core
            .dense
            .as_ref()
            .ok_or(MetalError::UnsupportedShape("dense round before adoption"))?;
        let (e_in_buffer, e_out_buffer) = wrap_eq(context, e_in, e_out)?;
        let mut params = vec![
            geometry.groups as u32,
            u32::from(bind.is_some()),
            geometry.num_tgs as u32,
            e_in.len().trailing_zeros(),
            core.num_polys as u32,
            self.batch as u32,
            geometry.len as u32,
        ];
        params.extend_from_slice(&fr_to_u32_limbs(bind.unwrap_or_else(|| Fr::from_u64(0))));
        let cur = dense.cur.device_buffer();
        let nxt = dense.nxt.device_buffer();
        let partials = core.partials.buffer().device_buffer();
        let mut pass = context.begin_pass()?;
        pass.dispatch(
            KernelId::RavDenseRound,
            &params,
            &[&cur, &nxt, &e_in_buffer, &e_out_buffer, &partials],
            geometry.groups,
        );
        pass.run()?;
        testing::note_device_round();
        Ok(core.partials.sums(geometry.num_tgs))
    }
}

impl LazyRaDevice<Fr> for RavDriver {
    fn lazy_lanes(
        &mut self,
        tables: &[Vec<Fr>],
        width: usize,
        e_in: &[Fr],
        e_out: &[Fr],
    ) -> Option<Vec<Fr>> {
        let geometry = self.core.lazy_geometry(width, e_in, e_out)?;
        match self.dispatch_lazy(&geometry, tables, width, e_in, e_out) {
            Ok(lanes) => lanes,
            Err(error) => {
                self.core.device.failed(&error);
                None
            }
        }
    }

    fn adopt_dense(&mut self, tables: &[Vec<Fr>]) -> bool {
        self.core.adopt(tables)
    }

    fn dense_round(&mut self, bind: Option<Fr>, e_in: &[Fr], e_out: &[Fr]) -> Option<Vec<Fr>> {
        let geometry = self.core.dense_geometry(bind.is_some(), e_in, e_out)?;
        match self.dispatch_dense(&geometry, bind, e_in, e_out) {
            Ok(lanes) => {
                self.core.dense_advance(bind.is_some());
                Some(lanes)
            }
            Err(error) => {
                self.core.device.failed(&error);
                None
            }
        }
    }

    fn take_dense(&mut self) -> Vec<Vec<Fr>> {
        self.core.take_dense()
    }
}

/// Lockstep parity with the device drivers forced on (and a mid-sumcheck
/// handoff variant per consumer): byte-equal round polynomials, output
/// claims, and derived-table scalars, with EXACT dispatch counts proving
/// where the device ran.
#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_sumcheck::ProveRounds;
    use jolt_verifier::stages::relations::ConcreteSumcheck;
    use jolt_verifier::stages::stage6b::booleanity::{
        BooleanityCyclePhaseChallenges, BooleanityInputClaims,
    };

    use super::*;
    use crate::metal::testing::{device_probe_count, gpu_lock};
    use crate::optimized::booleanity::testing::with_booleanity_backend;
    use crate::optimized::booleanity::OptimizedBooleanityCycle;

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    fn challenge(round: usize) -> Fr {
        fr(0xC0FF_EE11_D00D_F00D ^ (round as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x2A)
    }

    fn point(seed: u64, len: usize) -> Vec<Fr> {
        (0..len).map(|i| fr(seed + 17 * i as u64)).collect()
    }

    fn splitmix(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Lockstep drive: both kernels see the same challenge and claim
    /// stream; byte-equal wire polynomials pin parity. `initial_claim` must
    /// be honest for kernels that round-check (the RAM RAV CPU recipe);
    /// grid-recovery kernels accept anything.
    fn drive_pair(
        cpu: &mut dyn ProveRounds<Fr>,
        device: &mut dyn ProveRounds<Fr>,
        label: &str,
        initial_claim: Fr,
    ) -> Vec<Fr> {
        let rounds = cpu.num_rounds();
        assert_eq!(rounds, device.num_rounds());
        let mut claim = initial_claim;
        let mut drawn = Vec::new();
        for round in 0..rounds {
            let bind = round.checked_sub(1).map(challenge);
            let cpu_poly = cpu.prove_round(bind, round, claim).unwrap();
            let device_poly = device.prove_round(bind, round, claim).unwrap();
            assert_eq!(
                cpu_poly.coefficients(),
                device_poly.coefficients(),
                "{label}: round {round} polynomial mismatch"
            );
            let r = challenge(round);
            claim = cpu_poly.evaluate(r);
            drawn.push(r);
        }
        let last = challenge(rounds - 1);
        cpu.finish_rounds(last).unwrap();
        device.finish_rounds(last).unwrap();
        drawn
    }

    fn rav_parity(log_t: usize, min_terms: usize, device_rounds: u64) {
        std::env::remove_var("JOLT_METAL_DISABLE");
        std::env::set_var("JOLT_METAL_MIN_TERMS", min_terms.to_string());
        let (num_virtual, per_virtual, chunk_bits) = (4usize, 4usize, 8usize);
        let num_committed = num_virtual * per_virtual;
        let mut state = 0x5EED_0000 + log_t as u64;
        let rows: Arc<Vec<InstructionCycleRow>> = Arc::new(
            (0..1usize << log_t)
                .map(|j| {
                    let index = match j {
                        0 => 0u128,
                        1 => u128::MAX,
                        _ => ((splitmix(&mut state) as u128) << 64) | splitmix(&mut state) as u128,
                    };
                    InstructionCycleRow::new(index, None, false, None, None)
                })
                .collect(),
        );
        let instruction_address = point(300, num_committed * chunk_bits);
        let r_cycle = point(7000, log_t);
        let gamma = fr(0xFEED_5EED);

        let new_kernel = |driver| {
            OptimizedInstructionRaVirtualizationKernel::new_with_driver(
                log_t,
                num_virtual,
                per_virtual,
                &instruction_address,
                &r_cycle,
                chunk_bits,
                Arc::clone(&rows),
                gamma,
                driver,
            )
            .unwrap()
        };
        let mut cpu = new_kernel(None);
        let probes_before = device_probe_count();
        let driver = build_rav_driver(&rows, num_committed, per_virtual, chunk_bits)
            .map(|driver| Box::new(driver) as Box<dyn LazyRaDevice<Fr>>);
        assert!(driver.is_some(), "driver install declined");
        let mut device = new_kernel(driver);

        let _ = drive_pair(
            &mut cpu,
            &mut device,
            "instruction_ra_virtualization",
            fr(0xBEEF),
        );
        let claims =
            jolt_claims::protocols::jolt::relations::instruction::InstructionRaVirtualizationInputClaims {
                instruction_ra: Vec::new(),
            };
        let cpu_outputs = cpu.output_claims(&claims).unwrap();
        let device_outputs = device.output_claims(&claims).unwrap();
        assert_eq!(
            cpu_outputs.committed_instruction_ra,
            device_outputs.committed_instruction_ra
        );
        assert_eq!(
            device_probe_count() - probes_before,
            device_rounds,
            "device dispatch count drifted (lazy + adoption + dense)"
        );
    }

    /// Every phase on device: 3 lazy rounds + the adopting materialization
    /// + one per dense message.
    #[test]
    fn rav_parity_full_device() {
        let _lock = gpu_lock();
        rav_parity(13, 0, 3 + 1 + 10);
    }

    /// The gate declines mid-dense (cycles 8192: lazy 8192/4096/2048,
    /// adopt at 1024, four fused dense rounds down to len 256, then
    /// 128 < 256 hands off with a pending fold).
    #[test]
    fn rav_parity_dense_handoff() {
        let _lock = gpu_lock();
        rav_parity(13, 256, 3 + 1 + 4);
    }

    fn bool_parity(log_t: usize, log_k_chunk: u8, min_terms: usize, device_rounds: u64) {
        std::env::remove_var("JOLT_METAL_DISABLE");
        std::env::set_var("JOLT_METAL_MIN_TERMS", min_terms.to_string());
        with_booleanity_backend(log_t, log_k_chunk, |backend, dimensions| {
            let r_address = point(110, dimensions.log_k_chunk);
            let reference_address = point(700, dimensions.log_k_chunk);
            let reference_cycle = point(400, log_t);
            let relation = jolt_verifier::stages::stage6b::booleanity::Booleanity::new(
                dimensions,
                r_address,
                reference_address,
                reference_cycle,
            );
            let claims = BooleanityInputClaims {
                address_phase: fr(0),
            };
            let points = BooleanityInputClaims {
                address_phase: point(50, dimensions.log_k_chunk),
            };
            let challenges = BooleanityCyclePhaseChallenges { gamma: fr(31) };
            let inputs = || ProverInputs {
                relation: &relation,
                claims: &claims,
                points: &points,
                challenges: &challenges,
            };

            let mut cpu = OptimizedBooleanityCycle
                .prepare(&mut ProofSession::default(), backend, inputs())
                .unwrap();
            let probes_before = device_probe_count();
            let mut device = MetalBooleanityCycle
                .prepare(&mut ProofSession::default(), backend, inputs())
                .unwrap();

            let drawn = drive_pair(
                cpu.as_mut(),
                device.as_mut(),
                "booleanity_cycle",
                fr(0xBEEF),
            );
            assert_eq!(
                cpu.output_claims(&claims).unwrap(),
                device.output_claims(&claims).unwrap()
            );
            let output_points = relation.derive_opening_points(&drawn, &points).unwrap();
            cpu.validate_derived_tables(&relation, &points, &output_points, &challenges)
                .unwrap();
            device
                .validate_derived_tables(&relation, &points, &output_points, &challenges)
                .unwrap();
            assert_eq!(
                device_probe_count() - probes_before,
                device_rounds,
                "device dispatch count drifted (lazy + adoption + dense)"
            );
        });
    }

    /// Every phase on device (real trace-backed witness: hot/cold bytecode
    /// and RAM columns exercise the sentinel gathers).
    #[test]
    fn bool_parity_full_device() {
        let _lock = gpu_lock();
        bool_parity(13, 4, 0, 3 + 1 + 10);
    }

    /// Mid-dense handoff, as the RA-virtualization variant.
    #[test]
    fn bool_parity_dense_handoff() {
        let _lock = gpu_lock();
        bool_parity(13, 4, 256, 3 + 1 + 4);
    }

    fn ram_rav_parity(log_t: usize, min_terms: usize, device_rounds: u64) {
        use jolt_claims::protocols::jolt::geometry::dimensions::committed_address_chunks;
        use jolt_claims::protocols::jolt::geometry::ram::{
            committed_ram_ra, RamRaVirtualizationDimensions,
        };
        use jolt_claims::protocols::jolt::relations::ram::RamRaVirtualizationInputClaims;
        use jolt_claims::NoChallenges;
        use jolt_verifier::stages::stage6b::ram_ra_virtualization::RamRaVirtualization;

        use crate::optimized::testing::{with_ram_fixture, FixtureShape, RamOp};
        use crate::optimized::OptimizedBackend;
        use crate::reference::views::{address_fold, eq_table};

        std::env::remove_var("JOLT_METAL_DISABLE");
        std::env::set_var("JOLT_METAL_MIN_TERMS", min_terms.to_string());
        // ram_k = 256 with the fixture's 4-bit chunks: two committed RA
        // polynomials (the minimum product-grid geometry), hot words on
        // both sides of the chunk boundary, cold padding cycles dominant.
        let chunk_bits = 4usize;
        let shape = FixtureShape { log_t, ram_k: 256 };
        let ops = vec![
            RamOp::Write { word: 3, post: 5 },
            RamOp::Write { word: 200, post: 7 },
            RamOp::Read { word: 200 },
            RamOp::None,
            RamOp::Read { word: 3 },
            RamOp::Write { word: 63, post: 2 },
            RamOp::Read { word: 129 },
            RamOp::Write { word: 255, post: 1 },
        ];
        with_ram_fixture(shape, ops, |witness| {
            let log_k = shape.log_k();
            let num_committed = log_k.div_ceil(chunk_bits);
            assert_eq!(num_committed, 2);
            let ram_reduced_address = point(0xADD2, log_k);
            let ram_reduced_cycle = point(0xC1C2, log_t);
            let relation = RamRaVirtualization::<Fr>::new(
                RamRaVirtualizationDimensions::new(log_t, num_committed),
                ram_reduced_address.clone(),
                ram_reduced_cycle.clone(),
                chunk_bits,
            );

            // The honest reduced claim off the oracle grids (the CPU recipe
            // round-checks, so the drive needs the true chain).
            let chunks = committed_address_chunks(&ram_reduced_address, chunk_bits);
            let folded: Vec<Vec<Fr>> = chunks
                .iter()
                .enumerate()
                .map(|(index, chunk)| {
                    address_fold(witness, committed_ram_ra(index), log_t, chunk).unwrap()
                })
                .collect();
            let eq_cycle = eq_table(&ram_reduced_cycle);
            let input_claim: Fr = (0..1usize << log_t)
                .map(|j| {
                    folded
                        .iter()
                        .fold(eq_cycle[j], |product, table| product * table[j])
                })
                .sum();
            assert_ne!(input_claim, fr(0), "degenerate fixture");

            let claims = RamRaVirtualizationInputClaims {
                ram_ra_reduced: input_claim,
            };
            let points = RamRaVirtualizationInputClaims::<Vec<Fr>>::default();
            let challenges = NoChallenges::default();
            let inputs = || ProverInputs {
                relation: &relation,
                claims: &claims,
                points: &points,
                challenges: &challenges,
            };

            let mut cpu = PrepareKernel::<Fr, _>::prepare(
                &OptimizedBackend,
                &mut ProofSession::default(),
                witness,
                inputs(),
            )
            .unwrap();
            let probes_before = device_probe_count();
            let mut device = MetalRamRaVirtualization
                .prepare(&mut ProofSession::default(), witness, inputs())
                .unwrap();

            let _ = drive_pair(
                cpu.as_mut(),
                device.as_mut(),
                "ram_ra_virtualization",
                input_claim,
            );
            assert_eq!(
                cpu.output_claims(&claims).unwrap().ram_ra,
                device.output_claims(&claims).unwrap().ram_ra,
            );
            assert_eq!(
                device_probe_count() - probes_before,
                device_rounds,
                "device dispatch count drifted (lazy + adoption + dense)"
            );
        });
    }

    /// Every phase on device (real trace-backed witness: mostly-cold RAM
    /// address column exercises the sentinel gathers).
    #[test]
    fn ram_rav_parity_full_device() {
        let _lock = gpu_lock();
        ram_rav_parity(13, 0, 3 + 1 + 10);
    }

    /// Mid-dense handoff, as the instruction variant.
    #[test]
    fn ram_rav_parity_dense_handoff() {
        let _lock = gpu_lock();
        ram_rav_parity(13, 256, 3 + 1 + 4);
    }
}
