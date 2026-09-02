//! Metal stage-8 joint-opening slot: device twin of the optimized
//! joint-opening kernel, fusing the PCS batch opening's per-constituent
//! `fold_rows` passes into one command buffer.
//!
//! # Shape (differs from the sumcheck slots)
//!
//! The joint opening is not a round loop: `prepare` returns lazy
//! [`MultilinearPoly`] views and the PCS batch opener later drives ONE
//! vector-matrix product through them — `RlcSource::fold_rows` calls every
//! constituent with the same `(left, σ)`. The optimized tier walks the
//! shared per-cycle columns once PER TRACE POLYNOMIAL (~42 full-`T` passes).
//! This slot's trace views share a [`FoldEngine`]: the first `fold_rows`
//! call runs one command buffer computing EVERY trace constituent's fold —
//! one dispatch per column family, so total device traffic is one pass over
//! the packed column set — and caches the per-slot results; the remaining
//! calls return cached vectors (revalidated against the exact `(left, σ)`
//! arguments, recomputed on mismatch). Host-side assembly (the RLC
//! combination in `RlcSource`, everything downstream) is untouched.
//!
//! # Device mapping (cycle-major only)
//!
//! Cycle-major placement puts the coefficient of `(cycle, address)` at grid
//! index `cycle + address·2^log_T`, and the batch opening's `σ` never
//! exceeds `log_T` on this placement's real geometries — so a grid index's
//! low σ bits are the CYCLE's low σ bits: every cycle of column class
//! `c = cycle mod 2^σ` lands in output column `c`. One thread per output
//! column owns its accumulators outright (no scatter, no atomics), walks its
//! cycle class coalesced, and gathers `left[j + address·2^(log_T−σ)]`.
//! Address-major placement scatters columns by address (no thread-exclusive
//! ownership) and is never production-derived — those proofs use the
//! optimized twin wholesale, as does any call whose `(left, σ)` falls
//! outside the mapping's domain.
//!
//! Fr sums here regroup only ACROSS exact field additions (ascending-cycle
//! per column on the device vs. range partials on the CPU), so the folds are
//! byte-identical; the i128 increments lift through the same
//! canonical-Montgomery conversion on both sides. Pinned by this module's
//! parity tests and the byte_diff metal arms.
//!
//! # Part 2 design — `combine_hints` windowed MSM (blocked on W3a's g1.metal)
//!
//! `DoryScheme::combine_hints` is stage 8's largest single ALU term
//! (~328k plain double-and-add G1 muls ≈ 1.25 G field-muls @2^23; 2.19 s of
//! the 6.42 s stage @2^22): `combined[row] = Σ_p scalar_p · hint_p[row]`
//! over ~42 ragged hints — per row an independent ≤42-term MSM, 2^ν rows.
//! Device design, once W3a's `g1.metal` (Fq CIOS + Jacobian a=0
//! add/double/mixed-add) lands on gpu/metal-backend:
//!
//! 1. **Host-built gather schedule** (gather-not-scatter): flatten the ragged
//!    hint matrix into CSR-like arrays — `points[]` (affine or Jacobian G1,
//!    hint-major), `scalars[]` (one per hint), and per-row spans — so a
//!    thread reads its row's (point, scalar-index) pairs contiguously and no
//!    two threads write one bucket.
//! 2. **Windowed buckets per row**: c-bit windows (c = 4 fits threadgroup
//!    memory: 15 Jacobian buckets × 96 B = 1.4 KiB per window pass, or
//!    thread-private for one window at a time), `⌈254/c⌉` passes per row
//!    accumulating `bucket[digit] += point`; bucket reduction by the usual
//!    suffix-sum, window combine by `c` doublings — all Jacobian, no
//!    inversions on device.
//! 3. **Row parallelism**: one threadgroup per row (2^ν ≈ 2^15 rows ≫
//!    occupancy), threads split the row's hint terms, threadgroup-reduce
//!    Jacobian partials (the W2 `jk_tg_sum` shape lifted to group adds).
//! 4. **Batch-normalize on the host**: device returns Jacobian rows; host
//!    runs one Montgomery-batch inversion to affine/projective-canonical
//!    form. Parity: group-equality against `scalar_mul` per row PLUS
//!    normalized-coordinate byte equality of the final `DoryHint` rows
//!    (the wire object holds projective `Bn254G1` — normalize both sides
//!    identically before comparing).
//! 5. **Seam**: a `combine_hints` hook on the jolt-dory scheme (post-rebase,
//!    W3a owns adjacent files today) delegating to a `jolt-kernels` device
//!    routine when the metal context is live and `rows · hints` clears the
//!    gate, CPU fallback otherwise — same fail-closed discipline as here.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, PoisonError};

use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, TracePolynomialOrder};
use jolt_field::Fr;
use jolt_poly::MultilinearPoly;
use jolt_utils::unsafe_allocate_zero_vec;
use jolt_witness::JoltWitnessPlane;

use crate::commitment::CommitmentGrid;
use crate::metal::buffers::DeviceBuffer;
use crate::metal::field::{fr_as_u32s_mut, FR_U32_LIMBS};
use crate::metal::runtime::{KernelId, MetalContext, OPENING_MAX_SEL};
use crate::metal::{metal_gate, testing, MetalError};
use crate::opening::JointOpeningPolynomials;
use crate::optimized::opening::{
    build_opening_views, is_block_embedded, OpeningColumns, OpeningView, TraceOpeningPoly,
};
use crate::optimized::OptimizedBackend;
use crate::{KernelError, ProofSession};

const KIND: &str = "joint_opening";

/// Slot front: device fold engine above the [`metal_gate`] threshold on
/// cycle-major grids, the optimized fallback otherwise. Device failures
/// after `prepare` degrade per call (each view folds itself on the CPU —
/// exactly the optimized tier's work).
pub struct MetalJointOpening {
    pub fallback: OptimizedBackend,
}

impl JointOpeningPolynomials<Fr> for MetalJointOpening {
    fn prefetch_session(&self, session: &mut ProofSession) -> ProofSession {
        <OptimizedBackend as JointOpeningPolynomials<Fr>>::prefetch_session(&self.fallback, session)
    }

    #[tracing::instrument(
        skip_all,
        name = "MetalJointOpening::prepare",
        fields(polynomials = polynomials.len(), total_vars = grid.total_vars)
    )]
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<Fr>,
        polynomials: &[JoltCommittedPolynomial],
        precommitted_tables: &BTreeMap<JoltCommittedPolynomial, Vec<Fr>>,
        grid: CommitmentGrid,
    ) -> Result<Vec<Box<dyn MultilinearPoly<Fr>>>, KernelError<Fr>> {
        // Stage 8's hint combination and dory-pcs's reduce-round vector ops
        // follow this prepare inside the PCS batch opening, where no backend
        // value flows — install the device combiner and fold routines for
        // the rest of the proof whenever the device is live (the guards park
        // in the session, uninstalling when the proof's session drops). The
        // hooks decline undersized or failed calls, so installation is
        // tier-selection only; it needs none of the fold engine's geometry,
        // hence its place ahead of the placement gate.
        if MetalContext::global().is_ok() {
            session.park(jolt_dory::install_combine_hints_hook(
                crate::metal::hint_combine::combine_hints_device,
            ));
            session.park(jolt_dory::install_routine_hooks(
                crate::metal::dory_folds::routine_hooks(),
            ));
            session.park(
                dory::backends::arkworks::reduce_hook::install_resident_round_hook(
                    crate::metal::dory_reduce::hooks(),
                ),
            );
            session.park(
                dory::backends::arkworks::pairing_hook::install_multi_pair_hook(
                    crate::metal::miller::multi_pair_device,
                ),
            );
        }

        let cycles = 1usize << grid.log_t;
        if grid.order != TracePolynomialOrder::CycleMajor || !metal_gate(KIND, cycles) {
            return self
                .fallback
                .prepare(session, witness, polynomials, precommitted_tables, grid);
        }
        let context = match MetalContext::global() {
            Ok(context) => context,
            Err(error) => {
                tracing::warn!(
                    slot = KIND,
                    %error,
                    "no device context; using the optimized fallback"
                );
                return self.fallback.prepare(
                    session,
                    witness,
                    polynomials,
                    precommitted_tables,
                    grid,
                );
            }
        };

        // Structural errors propagate — the fallback would fail identically.
        let views =
            build_opening_views::<Fr>(session, witness, polynomials, precommitted_tables, grid)?;
        let trace_ids: Vec<JoltCommittedPolynomial> = polynomials
            .iter()
            .copied()
            .filter(|&id| !is_block_embedded(id))
            .collect();
        // The trace views all share one column pass through their Arc.
        let columns = views.iter().find_map(|view| match view {
            OpeningView::Trace(poly) => Some(Arc::clone(&poly.columns)),
            OpeningView::Block(_) => None,
        });
        let engine = match (columns, FoldPlan::build(grid, &trace_ids)) {
            (Some(columns), Ok(plan)) => Some(Arc::new(FoldEngine {
                columns,
                plan,
                state: Mutex::new(EngineState {
                    context: Some(context),
                    cache: None,
                }),
            })),
            (None, _) => None, // no trace polynomials — nothing to fuse
            (_, Err(reason)) => {
                tracing::warn!(
                    slot = KIND,
                    reason,
                    "device joint opening unavailable; serving the optimized views"
                );
                None
            }
        };

        // The views ARE the optimized tier's — wrapping only redirects
        // fold_rows through the fused engine.
        let mut next_slot = 0usize;
        Ok(views
            .into_iter()
            .map(|view| match view {
                OpeningView::Block(block) => Box::new(block) as Box<dyn MultilinearPoly<Fr>>,
                OpeningView::Trace(inner) => {
                    let slot = next_slot;
                    next_slot += 1;
                    match &engine {
                        Some(engine) => Box::new(MetalTraceOpeningPoly {
                            inner,
                            engine: Arc::clone(engine),
                            slot,
                        }) as Box<dyn MultilinearPoly<Fr>>,
                        None => Box::new(inner),
                    }
                }
            })
            .collect())
    }
}

// ---------------------------------------------------------------------------
// Dispatch plan
// ---------------------------------------------------------------------------

/// The per-cycle streams behind the committed trace columns, keying which
/// [`OpeningColumns`] array a dispatch reads.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ColumnStream {
    RdInc,
    RamInc,
    LookupIndex,
    BytecodePc,
    RamAddress,
}

/// One device dispatch of the fused fold.
enum FamilyDispatch {
    /// A dense increment column: one accumulator, address slot zero.
    Dense { stream: ColumnStream, slot: usize },
    /// Up to [`OPENING_MAX_SEL`] one-hot columns off one shared stream, each
    /// selecting its own chunk of the element as the hot address.
    OneHot {
        stream: ColumnStream,
        /// Engine slots, accumulator-ordered.
        slots: Vec<usize>,
        /// Bit offset per accumulator — `(chunks − 1 − index)·log_k_chunk`,
        /// the [`RaChunkSelector`](jolt_witness::RaChunkSelector) formula.
        shifts: Vec<u32>,
        has_cold: bool,
        elem_words: u32,
    },
}

impl FamilyDispatch {
    fn stream(&self) -> ColumnStream {
        match self {
            Self::Dense { stream, .. } | Self::OneHot { stream, .. } => *stream,
        }
    }
}

/// The wrapped device buffer of `stream`.
#[expect(
    clippy::unwrap_used,
    reason = "device_fold wraps every stream its plan references before dispatching"
)]
fn stream_buffer<'a>(
    buffers: &'a [(ColumnStream, DeviceBuffer<'a>)],
    stream: ColumnStream,
) -> &'a DeviceBuffer<'a> {
    &buffers
        .iter()
        .find(|(candidate, _)| *candidate == stream)
        .unwrap()
        .1
}

/// The fused fold's dispatch schedule, fixed per proof.
struct FoldPlan {
    log_t: usize,
    total_vars: usize,
    sel_mask: u32,
    dispatches: Vec<FamilyDispatch>,
    /// Trace-slot count — every engine slot is covered by exactly one
    /// dispatch accumulator.
    slots: usize,
}

impl FoldPlan {
    /// Schedule `trace_ids` (engine-slot order) over the column streams.
    /// `Err` means the geometry falls outside the device kernels' domain —
    /// the caller serves the optimized views instead.
    fn build(
        grid: CommitmentGrid,
        trace_ids: &[JoltCommittedPolynomial],
    ) -> Result<Self, &'static str> {
        let kc = grid.log_k_chunk;
        if kc == 0 || kc >= 32 {
            return Err("chunk width outside the device fold's domain");
        }
        let count = |matches: fn(JoltCommittedPolynomial) -> bool| {
            trace_ids.iter().copied().filter(|&id| matches(id)).count()
        };
        let instruction_chunks =
            count(|id| matches!(id, JoltCommittedPolynomial::InstructionRa(_)));
        let bytecode_chunks = count(|id| matches!(id, JoltCommittedPolynomial::BytecodeRa(_)));
        let ram_chunks = count(|id| matches!(id, JoltCommittedPolynomial::RamRa(_)));
        // The same shift math as `RaChunkSelector::new` — chunk `index` of
        // `chunks` covers bits [(chunks − 1 − index)·kc, +kc).
        let shift = |chunks: usize, index: usize| -> Result<u32, &'static str> {
            let remaining = chunks
                .checked_sub(index + 1)
                .ok_or("RA chunk index outside its family")?;
            Ok((remaining * kc) as u32)
        };
        if instruction_chunks * kc > 128 || bytecode_chunks * kc > 64 || ram_chunks * kc > 64 {
            return Err("RA chunk family wider than its column stream");
        }

        let mut dispatches = Vec::new();
        let mut instruction: Vec<(usize, u32)> = Vec::new();
        let mut bytecode: Vec<(usize, u32)> = Vec::new();
        let mut ram: Vec<(usize, u32)> = Vec::new();
        for (slot, &id) in trace_ids.iter().enumerate() {
            match id {
                JoltCommittedPolynomial::RdInc => dispatches.push(FamilyDispatch::Dense {
                    stream: ColumnStream::RdInc,
                    slot,
                }),
                JoltCommittedPolynomial::RamInc => dispatches.push(FamilyDispatch::Dense {
                    stream: ColumnStream::RamInc,
                    slot,
                }),
                JoltCommittedPolynomial::InstructionRa(index) => {
                    instruction.push((slot, shift(instruction_chunks, index)?));
                }
                JoltCommittedPolynomial::BytecodeRa(index) => {
                    bytecode.push((slot, shift(bytecode_chunks, index)?));
                }
                JoltCommittedPolynomial::RamRa(index) => {
                    ram.push((slot, shift(ram_chunks, index)?));
                }
                _ => return Err("non-trace polynomial scheduled onto the trace streams"),
            }
        }
        let one_hot = |family: Vec<(usize, u32)>,
                       stream: ColumnStream,
                       has_cold: bool,
                       elem_words: u32,
                       dispatches: &mut Vec<FamilyDispatch>| {
            for chunk in family.chunks(OPENING_MAX_SEL) {
                dispatches.push(FamilyDispatch::OneHot {
                    stream,
                    slots: chunk.iter().map(|&(slot, _)| slot).collect(),
                    shifts: chunk.iter().map(|&(_, shift)| shift).collect(),
                    has_cold,
                    elem_words,
                });
            }
        };
        one_hot(
            instruction,
            ColumnStream::LookupIndex,
            false,
            4,
            &mut dispatches,
        );
        one_hot(
            bytecode,
            ColumnStream::BytecodePc,
            false,
            2,
            &mut dispatches,
        );
        one_hot(ram, ColumnStream::RamAddress, true, 2, &mut dispatches);

        Ok(Self {
            log_t: grid.log_t,
            total_vars: grid.total_vars,
            sel_mask: (1u32 << kc) - 1,
            dispatches,
            slots: trace_ids.len(),
        })
    }
}

// ---------------------------------------------------------------------------
// Fused fold engine
// ---------------------------------------------------------------------------

struct CachedFold {
    sigma: usize,
    left: Vec<Fr>,
    folds: Vec<Vec<Fr>>,
}

struct EngineState {
    /// Device liveness: dropped (with one warning) on the first failure so
    /// every later call folds on the CPU.
    context: Option<&'static MetalContext>,
    cache: Option<CachedFold>,
}

/// The fused device fold shared by every trace view of one proof.
struct FoldEngine {
    columns: Arc<OpeningColumns>,
    plan: FoldPlan,
    state: Mutex<EngineState>,
}

impl FoldEngine {
    /// Slot `slot`'s fold at `(left, σ)`, from the cache or one fused device
    /// pass. `None` means the device cannot serve this call (dead device or
    /// geometry outside the cycle-major mapping) — the caller folds its own
    /// view on the CPU, which is the optimized tier's exact work.
    fn fold_for(&self, slot: usize, left: &[Fr], sigma: usize) -> Option<Vec<Fr>> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(cache) = &state.cache {
            let _serve = tracing::info_span!("opening_fold_serve").entered();
            if cache.sigma == sigma && cache.left == left {
                return Some(cache.folds[slot].clone());
            }
        }
        let context = state.context?;
        // The thread-per-column mapping needs σ ≤ log_T (columns determined
        // by cycles) and the caller's row count to match the grid.
        if sigma > self.plan.log_t
            || left.len() != 1usize << (self.plan.total_vars - sigma)
            || !metal_gate(KIND, self.columns.cycles())
        {
            return None;
        }
        match self.device_fold(context, left, sigma) {
            Ok(folds) => {
                let fold = folds[slot].clone();
                state.cache = Some(CachedFold {
                    sigma,
                    left: left.to_vec(),
                    folds,
                });
                Some(fold)
            }
            Err(error) => {
                tracing::warn!(
                    slot = KIND,
                    %error,
                    "device fold failed; folding the batch on the CPU"
                );
                state.context = None;
                None
            }
        }
    }

    /// One command buffer, one dispatch per plan entry: every trace slot's
    /// fold vector from one pass over the packed column streams.
    fn device_fold(
        &self,
        context: &'static MetalContext,
        left: &[Fr],
        sigma: usize,
    ) -> Result<Vec<Vec<Fr>>, MetalError> {
        let num_cols = 1usize << sigma;
        let steps = self.columns.cycles() >> sigma;
        let row_shift = (self.plan.log_t - sigma) as u32;

        // Wrap each referenced stream once (no-copy when eligible; copies
        // are counted) plus the caller's left vector.
        let buffers_span = tracing::info_span!("opening_fold_buffers").entered();
        let mut copied = 0u64;
        let mut stream_buffers: Vec<(ColumnStream, DeviceBuffer<'_>)> = Vec::new();
        for dispatch in &self.plan.dispatches {
            let stream = dispatch.stream();
            if stream_buffers.iter().any(|(seen, _)| *seen == stream) {
                continue;
            }
            let buffer = match stream {
                ColumnStream::RdInc => context.wrap_slice(&self.columns.rd_inc),
                ColumnStream::RamInc => context.wrap_slice(&self.columns.ram_inc),
                ColumnStream::LookupIndex => context.wrap_slice(&self.columns.lookup_index),
                ColumnStream::BytecodePc => context.wrap_slice(&self.columns.bytecode_pc),
                ColumnStream::RamAddress => context.wrap_slice(&self.columns.ram_address),
            }?;
            copied += u64::from(buffer.was_copied());
            stream_buffers.push((stream, buffer));
        }
        let left_buffer = context.wrap_slice(left)?;
        copied += u64::from(left_buffer.was_copied());
        testing::note_copied_buffers(copied);

        let outs: Vec<DeviceBuffer<'static>> = self
            .plan
            .dispatches
            .iter()
            .map(|dispatch| {
                let accumulators = match dispatch {
                    FamilyDispatch::Dense { .. } => 1,
                    FamilyDispatch::OneHot { slots, .. } => slots.len(),
                };
                context.alloc_u32s(accumulators * num_cols * FR_U32_LIMBS)
            })
            .collect::<Result<_, MetalError>>()?;

        drop(buffers_span);

        let kernel_span = tracing::info_span!("opening_fold_kernel").entered();
        let mut pass = context.begin_pass()?;
        for (dispatch, out) in self.plan.dispatches.iter().zip(&outs) {
            let column = stream_buffer(&stream_buffers, dispatch.stream());
            match dispatch {
                FamilyDispatch::Dense { .. } => {
                    let params = [sigma as u32, steps as u32];
                    pass.dispatch(
                        KernelId::OpeningFoldDense,
                        &params,
                        &[column, &left_buffer, out],
                        num_cols,
                    );
                }
                FamilyDispatch::OneHot {
                    slots,
                    shifts,
                    has_cold,
                    elem_words,
                    ..
                } => {
                    let mut params = vec![
                        sigma as u32,
                        steps as u32,
                        row_shift,
                        *elem_words,
                        u32::from(*has_cold),
                        slots.len() as u32,
                        self.plan.sel_mask,
                    ];
                    let mut padded = shifts.clone();
                    padded.resize(OPENING_MAX_SEL, 0);
                    params.extend_from_slice(&padded);
                    pass.dispatch(
                        KernelId::OpeningFoldOneHot,
                        &params,
                        &[column, &left_buffer, out],
                        num_cols,
                    );
                }
            }
        }
        pass.run()?;
        testing::note_device_round();
        drop(kernel_span);

        // Split each dispatch's accumulator block into per-slot fold vectors.
        let _readback = tracing::info_span!("opening_fold_readback").entered();
        let mut folds: Vec<Vec<Fr>> = (0..self.plan.slots).map(|_| Vec::new()).collect();
        for (dispatch, out) in self.plan.dispatches.iter().zip(&outs) {
            let slots: &[usize] = match dispatch {
                FamilyDispatch::Dense { slot, .. } => std::slice::from_ref(slot),
                FamilyDispatch::OneHot { slots, .. } => slots,
            };
            let mut block: Vec<Fr> = unsafe_allocate_zero_vec(slots.len() * num_cols);
            out.copy_to_u32s(fr_as_u32s_mut(&mut block));
            for &slot in slots.iter().rev() {
                folds[slot] = block.split_off(block.len() - num_cols);
            }
        }
        Ok(folds)
    }
}

// ---------------------------------------------------------------------------
// Trace view
// ---------------------------------------------------------------------------

/// A trace polynomial view whose `fold_rows` goes through the shared fused
/// engine; everything else (and every fallback) is the wrapped optimized
/// view's own lazy math.
struct MetalTraceOpeningPoly {
    inner: TraceOpeningPoly<Fr>,
    engine: Arc<FoldEngine>,
    slot: usize,
}

impl MultilinearPoly<Fr> for MetalTraceOpeningPoly {
    fn num_vars(&self) -> usize {
        self.inner.num_vars()
    }

    fn evaluate(&self, point: &[Fr]) -> Fr {
        self.inner.evaluate(point)
    }

    fn for_each_row(&self, sigma: usize, f: &mut dyn FnMut(usize, &[Fr])) {
        self.inner.for_each_row(sigma, f);
    }

    fn fold_rows(&self, left: &[Fr], sigma: usize) -> Vec<Fr> {
        self.engine
            .fold_for(self.slot, left, sigma)
            .unwrap_or_else(|| self.inner.fold_rows(left, sigma))
    }
}

/// Parity against the optimized views (identical folds, dense tables, and
/// evaluations) with the device path FORCED and probed, plus a synthetic
/// full-width fixture pinning the kernels' chunk extraction, i128 lift, and
/// cold handling against the CPU formulas at a no-copy-eligible scale.
#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use std::marker::PhantomData;
    use std::time::Instant;

    use jolt_claims::protocols::jolt::TracePolynomialOrder;
    use jolt_field::Ring;

    use super::*;
    use crate::metal::testing::{copied_buffer_count, device_probe_count, gpu_lock, seeded_frs};
    use crate::opening::JointOpeningPolynomials;
    use crate::optimized::opening::{TracePlacement, COLD};
    use crate::optimized::testing::{random_scalars, with_ram_fixture, FixtureShape, RamOp};
    use crate::reference::commitment::column_kinds;

    fn force_device_gate() {
        // nextest runs one process per test, so env mutation is safe.
        std::env::remove_var("JOLT_METAL_DISABLE");
        std::env::set_var("JOLT_METAL_MIN_TERMS", "0");
    }

    const LOG_T: usize = 4;
    const RAM_K: usize = 16;

    fn fixture_ops() -> Vec<RamOp> {
        vec![
            RamOp::Write { word: 2, post: 7 },
            RamOp::Read { word: 2 },
            RamOp::None,
            RamOp::Write { word: 5, post: 11 },
            RamOp::Read { word: 9 },
            RamOp::Write { word: 2, post: 3 },
            RamOp::None,
            RamOp::Read { word: 5 },
        ]
    }

    /// The metal slot against the optimized slot on a real synthetic trace:
    /// same dense tables, same folds (σ at, below, and above the device
    /// mapping's domain), same evaluations. Cycle-major must dispatch on the
    /// device (probed); address-major must not (the fallback is wholesale).
    fn assert_slot_parity(order: TracePolynomialOrder, expect_device: bool) {
        let _lock = gpu_lock();
        force_device_gate();
        let shape = FixtureShape {
            log_t: LOG_T,
            ram_k: RAM_K,
        };
        with_ram_fixture(shape, fixture_ops(), |witness| {
            let grid = CommitmentGrid {
                total_vars: LOG_T + 4,
                log_t: LOG_T,
                log_k_chunk: 4,
                order,
            };
            let order_ids = witness.committed_order().unwrap();
            let precommitted_tables = BTreeMap::new();

            let optimized = JointOpeningPolynomials::<Fr>::prepare(
                &OptimizedBackend,
                &mut ProofSession::default(),
                witness,
                &order_ids,
                &precommitted_tables,
                grid,
            )
            .unwrap();
            let slot = MetalJointOpening {
                fallback: OptimizedBackend,
            };
            let rounds_before = device_probe_count();
            let metal = JointOpeningPolynomials::<Fr>::prepare(
                &slot,
                &mut ProofSession::default(),
                witness,
                &order_ids,
                &precommitted_tables,
                grid,
            )
            .unwrap();

            let point = random_scalars(grid.total_vars, 23);
            // σ = log_T exercises the device; the wider σs fall outside the
            // thread-per-column mapping and must take the per-view CPU path.
            let sigmas = [LOG_T, grid.total_vars.div_ceil(2) + 1, grid.total_vars];
            for ((id, optimized), metal) in order_ids.iter().zip(&optimized).zip(&metal) {
                assert_eq!(
                    metal.to_dense().as_ref(),
                    optimized.to_dense().as_ref(),
                    "{id:?}: dense table diverged"
                );
                for sigma in sigmas {
                    let left = random_scalars(1 << (grid.total_vars - sigma), 29 + sigma as u64);
                    assert_eq!(
                        metal.fold_rows(&left, sigma),
                        optimized.fold_rows(&left, sigma),
                        "{id:?}: fold_rows diverged at sigma {sigma}"
                    );
                }
                assert_eq!(
                    metal.evaluate(&point),
                    optimized.evaluate(&point),
                    "{id:?}: evaluation diverged"
                );
            }
            let dispatched = device_probe_count() > rounds_before;
            assert_eq!(
                dispatched, expect_device,
                "device dispatch presence diverged from the placement's eligibility"
            );
        });
    }

    #[test]
    fn joint_opening_cycle_major_matches_optimized() {
        assert_slot_parity(TracePolynomialOrder::CycleMajor, true);
    }

    #[test]
    fn joint_opening_address_major_matches_optimized() {
        assert_slot_parity(TracePolynomialOrder::AddressMajor, false);
    }

    /// Synthetic full-width parity at a no-copy-eligible scale: every column
    /// family live, extreme increments (±, zero, `i128::MIN/MAX`), cold
    /// cycles, all 16 instruction selectors (two device dispatches), byte
    /// equality against the CPU views, zero buffer copies for the column
    /// streams, and a wall-clock print (NOT a benchmark).
    #[test]
    #[expect(clippy::print_stdout, reason = "timing sanity readout")]
    fn joint_opening_device_parity_at_2e18() {
        let _lock = gpu_lock();
        force_device_gate();
        let context = MetalContext::global().unwrap();
        let log_t = 18usize;
        let cycles = 1usize << log_t;
        let kc = 8usize;
        let grid = CommitmentGrid {
            total_vars: log_t + kc,
            log_t,
            log_k_chunk: kc,
            order: TracePolynomialOrder::CycleMajor,
        };

        // Deterministic xorshift stream with planted extremes.
        let mut state = 0x243F_6A88_85A3_08D3u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut rd_inc = Vec::with_capacity(cycles);
        let mut ram_inc = Vec::with_capacity(cycles);
        let mut lookup_index = Vec::with_capacity(cycles);
        let mut bytecode_pc = Vec::with_capacity(cycles);
        let mut ram_address = Vec::with_capacity(cycles);
        for i in 0..cycles {
            rd_inc.push(match i % 7 {
                0 => 0,
                1 => i128::MIN,
                2 => i128::MAX,
                3 => -(next() as i128),
                _ => next() as i128,
            });
            ram_inc.push(match i % 5 {
                0 => 0,
                1 => -1,
                _ => (next() as i128) - (next() as i128),
            });
            lookup_index.push((u128::from(next()) << 64) | u128::from(next()));
            bytecode_pc.push(if i % 11 == 0 { 0 } else { next() % (1 << 16) });
            ram_address.push(if i % 13 == 0 {
                COLD
            } else {
                next() % (1 << 16)
            });
        }
        let columns = Arc::new(OpeningColumns {
            rd_inc,
            ram_inc,
            lookup_index,
            bytecode_pc,
            ram_address,
        });

        let mut ids = vec![
            JoltCommittedPolynomial::RdInc,
            JoltCommittedPolynomial::RamInc,
        ];
        ids.extend((0..16).map(JoltCommittedPolynomial::InstructionRa));
        ids.extend((0..2).map(JoltCommittedPolynomial::BytecodeRa));
        ids.extend((0..2).map(JoltCommittedPolynomial::RamRa));
        let kinds = column_kinds::<Fr>(&ids, grid).unwrap();
        let placement = TracePlacement::new(grid);
        let cpu_views: Vec<TraceOpeningPoly<Fr>> = kinds
            .iter()
            .map(|&kind| TraceOpeningPoly {
                columns: Arc::clone(&columns),
                kind,
                placement,
                _field: PhantomData,
            })
            .collect();

        let engine = Arc::new(FoldEngine {
            columns: Arc::clone(&columns),
            plan: FoldPlan::build(grid, &ids).unwrap(),
            state: Mutex::new(EngineState {
                context: Some(context),
                cache: None,
            }),
        });

        let sigma = grid.total_vars.div_ceil(2);
        let left = seeded_frs(0xF01D, 1 << (grid.total_vars - sigma));

        let copies_before = copied_buffer_count();
        let rounds_before = device_probe_count();
        let start = Instant::now();
        let device_folds: Vec<Vec<Fr>> = (0..ids.len())
            .map(|slot| engine.fold_for(slot, &left, sigma).unwrap())
            .collect();
        let device_wall = start.elapsed();
        assert_eq!(
            device_probe_count() - rounds_before,
            1,
            "the fused fold must run as exactly one device pass"
        );
        // The five column streams (4–8 MiB each) must wrap in place; only
        // the caller's left vector may copy.
        assert!(
            copied_buffer_count() - copies_before <= 1,
            "column streams were copied to the device"
        );

        let start = Instant::now();
        for (slot, (id, cpu_view)) in ids.iter().zip(&cpu_views).enumerate() {
            let expected = cpu_view.fold_rows(&left, sigma);
            assert_eq!(device_folds[slot], expected, "{id:?}: fold diverged");
        }
        let cpu_wall = start.elapsed();
        assert_ne!(
            device_folds[0],
            vec![Fr::from_u64(0); 1 << sigma],
            "degenerate fixture: dense fold is all zero"
        );
        println!(
            "joint opening 2^{log_t}, {} slots: device {device_wall:?}, cpu {cpu_wall:?}",
            ids.len()
        );
    }
}
