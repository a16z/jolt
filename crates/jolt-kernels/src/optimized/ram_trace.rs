//! The shared RAM access columns: one typed trace walk serving every
//! RAM-family kernel in this backend, parked in the [`ProofSession`] so the
//! stage-2 kernel's walk is reused by stages 4, 5 and 6b.
//!
//! The columns are the sparse view of the `(K × T)` RAM grids: per cycle,
//! the remapped word address (or a no-access sentinel) plus the pre- and
//! post-access word values. `ra(k, j)` is 1 exactly at `(addresses[j], j)`;
//! `val(k, j)` walks from the initial state through the writes.
//!
//! Lifetime split: only the stage-2 read-write kernel consumes the value
//! columns (its `prepare` builds the cycle-major matrix and `val_init` from
//! them), so [`RamAccessColumns`] owns them by value and they free with that
//! `prepare`. The address column serves five kernels across stages 2–6b and
//! is the only session-parked carry ([`SharedRamAddresses`]) — 8 B/cycle
//! instead of 24 across the prover's peak window.

use core::marker::PhantomData;
use std::sync::Arc;

#[cfg(feature = "allocative")]
use allocative::Visitor;
use jolt_field::JoltField;
use jolt_poly::EqPolynomial;
#[cfg(feature = "parallel")]
use jolt_utils::FirstErrorLatch;
use jolt_witness::witnesses::{RamReadValue, RamWriteValue, RemappedRamAddress};
#[cfg(feature = "parallel")]
use jolt_witness::RandomAccessRows;
use jolt_witness::{stream_witnesses, JoltWitnessPlane, StreamConsumer, WitnessBundle};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::{KernelError, ProofSession};

/// The row-window size of the streaming column splits below (matches
/// `support::collect_rows`: wide enough to amortize the per-chunk rayon
/// extraction dispatch).
const SPLIT_CHUNK: usize = 1 << 16;

/// The index-parallel split's scatter grain (the whole-range collectors'
/// load-balance tradeoff).
#[cfg(feature = "parallel")]
const PAR_CHUNK: usize = 1 << 14;

/// `addresses` sentinel for cycles with no (remappable) RAM access.
pub(crate) const NO_ACCESS: u64 = u64::MAX;

#[derive(Clone, Copy, Debug, WitnessBundle)]
struct RamAccessBundle {
    address: RemappedRamAddress,
    pre_value: RamReadValue,
    post_value: RamWriteValue,
}

#[derive(Clone, Copy, Debug, WitnessBundle)]
struct RamValueBundle {
    pre_value: RamReadValue,
    post_value: RamWriteValue,
}

#[derive(Clone, Copy, Debug, WitnessBundle)]
struct RamAddressBundle {
    address: RemappedRamAddress,
}

/// The session-parked per-cycle remapped-address column (stage 2 → 6b).
/// [`NO_ACCESS`] when the cycle makes no remappable RAM access (no-ops and
/// address 0).
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(crate) struct SharedRamAddresses(
    #[cfg_attr(feature = "allocative", allocative(visit = visit_shared_addresses))]
    pub(crate)  Arc<Vec<u64>>,
);

/// Bytes of the shared address column, attributed once at the session park.
#[cfg(feature = "allocative")]
fn visit_shared_addresses(addresses: &Arc<Vec<u64>>, visitor: &mut Visitor<'_>) {
    jolt_poly::visit_scalars(addresses.as_ref(), visitor);
}

impl SharedRamAddresses {
    /// The session-shared address column: collected on first request
    /// (whichever RAM kernel prepares first), cloned out as an [`Arc`]
    /// afterwards.
    #[expect(
        clippy::expect_used,
        reason = "the entry is parked by this function right above the read"
    )]
    pub fn shared<F: JoltField>(
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        log_t: usize,
    ) -> Result<Arc<Vec<u64>>, KernelError<F>> {
        if session.state::<Self>().is_none() {
            let cycles = 1usize << log_t;
            let [addresses] =
                collect_split_columns::<F, RamAddressBundle, 1>(witness, cycles, |bundle| {
                    [bundle.address.0.unwrap_or(NO_ACCESS)]
                })?;
            session.park(Self(Arc::new(addresses)));
        }
        let addresses = Arc::clone(
            &session
                .state::<Self>()
                .expect("RAM address column parked above")
                .0,
        );
        // Five kernels across stages 2–6b reclaim this column by type alone;
        // a wrong-domain reclaim means OOB indexing or a silently wrong RA
        // claim from a prefix-covering table, so hard-error like the
        // `PcRow::shared` twin instead of a release-compiled-out assert.
        if addresses.len() != 1usize << log_t {
            return Err(KernelError::TableSizeMismatch {
                table: "session-shared RAM address column".to_owned(),
                expected: 1usize << log_t,
                got: addresses.len(),
            });
        }
        Ok(addresses)
    }
}

/// Collect per-cycle `u64` columns straight into SoA form: index-parallel
/// scatter for slice-backed sources, streaming windows otherwise. `split` is
/// pure per row, so column values and order match a bundle collect + split.
fn collect_split_columns<F: JoltField, B, const N: usize>(
    witness: &dyn JoltWitnessPlane<F>,
    cycles: usize,
    split: impl Fn(&B) -> [u64; N] + Send + Sync,
) -> Result<[Vec<u64>; N], KernelError<F>>
where
    B: WitnessBundle + Copy + Send + Sync,
{
    #[cfg(feature = "parallel")]
    if let Some(access) = witness.random_access() {
        if cycles <= access.cycles() {
            return collect_split_columns_par(&access, cycles, &split);
        }
    }
    struct ColumnSplitter<B, S, const N: usize> {
        columns: [Vec<u64>; N],
        split: S,
        _bundle: PhantomData<B>,
    }
    impl<B, S, const N: usize> StreamConsumer for ColumnSplitter<B, S, N>
    where
        B: WitnessBundle + Copy + Send + Sync,
        S: Fn(&B) -> [u64; N] + Send + Sync,
    {
        type Witness = B;

        fn consume(&mut self, chunk: &[B]) {
            for bundle in chunk {
                for (column, value) in self.columns.iter_mut().zip((self.split)(bundle)) {
                    column.push(value);
                }
            }
        }
    }
    let mut consumers = (ColumnSplitter {
        columns: core::array::from_fn(|_| Vec::with_capacity(cycles)),
        split,
        _bundle: PhantomData::<B>,
    },);
    stream_witnesses(witness, 0..cycles, SPLIT_CHUNK, &mut consumers)?;
    Ok(consumers.0.columns)
}

/// The index-parallel split: rows scatter straight into their offsets of
/// every column's spare capacity (chunked so extraction load-balances).
#[cfg(feature = "parallel")]
fn collect_split_columns_par<F: JoltField, B, const N: usize>(
    access: &RandomAccessRows,
    cycles: usize,
    split: &(impl Fn(&B) -> [u64; N] + Sync),
) -> Result<[Vec<u64>; N], KernelError<F>>
where
    B: WitnessBundle + Copy + Send + Sync,
{
    use core::mem::MaybeUninit;
    let mut columns: [Vec<u64>; N] = core::array::from_fn(|_| Vec::with_capacity(cycles));
    let chunk_count = cycles.div_ceil(PAR_CHUNK).max(1);
    let error = FirstErrorLatch::new();
    {
        let mut chunk_views: Vec<[&mut [MaybeUninit<u64>]; N]> = Vec::with_capacity(chunk_count);
        let mut rests: [&mut [MaybeUninit<u64>]; N] = columns
            .each_mut()
            .map(|column| &mut column.spare_capacity_mut()[..cycles]);
        for chunk_index in 0..chunk_count {
            let take = PAR_CHUNK.min(cycles - chunk_index * PAR_CHUNK);
            let mut views: [&mut [MaybeUninit<u64>]; N] =
                core::array::from_fn(|_| Default::default());
            for (view, rest) in views.iter_mut().zip(rests.iter_mut()) {
                let (head, tail) = core::mem::take(rest).split_at_mut(take);
                *view = head;
                *rest = tail;
            }
            chunk_views.push(views);
        }
        chunk_views
            .into_par_iter()
            .enumerate()
            .for_each(|(chunk_index, mut views)| {
                let base = chunk_index * PAR_CHUNK;
                let take = PAR_CHUNK.min(cycles - base);
                for offset in 0..take {
                    match access.window::<B>(base + offset) {
                        Ok(bundle) => {
                            for (view, value) in views.iter_mut().zip(split(&bundle)) {
                                let _ = view[offset].write(value);
                            }
                        }
                        Err(failure) => {
                            error.record(base + offset, failure);
                            return;
                        }
                    }
                }
            });
    }
    if let Some(failure) = error.take() {
        return Err(failure.into());
    }
    // SAFETY: the error latch is empty, so every chunk ran to completion and
    // initialized its whole span of all N columns (the views partition each
    // column's first `cycles` spare slots exactly).
    unsafe {
        for column in &mut columns {
            column.set_len(cycles);
        }
    }
    Ok(columns)
}

/// The stage-2 full access view: the shared address column plus the pre-
/// and post-access value columns, owned by (and freed with) the stage-2
/// read-write `prepare` — their only consumer.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(crate) struct RamAccessColumns {
    /// The session-shared address column, attributed to the session park
    /// (not here).
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub addresses: Arc<Vec<u64>>,
    /// Pre-access word value per cycle (a read's value, a write's pre-value);
    /// 0 on no-access cycles.
    pub pre_values: Vec<u64>,
    /// Post-access word value per cycle (equals the pre-value for reads).
    pub post_values: Vec<u64>,
}

impl RamAccessColumns {
    /// Collect the full three-column view, sharing the address column with
    /// the session: one fused walk when the column is not parked yet, a
    /// value-only walk when it is.
    pub fn collect_full<F: JoltField>(
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        log_t: usize,
    ) -> Result<Self, KernelError<F>> {
        let cycles = 1usize << log_t;
        // Both walks split rows straight into the SoA columns — a
        // whole-trace bundle staging vector would co-reside with the columns
        // (24-32 B/cycle transiently, at the stage-4 batch's resident peak).
        // Column values and order are unchanged: the split is pure per row,
        // scattered index-parallel for slice-backed sources and concatenated
        // in cycle order by the streaming fallback.
        if session.state::<SharedRamAddresses>().is_some() {
            let addresses = SharedRamAddresses::shared(session, witness, log_t)?;
            let (pre_values, post_values) =
                collect_split_columns::<F, RamValueBundle, 2>(witness, cycles, |bundle| {
                    [bundle.pre_value.0, bundle.post_value.0]
                })
                .map(|[pre_values, post_values]| (pre_values, post_values))?;
            return Ok(Self {
                addresses,
                pre_values,
                post_values,
            });
        }
        let [addresses, pre_values, post_values] =
            collect_split_columns::<F, RamAccessBundle, 3>(witness, cycles, |bundle| {
                [
                    bundle.address.0.unwrap_or(NO_ACCESS),
                    bundle.pre_value.0,
                    bundle.post_value.0,
                ]
            })?;
        let addresses = Arc::new(addresses);
        session.park(SharedRamAddresses(Arc::clone(&addresses)));
        Ok(Self {
            addresses,
            pre_values,
            post_values,
        })
    }

    /// Reconstruct the initial RAM state from the trace and the final-state
    /// oracle: an accessed address's initial value is its first access's
    /// pre-value; a never-accessed address's value never changes, so its
    /// final value IS its initial value.
    ///
    /// WARNING: this is the honest-prover data path — it relies on the trace
    /// being consistent with the final memory image (exactly what the RAM
    /// val/output sumchecks prove). A dishonest witness diverges here and
    /// fails the engine's round checks loudly.
    pub fn reconstruct_val_init<F: JoltField>(&self, val_final: Vec<F>) -> Vec<F> {
        let mut val_init = val_final;
        let mut seen = vec![false; val_init.len()];
        for (&address, &pre_value) in self.addresses.iter().zip(&self.pre_values) {
            if address == NO_ACCESS {
                continue;
            }
            let address = address as usize;
            if !seen[address] {
                seen[address] = true;
                val_init[address] = F::from_u64(pre_value);
            }
        }
        val_init
    }
}

/// Bounds-check every accessed address against the proof's `K`, matching
/// the grid materializers' fail-loud contract.
pub(crate) fn validate_addresses<F: JoltField>(
    addresses: &[u64],
    ram_k: usize,
) -> Result<(), KernelError<F>> {
    if addresses
        .iter()
        .any(|&address| address != NO_ACCESS && address >= ram_k as u64)
    {
        return Err(KernelError::InvariantViolation {
            reason: "RAM access address remapped beyond ram_K",
        });
    }
    Ok(())
}

/// The address-eq fold of the one-hot `ra` grid:
/// `out[j] = Σ_k eq(r_address, k) · ra(k, j) = eq_address[addresses[j]]`
/// (0 on no-access cycles). Reproduces `views::address_fold` of the dense
/// grid without materializing it.
pub(crate) fn fold_addresses<F: JoltField>(addresses: &[u64], eq_address: &[F]) -> Vec<F> {
    addresses
        .iter()
        .map(|&address| {
            if address == NO_ACCESS {
                F::zero()
            } else {
                eq_address[address as usize]
            }
        })
        .collect()
}

/// Rows of eq staging materialized per chunk by [`fold_cycles`] — bounded
/// co-residency (32 MiB of `F`) instead of the full `T`-sized eq table.
const FOLD_CYCLES_CHUNK: usize = 1 << 20;

/// The cycle-eq fold of the one-hot `ra` grid:
/// `out[k] = Σ_j eq(r_cycle, j) · ra(k, j) = Σ_{j : addresses[j] = k} eq_cycle[j]`.
/// Reproduces `views::cycle_fold` of the dense grid without materializing
/// it — or the `T`-sized eq table: eq values come chunk-by-chunk from the
/// `e_hi ⊗ e_lo` split tensor (`eq(r, j) = eq(r_hi, j_hi) · eq(r_lo, j_lo)`,
/// an exact-field product of the same factors as the dense table's entries),
/// and each chunk folds serially in the same ascending-`j` order.
pub(crate) fn fold_cycles<F: JoltField>(addresses: &[u64], r_cycle: &[F], ram_k: usize) -> Vec<F> {
    let mid = r_cycle.len() / 2;
    let (r_hi, r_lo) = r_cycle.split_at(mid);
    let e_hi = EqPolynomial::<F>::evals(r_hi, None);
    let e_lo = EqPolynomial::<F>::evals(r_lo, None);
    let lo_bits = r_lo.len();
    let lo_mask = e_lo.len() - 1;

    let mut out = vec![F::zero(); ram_k];
    let mut staging: Vec<F> = vec![F::zero(); FOLD_CYCLES_CHUNK.min(addresses.len())];
    for (chunk_index, chunk) in addresses.chunks(FOLD_CYCLES_CHUNK).enumerate() {
        let base = chunk_index * FOLD_CYCLES_CHUNK;
        let staging = &mut staging[..chunk.len()];
        let fill = |(offset, slot): (usize, &mut F)| {
            let j = base + offset;
            *slot = e_hi[j >> lo_bits] * e_lo[j & lo_mask];
        };
        #[cfg(feature = "parallel")]
        staging
            .par_iter_mut()
            .enumerate()
            .with_min_len(1 << 10)
            .for_each(fill);
        #[cfg(not(feature = "parallel"))]
        staging.iter_mut().enumerate().for_each(fill);

        for (&address, &eq) in chunk.iter().zip(staging.iter()) {
            if address != NO_ACCESS {
                out[address as usize] += eq;
            }
        }
    }
    out
}
