//! Lazily bound address-folded one-hot selectors for the stage-6b RA
//! virtualization kernels — the legacy `SharedRaPolynomials` /
//! `RaPolynomial` round state machine, generalized over the hot-index
//! source.
//!
//! The direct shape materializes every committed selector dense over the
//! cycle domain at prepare: `N × T` field elements, the stage-6b memory wall
//! at scale (the committed instruction RA family alone is `8 × T`). But an
//! unbound selector column is a point mass — `ra_i(·, j)` is
//! `eq(r_chunk_i, chunk_i(j))`, one scale-table lookup per cycle — and the
//! first cycle binds preserve that structure: after `b < 4` binds the bound
//! value at index `j` is the gather
//!
//! ```text
//! value(i, j) = Σ_{offset < 2^b} branch_tables[i][offset][index(i, j·2^b + offset)]
//! ```
//!
//! where branch table `offset` is the base scale table pre-scaled by that
//! offset's bound-bit eq weight (legacy `SharedRaRound1→2→3` pre-scaling).
//! Pre-scaling keeps the round-loop gathers multiplication-free — one table
//! lookup and one addition per branch — because the eq weights are folded
//! into the `N × 2^b × 2^w` tables at bind time (a few thousand entries)
//! instead of multiplied per cycle. Only the fourth bind materializes dense
//! vectors, at `T/16` length, and drops the index source. Peak memory falls
//! from `N·T` field elements to the index source plus `N·T/16`.
//!
//! Byte parity: every gathered value is the same polynomial of the same
//! table entries and challenges as the iterated `lo + r·(hi − lo)` dense
//! bind — identical monomials, exact field algebra (pre-scaling only
//! reassociates the weight product) — so round messages and output claims
//! are bit-identical. The consumers' in-module parity tests pin this
//! against the naive dense path.

use jolt_field::JoltField;
use jolt_poly::{BindingOrder, Polynomial};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Per-cycle hot indices of `N` committed one-hot selector polynomials over
/// a shared compact backing store (typed witness rows, packed columns).
pub(crate) trait ChunkIndexSource: Send + Sync {
    /// Number of selector polynomials served.
    fn num_polys(&self) -> usize;

    /// The unbound cycle-domain length.
    fn cycles(&self) -> usize;

    /// The scale-table index of polynomial `i`'s hot address at unbound
    /// cycle `j`; `None` when the cycle is cold for that polynomial.
    fn index(&self, i: usize, j: usize) -> Option<usize>;
}

/// A device tier for the round-message mass of a lazy-RA consumer: the
/// summand is baked into the driver instance at construction (each consumer
/// kernel installs a driver that emits ITS lane layout — see the consumers'
/// `device_lanes` call sites for the contracts). Every method is effect-free
/// on `None`/`false` so the CPU paths can always run the same round from
/// unchanged state; the dense fused round writes only its ping-pong target
/// and partials, never the live table.
pub(crate) trait LazyRaDevice<F: JoltField>: Send + Sync {
    /// Notify a device whose lazy-round auxiliary state follows the same
    /// low-to-high bind schedule. Most consumers have no auxiliary table.
    fn bind_lazy(&mut self, _challenge: F) {}

    /// Last lazy branch width before dense adoption. The default `4` is the
    /// legacy schedule (three lazy rounds, then [`adopt_dense`](Self::adopt_dense)
    /// materializes at width 8). A driver returning `8` defers adoption one
    /// round: a fourth lazy round runs at width 8, and the adoption then
    /// rides [`launch_adopt`](Self::launch_adopt)/[`adopt_round`](Self::adopt_round)
    /// at width 16 — half the dense footprint, fused with that round's
    /// message. Only `4` and `8` are meaningful.
    fn lazy_horizon(&self) -> usize {
        4
    }

    /// Fused adoption round, synchronous: materialize every polynomial dense
    /// at `cycles / width` from the width-`width` branch tables AND emit this
    /// round's message lanes in one dispatch. `Some(lanes)` = adopted (the
    /// driver owns the dense tables); `None` = nothing ran or the device
    /// failed — the caller materializes on the CPU from the unchanged
    /// tables/source.
    fn adopt_round(
        &mut self,
        _tables: &[Vec<F>],
        _width: usize,
        _e_in: &[F],
        _e_out: &[F],
    ) -> Option<Vec<F>> {
        None
    }

    /// Launch the fused adoption round without blocking; as
    /// [`launch_lazy`](Self::launch_lazy). Collection rides
    /// [`collect_lanes`](Self::collect_lanes): `Some` installs the dense
    /// tables, `None` leaves the tables/source unchanged for the CPU
    /// recovery.
    fn launch_adopt(
        &mut self,
        _tables: &[Vec<F>],
        _width: usize,
        _e_in: &[F],
        _e_out: &[F],
    ) -> bool {
        false
    }

    /// Lazy-phase message lanes against the CURRENT branch tables
    /// (offset-major, per-poly `width · 2^w` entries) and gruen levels.
    fn lazy_lanes(
        &mut self,
        tables: &[Vec<F>],
        width: usize,
        e_in: &[F],
        e_out: &[F],
    ) -> Option<Vec<F>>;

    /// The third bind's materialization: gather every polynomial dense at
    /// `cycles / 8` into device-resident round state (from the DOUBLED
    /// width-8 branch tables). `true` = adopted; the driver owns the dense
    /// tables until [`take_dense`](Self::take_dense).
    fn adopt_dense(&mut self, tables: &[Vec<F>]) -> bool;

    /// One fused dense round: fold `bind` (when present) low-to-high, then
    /// the message lanes against the CURRENT (post-bind) gruen levels.
    /// `None` steps aside pre-round — the live tables are still the
    /// pre-`bind` state.
    fn dense_round(&mut self, bind: Option<F>, e_in: &[F], e_out: &[F]) -> Option<Vec<F>>;

    /// Hand the live dense tables back at their current length. Infallible
    /// for an adopted driver: the buffers are host-visible whatever the
    /// device's health.
    fn take_dense(&mut self) -> Vec<Vec<F>>;

    /// Launch this round's lazy-phase lanes without blocking, leaving the
    /// dispatch in flight for [`collect_lanes`](Self::collect_lanes).
    /// `false` = nothing launched (gate declined, unhealthy device,
    /// ineligible buffers) — the caller uses the synchronous paths. A
    /// flight owns copies of its per-round uploads (`e_in`/`e_out`, the
    /// branch tables' flattening), so the caller's backings stay free.
    fn launch_lazy(
        &mut self,
        _tables: &[Vec<F>],
        _width: usize,
        _e_in: &[F],
        _e_out: &[F],
    ) -> bool {
        false
    }

    /// Launch the fused dense round without blocking; as
    /// [`launch_lazy`](Self::launch_lazy). The pending `bind` folds inside
    /// the launched kernel, but the driver's dense state advances only at a
    /// successful collect — a failed flight leaves the pre-bind tables
    /// intact for the host recovery.
    fn launch_dense(&mut self, _bind: Option<F>, _e_in: &[F], _e_out: &[F]) -> bool {
        false
    }

    /// Collect a launched round's lanes: `Some` on success (a dense flight
    /// advances the ping-pong here); `None` when the wait surfaced a device
    /// failure — the driver has latched off with its state exactly as the
    /// synchronous failure paths leave it (lazy: untouched; dense: intact
    /// pre-bind `cur`).
    fn collect_lanes(&mut self) -> Option<Vec<F>> {
        None
    }
}

/// `N` address-folded selector columns bound `LowToHigh`, lazily until the
/// horizon bind materializes dense (the fourth without a driver — see
/// [`LazyRaDevice::lazy_horizon`]).
#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField, S: allocative::Allocative")
)]
pub(crate) enum LazyFoldedRa<F: JoltField, S> {
    /// Fewer than the horizon's binds: per-polynomial branch scale tables
    /// (the base table pre-scaled by each bound-bit pattern's eq weight),
    /// flattened offset-major — `tables[i][offset · stride_i + k]` with
    /// `stride_i = tables[i].len() / width` — plus the compact index source.
    Lazy {
        #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalar_rows))]
        tables: Vec<Vec<F>>,
        /// Bound-bit branch count (`2^binds`: 1, 2, 4, or 8).
        width: usize,
        source: S,
        /// Optional device tier; `None` keeps every phase on the CPU.
        #[cfg_attr(feature = "allocative", allocative(skip))]
        driver: Option<Box<dyn LazyRaDevice<F>>>,
    },
    /// A deferred-horizon driver's adoption bind has landed (tables doubled
    /// to `width = 2 · horizon`) but the fused adoption round has not run
    /// yet: the next message call adopts on the device
    /// ([`LazyRaDevice::adopt_round`] / [`LazyRaDevice::launch_adopt`]), or
    /// the CPU materializes from these unchanged tables on any decline.
    PendingAdopt {
        #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalar_rows))]
        tables: Vec<Vec<F>>,
        width: usize,
        source: S,
        #[cfg_attr(feature = "allocative", allocative(skip))]
        driver: Box<dyn LazyRaDevice<F>>,
    },
    /// Three or more binds with a device driver: the dense tables live in
    /// the driver's buffers; `pending_bind` is a challenge not yet folded
    /// (the device folds it fused with the next round, or
    /// [`ensure_host`](Self::ensure_host) applies it on the CPU).
    DeviceDense {
        #[cfg_attr(feature = "allocative", allocative(skip))]
        driver: Box<dyn LazyRaDevice<F>>,
        #[cfg_attr(feature = "allocative", allocative(skip))]
        pending_bind: Option<F>,
        num_polys: usize,
    },
    /// Four or more binds: plain dense multilinears (`T/16` at entry).
    Dense(Vec<Polynomial<F>>),
}

impl<F: JoltField, S: ChunkIndexSource> LazyFoldedRa<F, S> {
    /// One scale table per selector polynomial, in polynomial order.
    pub(crate) fn new(tables: Vec<Vec<F>>, source: S) -> Self {
        Self::new_with_driver(tables, source, None)
    }

    /// As [`new`](Self::new) with a device tier installed.
    pub(crate) fn new_with_driver(
        tables: Vec<Vec<F>>,
        source: S,
        driver: Option<Box<dyn LazyRaDevice<F>>>,
    ) -> Self {
        debug_assert_eq!(tables.len(), source.num_polys());
        Self::Lazy {
            tables,
            width: 1,
            source,
            driver,
        }
    }

    pub(crate) fn num_polys(&self) -> usize {
        match self {
            Self::Lazy { tables, .. } | Self::PendingAdopt { tables, .. } => tables.len(),
            Self::DeviceDense { num_polys, .. } => *num_polys,
            Self::Dense(polys) => polys.len(),
        }
    }

    /// Resolve a pending adoption on the CPU: materialize dense from the
    /// held tables at their current width. No-op in any other state.
    fn materialize_pending(&mut self) {
        let Self::PendingAdopt { .. } = self else {
            return;
        };
        let Self::PendingAdopt {
            tables,
            width,
            source,
            ..
        } = std::mem::replace(self, Self::Dense(Vec::new()))
        else {
            unreachable!("just matched");
        };
        *self = Self::Dense(materialize(&tables, &source, width));
    }

    /// Promote a successfully adopted pending state to device-dense. The
    /// driver already owns the materialized tables.
    fn promote_adopted(&mut self) {
        let Self::PendingAdopt { .. } = self else {
            return;
        };
        let Self::PendingAdopt { tables, driver, .. } =
            std::mem::replace(self, Self::Dense(Vec::new()))
        else {
            unreachable!("just matched");
        };
        *self = Self::DeviceDense {
            num_polys: tables.len(),
            driver,
            pending_bind: None,
        };
    }

    /// The current (bound) evaluation of polynomial `i` at index `j` —
    /// exactly the value a dense representation would hold after the same
    /// binds. Callers reach device-resident state only through
    /// [`device_lanes`](Self::device_lanes) /
    /// [`ensure_host`](Self::ensure_host), so the `DeviceDense` arm is
    /// unreachable here by sequencing.
    #[inline]
    pub(crate) fn value(&self, i: usize, j: usize) -> F {
        match self {
            Self::Lazy {
                tables,
                width,
                source,
                ..
            }
            | Self::PendingAdopt {
                tables,
                width,
                source,
                ..
            } => gather(&tables[i], *width, source, i, j),
            Self::DeviceDense { .. } => unreachable!("device-resident tables: ensure_host first"),
            Self::Dense(polys) => polys[i].evals()[j],
        }
    }

    /// The `(lo, hi) = (value(i, 2·row), value(i, 2·row + 1))` pair the
    /// round messages consume.
    #[inline]
    pub(crate) fn lo_hi(&self, i: usize, row: usize) -> (F, F) {
        (self.value(i, 2 * row), self.value(i, 2 * row + 1))
    }

    /// All polynomials' `(lo, hi)` pairs at `row`, into `out` (length
    /// `num_polys`). One state dispatch per row instead of `2N`, with
    /// per-polynomial table slices hoisted out of the gather loop — the
    /// round-message hot path.
    #[inline]
    pub(crate) fn lo_hi_all(&self, row: usize, out: &mut [(F, F)]) {
        match self {
            Self::Lazy {
                tables,
                width,
                source,
                ..
            }
            | Self::PendingAdopt {
                tables,
                width,
                source,
                ..
            } => {
                let width = *width;
                for (i, (out, table)) in out.iter_mut().zip(tables).enumerate() {
                    *out = (
                        gather(table, width, source, i, 2 * row),
                        gather(table, width, source, i, 2 * row + 1),
                    );
                }
            }
            Self::DeviceDense { .. } => unreachable!("device-resident tables: ensure_host first"),
            Self::Dense(polys) => {
                for (out, poly) in out.iter_mut().zip(polys) {
                    let evals = poly.evals();
                    *out = (evals[2 * row], evals[2 * row + 1]);
                }
            }
        }
    }

    /// The fully bound claims, in polynomial order (any state, so short
    /// cycle geometries extract correctly).
    pub(crate) fn final_values(&self) -> Vec<F> {
        (0..self.num_polys()).map(|i| self.value(i, 0)).collect()
    }

    /// Bind the next cycle variable `LowToHigh`: re-scale the branch tables
    /// below the lazy horizon (the fourth bind without a driver),
    /// materialize dense at the horizon — into the driver's device buffers
    /// when it adopts, on the CPU (dropping the source) otherwise — plain
    /// multilinear binds after. A deferred-horizon driver's adoption bind
    /// only doubles the tables here ([`PendingAdopt`](Self::PendingAdopt));
    /// the driver materializes fused with the next round's message. A
    /// device-resident bind is only RECORDED here; the driver folds it fused
    /// with the next round's message (or [`ensure_host`](Self::ensure_host)
    /// applies it).
    pub(crate) fn bind(&mut self, challenge: F) {
        *self = match std::mem::replace(self, Self::Dense(Vec::new())) {
            Self::Lazy {
                tables,
                width,
                source,
                mut driver,
            } => {
                if let Some(driver) = driver.as_mut() {
                    driver.bind_lazy(challenge);
                }
                let horizon = driver.as_ref().map_or(8, |driver| driver.lazy_horizon());
                let tables = double_branches(tables, challenge);
                if width < horizon {
                    Self::Lazy {
                        tables,
                        width: width * 2,
                        source,
                        driver,
                    }
                } else if horizon > 4 && driver.is_some() {
                    #[expect(clippy::unwrap_used, reason = "is_some just checked")]
                    Self::PendingAdopt {
                        width: width * 2,
                        tables,
                        source,
                        driver: driver.unwrap(),
                    }
                } else if driver
                    .as_mut()
                    .is_some_and(|driver| driver.adopt_dense(&tables))
                {
                    Self::DeviceDense {
                        num_polys: tables.len(),
                        // The `is_some_and` above just proved it.
                        #[expect(clippy::unwrap_used, reason = "adoption implies presence")]
                        driver: driver.unwrap(),
                        pending_bind: None,
                    }
                } else {
                    Self::Dense(materialize(&tables, &source, width * 2))
                }
            }
            // Reachable only when the adoption bind is the sumcheck's LAST
            // (`finish_rounds` at tiny cycle geometries): no message follows,
            // so resolve on the CPU and bind dense.
            Self::PendingAdopt {
                tables,
                width,
                source,
                ..
            } => {
                let mut polys = materialize(&tables, &source, width);
                for poly in &mut polys {
                    poly.bind_with_order(challenge, BindingOrder::LowToHigh);
                }
                Self::Dense(polys)
            }
            Self::DeviceDense {
                driver,
                pending_bind,
                num_polys,
            } => {
                // Never two challenges deep: every bind is followed by a
                // message or the terminal ensure_host.
                debug_assert!(pending_bind.is_none());
                Self::DeviceDense {
                    driver,
                    pending_bind: Some(challenge),
                    num_polys,
                }
            }
            Self::Dense(mut polys) => {
                for poly in &mut polys {
                    poly.bind_with_order(challenge, BindingOrder::LowToHigh);
                }
                Self::Dense(polys)
            }
        };
    }

    /// The device tier's shot at this round's message: lazy-phase lanes
    /// against the current branch tables, or the fused dense round (folding
    /// any pending challenge). `None` means the CPU must produce this
    /// round's message — from state this call has already normalized
    /// (a declined dense round hands the tables back and applies the
    /// pending fold, so the ordinary CPU paths just work).
    pub(crate) fn device_lanes(&mut self, e_in: &[F], e_out: &[F]) -> Option<Vec<F>> {
        match self {
            Self::Lazy {
                tables,
                width,
                driver: Some(driver),
                ..
            } => driver.lazy_lanes(tables, *width, e_in, e_out),
            Self::PendingAdopt {
                tables,
                width,
                driver,
                ..
            } => {
                if let Some(lanes) = driver.adopt_round(tables, *width, e_in, e_out) {
                    self.promote_adopted();
                    Some(lanes)
                } else {
                    self.materialize_pending();
                    None
                }
            }
            Self::DeviceDense {
                driver,
                pending_bind,
                ..
            } => {
                if let Some(lanes) = driver.dense_round(*pending_bind, e_in, e_out) {
                    *pending_bind = None;
                    Some(lanes)
                } else {
                    self.ensure_host();
                    None
                }
            }
            _ => None,
        }
    }

    /// Two-phase variant of [`device_lanes`](Self::device_lanes): launch the
    /// device tier's round without blocking. `true` = in flight — the caller
    /// must follow with [`collect_device_lanes`](Self::collect_device_lanes)
    /// before any bind or table access. `false` = nothing launched, and (as
    /// with a `device_lanes` decline) the state is already normalized for
    /// the CPU paths.
    pub(crate) fn launch_device_lanes(&mut self, e_in: &[F], e_out: &[F]) -> bool {
        match self {
            Self::Lazy {
                tables,
                width,
                driver: Some(driver),
                ..
            } => driver.launch_lazy(tables, *width, e_in, e_out),
            Self::PendingAdopt {
                tables,
                width,
                driver,
                ..
            } => {
                if driver.launch_adopt(tables, *width, e_in, e_out) {
                    true
                } else {
                    self.materialize_pending();
                    false
                }
            }
            Self::DeviceDense {
                driver,
                pending_bind,
                ..
            } => {
                if driver.launch_dense(*pending_bind, e_in, e_out) {
                    true
                } else {
                    self.ensure_host();
                    false
                }
            }
            _ => false,
        }
    }

    /// Collect the lanes of the round launched by
    /// [`launch_device_lanes`](Self::launch_device_lanes). `None` = the wait
    /// failed; the state is normalized for a CPU recompute of the SAME round
    /// (lazy: untouched tables; dense: reclaimed host tables with the
    /// pending fold applied).
    pub(crate) fn collect_device_lanes(&mut self) -> Option<Vec<F>> {
        match self {
            Self::Lazy {
                driver: Some(driver),
                ..
            } => driver.collect_lanes(),
            Self::PendingAdopt { driver, .. } => {
                if let Some(lanes) = driver.collect_lanes() {
                    self.promote_adopted();
                    Some(lanes)
                } else {
                    self.materialize_pending();
                    None
                }
            }
            Self::DeviceDense {
                driver,
                pending_bind,
                ..
            } => {
                if let Some(lanes) = driver.collect_lanes() {
                    *pending_bind = None;
                    Some(lanes)
                } else {
                    self.ensure_host();
                    None
                }
            }
            _ => None,
        }
    }

    /// Reclaim device-resident tables into ordinary dense state, applying
    /// any pending fold. No-op otherwise. Consumers call this before any
    /// direct table access (`value`, `lo_hi_all`, `final_values`) that could
    /// follow a device phase — e.g. at output claims.
    pub(crate) fn ensure_host(&mut self) {
        if matches!(self, Self::PendingAdopt { .. }) {
            self.materialize_pending();
            return;
        }
        let Self::DeviceDense {
            driver,
            pending_bind,
            ..
        } = self
        else {
            return;
        };
        let pending = *pending_bind;
        let mut polys: Vec<Polynomial<F>> = driver
            .take_dense()
            .into_iter()
            .map(Polynomial::new)
            .collect();
        if let Some(challenge) = pending {
            for poly in &mut polys {
                poly.bind_with_order(challenge, BindingOrder::LowToHigh);
            }
        }
        *self = Self::Dense(polys);
    }

    /// Permanently remove the device tier while preserving the current RA
    /// state. Lazy tables already live on the host; dense tables are
    /// reclaimed through the ordinary recovery path.
    pub(crate) fn disable_device(&mut self) {
        if let Self::Lazy { driver, .. } = self {
            *driver = None;
        } else {
            self.ensure_host();
        }
    }
}

/// The eq-weighted branch gather at unbound width `width`: one lookup and
/// one add per hot branch, no multiplications (the weights are pre-scaled
/// into the branch tables).
#[inline]
fn gather<F: JoltField, S: ChunkIndexSource>(
    table: &[F],
    width: usize,
    source: &S,
    i: usize,
    j: usize,
) -> F {
    if width == 1 {
        return source.index(i, j).map_or_else(F::zero, |k| table[k]);
    }
    let stride = table.len() / width;
    let mut sum = F::zero();
    let mut base = 0;
    for offset in 0..width {
        if let Some(k) = source.index(i, j * width + offset) {
            sum += table[base + k];
        }
        base += stride;
    }
    sum
}

/// Doubles every polynomial's branch set for the next bound bit: the first
/// half keeps the existing branches scaled by `1 − challenge` (bit 0), the
/// second half by `challenge` (bit 1) — offset layout
/// `b0 + 2·b1 + 4·b2`, matching the low bits of the original cycle index.
fn double_branches<F: JoltField>(tables: Vec<Vec<F>>, challenge: F) -> Vec<Vec<F>> {
    let one_minus = F::one() - challenge;
    let double = |table: Vec<F>| -> Vec<F> {
        let mut next = Vec::with_capacity(table.len() * 2);
        next.extend(table.iter().map(|value| one_minus * *value));
        next.extend(table.iter().map(|value| challenge * *value));
        next
    };
    #[cfg(feature = "parallel")]
    {
        tables.into_par_iter().map(double).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        tables.into_iter().map(double).collect()
    }
}

/// The switching bind's materialization: gather every polynomial dense at
/// `cycles / branches` length through the pre-scaled branch tables —
/// lookups and adds only. The switch depth trades the dense tables'
/// footprint (`N · T / branches` field elements — the stage-6b peak at
/// large T) against one more gather round and double the branch tables;
/// measured on a 64-thread host, T/16 beats the original T/8 on both axes.
fn materialize<F: JoltField, S: ChunkIndexSource>(
    tables: &[Vec<F>],
    source: &S,
    branches: usize,
) -> Vec<Polynomial<F>> {
    debug_assert!(source.cycles() >= branches);
    let new_len = source.cycles() / branches;
    let materialize_poly = |i: usize| -> Polynomial<F> {
        let table = tables[i].as_slice();
        let eval = |j: usize| gather(table, branches, source, i, j);
        #[cfg(feature = "parallel")]
        let evals: Vec<F> = (0..new_len).into_par_iter().map(eval).collect();
        #[cfg(not(feature = "parallel"))]
        let evals: Vec<F> = (0..new_len).map(eval).collect();
        Polynomial::new(evals)
    };
    #[cfg(feature = "parallel")]
    {
        (0..tables.len())
            .into_par_iter()
            .map(materialize_poly)
            .collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        (0..tables.len()).map(materialize_poly).collect()
    }
}
