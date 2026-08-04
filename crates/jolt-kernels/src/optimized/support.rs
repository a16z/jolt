//! Small shared primitives of the optimized kernels.

use std::ops::Range;

use jolt_field::{Field, RingAccumulator, SignedScalarAccumulator};
use jolt_poly::{
    BindingOrder, EqPolynomial, GruenSplitEqPolynomial, LtPolynomial, Polynomial, UnivariatePoly,
};
use jolt_sumcheck::SumcheckError;
use jolt_witness::{
    collect_bundles_par, stream_witnesses, RowSource, StreamConsumer, WitnessBundle, WitnessError,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::{KernelError, SumcheckKernelError};

/// A kernel's bound-round count against its total — the one home of the
/// "claims only after every round is bound" invariant.
pub(crate) struct RoundProgress {
    bound: usize,
    total: usize,
}

impl RoundProgress {
    pub(crate) fn new(total: usize) -> Self {
        Self { bound: 0, total }
    }

    /// Total rounds — the kernel's `ProveRounds::num_rounds`.
    pub(crate) fn total(&self) -> usize {
        self.total
    }

    /// Rounds bound so far (multi-phase kernels key their transitions on it).
    pub(crate) fn bound(&self) -> usize {
        self.bound
    }

    /// Record one bound round.
    pub(crate) fn advance(&mut self) {
        self.bound += 1;
    }

    /// Gate for every output-claim / derived-table entry point.
    pub(crate) fn require_complete<F: Field>(&self) -> Result<(), SumcheckKernelError<F>> {
        if self.bound == self.total {
            Ok(())
        } else {
            Err(SumcheckKernelError::NotFullyBound {
                remaining: self.total - self.bound,
            })
        }
    }
}

/// The streaming chunk of [`collect_rows`]: large enough that the per-chunk
/// rayon extraction dispatch amortizes (the stock bundle pass uses 2^12-row
/// chunks — at 2^23 cycles the two thousand dispatches rival the extraction
/// itself).
const COLLECT_ROWS_CHUNK: usize = 1 << 16;

/// `jolt_witness::collect_bundles` with a wider streaming chunk and a
/// pre-sized destination (the stock pass also grows its vector realloc by
/// realloc). Chunk size never changes the collected bundles — the pass
/// carries the lookahead row across chunk boundaries — so this is walk-shape
/// only.
pub(crate) fn collect_rows<B: WitnessBundle + Copy + Send + Sync>(
    source: &(impl RowSource + ?Sized),
    cycles: usize,
) -> Result<Vec<B>, WitnessError> {
    // Slice-backed sources collect index-parallel — no chunk staging, no
    // serial consume copy (out-of-range requests fall through for the
    // walk's validation).
    if let Some(access) = source.random_access() {
        if cycles <= access.cycles() {
            return collect_bundles_par(&access, cycles);
        }
    }
    struct Presized<B> {
        rows: Vec<B>,
    }
    impl<B: WitnessBundle + Copy + Send + Sync> StreamConsumer for Presized<B> {
        type Witness = B;

        fn consume(&mut self, chunk: &[B]) {
            self.rows.extend_from_slice(chunk);
        }
    }
    let mut consumers = (Presized::<B> {
        rows: Vec::with_capacity(cycles),
    },);
    stream_witnesses(source, 0..cycles, COLLECT_ROWS_CHUNK, &mut consumers)?;
    Ok(consumers.0.rows)
}

/// Accumulates `Π factors` into `lane`, fusing the last multiply into the
/// deferred-reduction accumulator. Requires at least two factors.
#[inline]
pub(crate) fn accumulate_product<F: Field>(factors: &[F], lane: &mut F::Accumulator) {
    debug_assert!(factors.len() >= 2);
    let last = factors.len() - 1;
    let mut product = factors[0];
    for factor in &factors[1..last] {
        product *= *factor;
    }
    lane.fmadd(product, factors[last]);
}

/// Accumulate `eq · F(value)` for a full-range `u64` on the small-scalar
/// accumulator without overflowing it: the accumulator's headroom is one
/// extra limb, which ~4 full-magnitude `field × u64` products exhaust, so the
/// value is split into u32 halves (products ≤ 2^286, headroom ≥ 2^34 terms).
/// `eq_shifted` must be `eq · 2^32`; the two fused adds sum to exactly
/// `eq · F(value)`.
#[inline]
pub(crate) fn fmadd_u64_split<F: Field>(
    accumulator: &mut F::SmallScalarAccumulator,
    eq: F,
    eq_shifted: F,
    value: u64,
) {
    accumulator.fmadd_u64(eq_shifted, value >> 32);
    accumulator.fmadd_u64(eq, value & 0xFFFF_FFFF);
}

/// `[1, γ, γ², …, γ^{N−1}]`.
pub(crate) fn gamma_powers_array<F: Field, const N: usize>(gamma: F) -> [F; N] {
    let mut powers = [F::one(); N];
    for i in 1..N {
        powers[i] = powers[i - 1] * gamma;
    }
    powers
}

/// `[1, γ, γ², …]` of length `count`.
pub(crate) fn gamma_powers<F: Field>(gamma: F, count: usize) -> Vec<F> {
    let mut powers = Vec::with_capacity(count);
    let mut power = F::one();
    for _ in 0..count {
        powers.push(power);
        power *= gamma;
    }
    powers
}

/// `(γ^i, γ^{-i})` pairs for pre-scaled shared tables. The inverse powers
/// unscale the final claims back to the committed polynomials' values;
/// `γ^i · γ^{-i} = 1` exactly, so unscaling is byte-exact. `reason` names
/// the batching challenge in the (unreachable) non-invertible error.
pub(crate) fn gamma_power_pairs<F: Field>(
    gamma: F,
    count: usize,
    reason: &'static str,
) -> Result<(Vec<F>, Vec<F>), KernelError<F>> {
    let gamma_inv = gamma
        .inverse()
        .ok_or(KernelError::InvariantViolation { reason })?;
    let mut powers = Vec::with_capacity(count);
    let mut powers_inv = Vec::with_capacity(count);
    let mut power = F::one();
    let mut power_inv = F::one();
    for _ in 0..count {
        powers.push(power);
        powers_inv.push(power_inv);
        power *= gamma;
        power_inv *= gamma_inv;
    }
    Ok((powers, powers_inv))
}

/// `scale · eq(point, ·)` evaluations, big-endian (`point[0]` pairs the index
/// MSB) — the scaled variant of the reference tier's `eq_table`.
pub(crate) fn scaled_eq_table<F: Field>(point: &[F], scale: F) -> Vec<F> {
    EqPolynomial::<F>::evals(point, Some(scale))
}

/// `eq(point, ·)` evaluations, big-endian.
pub(crate) fn eq_table<F: Field>(point: &[F]) -> Vec<F> {
    EqPolynomial::<F>::evals(point, None)
}

/// The `(lo, hi)` sumcheck pair of a low-to-high-bound table at group `y`:
/// the two evaluations whose linear extension `lo + t·(hi − lo)` is the
/// table's per-round univariate restriction.
#[inline(always)]
pub(crate) fn pair<F: Field>(table: &Polynomial<F>, y: usize) -> (F, F) {
    let evals = table.evals();
    (evals[2 * y], evals[2 * y + 1])
}

/// Bind every table one round low-to-high, in place.
pub(crate) fn bind_all<'a, F: Field>(
    tables: impl IntoIterator<Item = &'a mut Polynomial<F>>,
    challenge: F,
) {
    for table in tables {
        table.bind_with_order(challenge, BindingOrder::LowToHigh);
    }
}

/// In-place low-to-high bind of a raw table:
/// `t[y] ← t[2y] + r·(t[2y+1] − t[2y])`.
pub(crate) fn bind_pairs<F: Field>(table: &mut Vec<F>, r: F) {
    let half = table.len() / 2;
    for y in 0..half {
        let even = table[2 * y];
        table[y] = even + r * (table[2 * y + 1] - even);
    }
    table.truncate(half);
}

/// Kernel-side extension of [`GruenSplitEqPolynomial`]: assemble a round
/// message from the eq-stripped inner factor's evaluations.
pub(crate) trait GruenRoundMessage<F: Field> {
    /// `s(t) = ℓ(t) · q(t)` at `t = 0, 1, …, q_evals.len() − 1`, checked
    /// against `s(0) + s(1) = previous_claim` (the reference tier's round
    /// consistency pin) and interpolated through `UnivariatePoly::from_evals`.
    ///
    /// This is the assembly half of the Gruen trick: the split-eq factor
    /// contributes only its per-round linear term `ℓ`, so kernels sample the
    /// remaining summand `q` alone and the product is restored per point —
    /// never a full-domain eq-weighted sweep.
    fn checked_round_poly(
        &self,
        q_evals: &[F],
        previous_claim: F,
        round: usize,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>>;
}

impl<F: Field> GruenRoundMessage<F> for GruenSplitEqPolynomial<F> {
    fn checked_round_poly(
        &self,
        q_evals: &[F],
        previous_claim: F,
        round: usize,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        debug_assert!(q_evals.len() >= 2);
        let (l_at_0, l_at_1) = self.current_linear_evals();
        let l_step = l_at_1 - l_at_0;
        let mut l_eval = l_at_0;
        let mut evals = Vec::with_capacity(q_evals.len());
        for q in q_evals {
            evals.push(l_eval * *q);
            l_eval += l_step;
        }

        let round_sum = evals[0] + evals[1];
        if round_sum != previous_claim {
            return Err(SumcheckError::RoundCheckFailed {
                round,
                expected: previous_claim,
                actual: round_sum,
            });
        }
        Ok(UnivariatePoly::from_evals(&evals))
    }
}

/// Assemble a round message from evaluations at `{0, 2, 3, .., degree}`,
/// recovering `s(1) = previous_claim − s(0)` — exactly the evaluation vector
/// the reference tier computes directly (its own round check pins
/// `s(0) + s(1) = previous_claim`), interpolated through the same
/// `UnivariatePoly::from_evals` path, so the coefficient vectors are
/// byte-identical on honest inputs.
pub(crate) fn round_poly_from_skipped_evals<F: Field>(
    evals_without_one: &[F],
    previous_claim: F,
) -> UnivariatePoly<F> {
    let mut evals = Vec::with_capacity(evals_without_one.len() + 1);
    evals.push(evals_without_one[0]);
    evals.push(previous_claim - evals_without_one[0]);
    evals.extend_from_slice(&evals_without_one[1..]);
    UnivariatePoly::from_evals(&evals)
}

/// Sum per-thread accumulator vectors elementwise.
#[cfg(feature = "parallel")]
pub(crate) fn merge_evals<F: Field>(mut left: Vec<F>, right: Vec<F>) -> Vec<F> {
    for (left, right) in left.iter_mut().zip(right) {
        *left += right;
    }
    left
}

// --- parallel shims --------------------------------------------------------
//
// Kernels' custom scans need chunked map-reduce and indexed maps; the serial
// fallbacks compute the same field values (sums and products of the same
// terms), so parity is unaffected by the feature.

/// `merge`-fold of `map` over index chunks of at most `chunk_size`.
pub(crate) fn map_reduce_chunks<R: Send>(
    len: usize,
    chunk_size: usize,
    map: impl Fn(Range<usize>) -> R + Send + Sync,
    merge: impl Fn(R, R) -> R + Send + Sync,
    identity: impl Fn() -> R + Send + Sync,
) -> R {
    if len == 0 {
        return identity();
    }
    #[cfg(feature = "parallel")]
    {
        let chunks = len.div_ceil(chunk_size);
        (0..chunks)
            .into_par_iter()
            .map(|c| map(c * chunk_size..((c + 1) * chunk_size).min(len)))
            .reduce(identity, merge)
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = (merge, identity, chunk_size);
        map(0..len)
    }
}

/// Collect `f(0), …, f(len − 1)`.
pub(crate) fn map_indices<T: Send>(len: usize, f: impl Fn(usize) -> T + Send + Sync) -> Vec<T> {
    #[cfg(feature = "parallel")]
    {
        (0..len).into_par_iter().map(f).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        (0..len).map(f).collect()
    }
}

/// Indexed in-place update of a slice.
pub(crate) fn for_each_index_mut<T: Send>(
    items: &mut [T],
    f: impl Fn(usize, &mut T) + Send + Sync,
) {
    #[cfg(feature = "parallel")]
    {
        items
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, item)| f(index, item));
    }
    #[cfg(not(feature = "parallel"))]
    {
        items
            .iter_mut()
            .enumerate()
            .for_each(|(index, item)| f(index, item));
    }
}

/// Pool-scaled chunk size for the chunked scans.
pub(crate) fn scan_chunk_size(len: usize) -> usize {
    #[cfg(feature = "parallel")]
    {
        len.div_ceil(rayon::current_num_threads()).max(1024)
    }
    #[cfg(not(feature = "parallel"))]
    {
        len.max(1)
    }
}

/// `LT(·, r) + constant` served from split tables and bound low-to-high
/// (legacy `LtPolynomial` port).
///
/// Big-endian index `j = j_hi ‖ j_lo` with `r = r_hi ‖ r_lo`:
/// `LT(j, r) = LT(j_hi, r_hi) + eq(j_hi, r_hi) · LT(j_lo, r_lo)`, so an
/// additive constant folds into the `~√T` hi table and low-to-high binding
/// touches only `lt_lo`; once the lo variables are exhausted the lo scalar
/// folds into `lt_hi` and binding continues densely. Values equal the dense
/// `LtPolynomial::evaluations(r)` table (plus the constant) bound identically
/// — binding acts linearly on the `j_lo` tensor factor. (jolt-poly's
/// `LtPolynomial` binds high-to-low only, so the low-to-high variant lives
/// here.)
pub(crate) enum SplitLt<F> {
    Split {
        lt_lo: Vec<F>,
        lt_hi: Vec<F>,
        eq_hi: Vec<F>,
    },
    Dense(Vec<F>),
}

#[cfg(feature = "allocative")]
impl<F> SplitLt<F> {
    pub(crate) fn heap_bytes(&self) -> usize {
        use crate::backend::vec_heap_bytes;
        match self {
            Self::Split {
                lt_lo,
                lt_hi,
                eq_hi,
            } => vec_heap_bytes(lt_lo) + vec_heap_bytes(lt_hi) + vec_heap_bytes(eq_hi),
            Self::Dense(table) => vec_heap_bytes(table),
        }
    }
}

impl<F: Field> SplitLt<F> {
    pub(crate) fn new(r_cycle: &[F]) -> Self {
        Self::new_plus_constant(r_cycle, F::zero())
    }

    /// `LT(·, r_cycle) + constant` — the constant rides in the hi table.
    pub(crate) fn new_plus_constant(r_cycle: &[F], constant: F) -> Self {
        let mid = r_cycle.len() / 2;
        let (r_hi, r_lo) = r_cycle.split_at(r_cycle.len() - mid);
        if r_lo.is_empty() {
            return Self::Dense(
                LtPolynomial::evaluations(r_hi)
                    .into_iter()
                    .map(|lt| lt + constant)
                    .collect(),
            );
        }
        Self::Split {
            lt_lo: LtPolynomial::evaluations(r_lo),
            lt_hi: LtPolynomial::evaluations(r_hi)
                .into_iter()
                .map(|lt| lt + constant)
                .collect(),
            eq_hi: EqPolynomial::<F>::evals(r_hi, None),
        }
    }

    /// `(LT[2y], LT[2y + 1])` under low-to-high pairing.
    #[inline]
    pub(crate) fn pair(&self, y: usize) -> (F, F) {
        match self {
            Self::Split {
                lt_lo,
                lt_hi,
                eq_hi,
            } => {
                let lo_len = lt_lo.len();
                let j = 2 * y;
                let hi = j / lo_len;
                let base = lt_hi[hi];
                let scale = eq_hi[hi];
                debug_assert!(lo_len >= 2, "adjacent lo indices share the hi part");
                (
                    base + scale * lt_lo[j % lo_len],
                    base + scale * lt_lo[(j + 1) % lo_len],
                )
            }
            Self::Dense(table) => (table[2 * y], table[2 * y + 1]),
        }
    }

    pub(crate) fn bind(&mut self, r: F) {
        match self {
            Self::Split {
                lt_lo,
                lt_hi,
                eq_hi,
            } => {
                let half = lt_lo.len() / 2;
                for y in 0..half {
                    let lo = lt_lo[2 * y];
                    lt_lo[y] = lo + r * (lt_lo[2 * y + 1] - lo);
                }
                lt_lo.truncate(half);
                if half == 1 {
                    // Lo variables exhausted: fold the lo scalar into the hi
                    // table and continue densely.
                    let lo_scalar = lt_lo[0];
                    let dense: Vec<F> = lt_hi
                        .iter()
                        .zip(eq_hi.iter())
                        .map(|(&lt, &eq)| lt + eq * lo_scalar)
                        .collect();
                    *self = Self::Dense(dense);
                }
            }
            Self::Dense(table) => {
                let half = table.len() / 2;
                for y in 0..half {
                    let lo = table[2 * y];
                    table[y] = lo + r * (table[2 * y + 1] - lo);
                }
                table.truncate(half);
            }
        }
    }

    pub(crate) fn final_value(&self) -> F {
        match self {
            Self::Dense(table) => {
                debug_assert_eq!(table.len(), 1);
                table[0]
            }
            Self::Split { .. } => unreachable!("split state always has lo variables to bind"),
        }
    }
}

/// Where a kernel's typed rows live: a slice-backed witness serves an
/// owning handle and every pass re-extracts its windows on the fly — the
/// materialized row vector never exists; re-emulating sources retain the
/// collected rows. The generic twin of the spartan-outer kernel's store,
/// for every carry-style typed-row consumer.
pub(crate) enum BundleStore<B> {
    Owned(jolt_witness::OwnedRows),
    Retained(Vec<B>),
}

#[cfg(feature = "allocative")]
impl<B> BundleStore<B> {
    pub(crate) fn heap_bytes(&self) -> usize {
        match self {
            Self::Owned(_) => 0,
            Self::Retained(rows) => crate::backend::vec_heap_bytes(rows),
        }
    }
}

impl<B: WitnessBundle + Copy + Send + Sync> BundleStore<B> {
    /// Resolve for a witness plane: the owning handle when the source is
    /// slice-backed (and covers the cycle domain), a materialized collect
    /// otherwise.
    pub(crate) fn resolve<F: Field>(
        witness: &dyn jolt_witness::JoltWitnessPlane<F>,
        cycles: usize,
    ) -> Result<Self, crate::KernelError<F>> {
        match witness.owned_rows() {
            Some(owned) if cycles <= owned.cycles() => Ok(Self::Owned(owned)),
            _ => Ok(Self::Retained(collect_rows(witness, cycles)?)),
        }
    }

    pub(crate) fn access(&self) -> BundleAccess<'_, B> {
        match self {
            Self::Owned(owned) => BundleAccess::View(owned.view()),
            Self::Retained(rows) => BundleAccess::Retained(rows),
        }
    }
}

/// One pass's borrowed row provider over a [`BundleStore`].
pub(crate) enum BundleAccess<'a, B> {
    View(jolt_witness::RandomAccessRows<'a>),
    Retained(&'a [B]),
}

impl<B: WitnessBundle + Copy> BundleAccess<'_, B> {
    /// The typed row at cycle `t` — an extraction window over a slice-backed
    /// source, an indexed copy from a retained vector. Pure per index.
    #[inline]
    pub(crate) fn row(&self, t: usize) -> Result<B, WitnessError> {
        match self {
            Self::View(view) => view.window(t),
            Self::Retained(rows) => Ok(rows[t]),
        }
    }
}
