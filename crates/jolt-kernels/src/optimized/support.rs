//! Shared machinery of the optimized kernels: the split-eq (Gruen) round
//! driver, round/typed-row bookkeeping, deferred-reduction accumulator
//! helpers, and the cfg(parallel) fold shims. One home per idiom — kernels
//! hold the summand math, this module holds the plumbing they all repeat.

use std::ops::Range;

use jolt_claims::protocols::jolt::JoltDerivedId;
use jolt_field::{Accumulator, JoltField};
use jolt_poly::{
    BindingOrder, EqPolynomial, GruenSplitEqPolynomial, LtPolynomial, Polynomial, UnivariatePoly,
};
use jolt_sumcheck::SumcheckError;
#[cfg(feature = "parallel")]
use jolt_utils::par_collect_windows;
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, ConcreteSumcheckChallenges, SumcheckInputPoints, SumcheckOutputPoints,
};
use jolt_verifier::VerifierError;
use jolt_witness::{
    stream_witnesses, RandomAccessRows, RowSource, StreamConsumer, WitnessBundle, WitnessError,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::{KernelError, SumcheckKernelError};

/// A kernel's bound-round count against its total — the one home of the
/// "claims only after every round is bound" invariant.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
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
    pub(crate) fn require_complete<F: JoltField>(&self) -> Result<(), SumcheckKernelError<F>> {
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

pub(crate) fn collect_par_map<B: WitnessBundle, V: Copy + Send>(
    access: &RandomAccessRows,
    cycles: usize,
    pack: impl Fn(B) -> V + Send + Sync,
) -> Result<Vec<V>, WitnessError> {
    let window = |index| access.window::<B>(index).map(&pack);
    #[cfg(feature = "parallel")]
    return par_collect_windows(cycles, window);
    #[cfg(not(feature = "parallel"))]
    (0..cycles).map(window).collect()
}

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
            return collect_par_map(&access, cycles, |bundle: B| bundle);
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
pub(crate) fn accumulate_product<F: JoltField>(factors: &[F], lane: &mut F::Accumulator) {
    debug_assert!(factors.len() >= 2);
    let last = factors.len() - 1;
    let mut product = factors[0];
    for factor in &factors[1..last] {
        product *= *factor;
    }
    lane.fmadd(product, factors[last]);
}

/// Walk one row's product grid: with `evals` seeded at the `t = 1` factor
/// values and `steps` their per-factor linear steps, accumulate the factor
/// product `Π evals` into `lanes[t − 1]` for `t = 1, …, n − 1` (advancing
/// every factor by its step between points) and the leading coefficient
/// `Π steps` into `lanes[n − 1]`, where `n = lanes.len()`.
#[inline]
pub(crate) fn accumulate_product_grid<F: JoltField>(
    evals: &mut [F],
    steps: &[F],
    lanes: &mut [F::Accumulator],
) {
    let n = lanes.len();
    accumulate_product(evals, &mut lanes[0]);
    for lane in &mut lanes[1..n - 1] {
        for (eval, step) in evals.iter_mut().zip(steps) {
            *eval += *step;
        }
        accumulate_product(evals, lane);
    }
    accumulate_product(steps, &mut lanes[n - 1]);
}

/// Accumulate `eq · F(value)` for a full-range `u64` on the small-scalar
/// accumulator without overflowing it: the accumulator's headroom is one
/// extra limb, which ~4 full-magnitude `field × u64` products exhaust, so the
/// value is split into u32 halves (products ≤ 2^286, headroom ≥ 2^34 terms).
/// `eq_shifted` must be `eq · 2^32`; the two fused adds sum to exactly
/// `eq · F(value)`.
#[inline]
pub(crate) fn fmadd_u64_split<F: JoltField>(
    accumulator: &mut F::SmallScalarAccumulator,
    eq: F,
    eq_shifted: F,
    value: u64,
) {
    accumulator.fmadd_u64(eq_shifted, value >> 32);
    accumulator.fmadd_u64(eq, value & 0xFFFF_FFFF);
}

/// `[1, γ, γ², …, γ^{N−1}]`.
pub(crate) fn gamma_powers_array<F: JoltField, const N: usize>(gamma: F) -> [F; N] {
    let mut powers = [F::one(); N];
    for i in 1..N {
        powers[i] = powers[i - 1] * gamma;
    }
    powers
}

/// `[1, γ, γ², …]` of length `count`.
pub(crate) fn gamma_powers<F: JoltField>(gamma: F, count: usize) -> Vec<F> {
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
pub(crate) fn gamma_power_pairs<F: JoltField>(
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
pub(crate) fn scaled_eq_table<F: JoltField>(point: &[F], scale: F) -> Vec<F> {
    EqPolynomial::<F>::evals(point, Some(scale))
}

/// `eq(point, ·)` evaluations, big-endian.
pub(crate) fn eq_table<F: JoltField>(point: &[F]) -> Vec<F> {
    EqPolynomial::<F>::evals(point, None)
}

/// The `(lo, hi)` sumcheck pair of a low-to-high-bound table at group `y`:
/// the two evaluations whose linear extension `lo + t·(hi − lo)` is the
/// table's per-round univariate restriction.
#[inline(always)]
pub(crate) fn pair<F: JoltField>(table: &Polynomial<F>, y: usize) -> (F, F) {
    let evals = table.evals();
    (evals[2 * y], evals[2 * y + 1])
}

/// Bind every table one round low-to-high, in place.
pub(crate) fn bind_all<'a, F: JoltField>(
    tables: impl IntoIterator<Item = &'a mut Polynomial<F>>,
    challenge: F,
) {
    for table in tables {
        table.bind_with_order(challenge, BindingOrder::LowToHigh);
    }
}

/// In-place low-to-high bind of a raw table:
/// `t[y] ← t[2y] + r·(t[2y+1] − t[2y])`.
pub(crate) fn bind_pairs<F: JoltField>(table: &mut Vec<F>, r: F) {
    let half = table.len() / 2;
    for y in 0..half {
        let even = table[2 * y];
        table[y] = even + r * (table[2 * y + 1] - even);
    }
    table.truncate(half);
}

/// The drawn challenges of a kernel's bound rounds, tracked against the
/// round total — one authority for both the challenge history and the
/// bound-rounds invariant. Kernels that never revisit their challenges use
/// [`RoundProgress`] instead.
#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F")
)]
pub(crate) struct RoundChallenges<F> {
    #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
    challenges: Vec<F>,
    total: usize,
}

impl<F: JoltField> RoundChallenges<F> {
    pub(crate) fn new(total: usize) -> Self {
        Self {
            challenges: Vec::with_capacity(total),
            total,
        }
    }

    /// Total rounds — the kernel's `ProveRounds::num_rounds`.
    pub(crate) fn total(&self) -> usize {
        self.total
    }

    /// Rounds bound so far.
    pub(crate) fn bound(&self) -> usize {
        self.challenges.len()
    }

    /// Record one bound round's challenge.
    pub(crate) fn push(&mut self, challenge: F) {
        self.challenges.push(challenge);
    }

    /// The challenges bound so far, in binding order.
    pub(crate) fn as_slice(&self) -> &[F] {
        &self.challenges
    }

    /// Gate for every output-claim / derived-table entry point.
    pub(crate) fn require_complete(&self) -> Result<(), SumcheckKernelError<F>> {
        if self.challenges.len() == self.total {
            Ok(())
        } else {
            Err(SumcheckKernelError::NotFullyBound {
                remaining: self.total - self.challenges.len(),
            })
        }
    }
}

/// Pin a kernel-maintained derived value (typically its fully bound split-eq
/// scalar) against the verifier's own `derive_output_term` — the optimized
/// tier's drift detector for tables it never materializes, mirroring the
/// naive tier's check on its hand-materialized derived tables.
pub(crate) fn pin_derived_term<F: JoltField, R: ConcreteSumcheck<F>>(
    relation: &R,
    id: JoltDerivedId,
    input_points: &SumcheckInputPoints<F, R>,
    output_points: &SumcheckOutputPoints<F, R>,
    challenges: &ConcreteSumcheckChallenges<F, R>,
    got: F,
) -> Result<(), SumcheckKernelError<F>> {
    let expected = relation.derive_output_term(&id, input_points, output_points, challenges)?;
    if got != expected {
        return Err(SumcheckKernelError::DerivedTableDrift { id, expected, got });
    }
    Ok(())
}

/// [`pin_derived_term`], passing vacuously when the relation does not derive
/// the term under this proof shape (`MissingStageClaimDerived`).
pub(crate) fn pin_derived_term_if_derived<F: JoltField, R: ConcreteSumcheck<F>>(
    relation: &R,
    id: JoltDerivedId,
    input_points: &SumcheckInputPoints<F, R>,
    output_points: &SumcheckOutputPoints<F, R>,
    challenges: &ConcreteSumcheckChallenges<F, R>,
    got: F,
) -> Result<(), SumcheckKernelError<F>> {
    match relation.derive_output_term(&id, input_points, output_points, challenges) {
        Ok(expected) if got != expected => {
            Err(SumcheckKernelError::DerivedTableDrift { id, expected, got })
        }
        Ok(_) | Err(VerifierError::MissingStageClaimDerived { .. }) => Ok(()),
        Err(error) => Err(error.into()),
    }
}

/// Kernel-side extension of [`GruenSplitEqPolynomial`]: assemble a round
/// message from the eq-stripped inner factor's evaluations.
pub(crate) trait GruenRoundMessage<F: JoltField> {
    /// `s(t) = ℓ(t) · q(t)` at `t = 0, 1, …, q_evals.len() − 1`, checked
    /// against `s(0) + s(1) = previous_claim` (the reference tier's round
    /// consistency pin) and interpolated through `UnivariatePoly::from_evals`.
    /// `q_evals` is scaled into the `s` evaluations in place.
    ///
    /// This is the assembly half of the Gruen trick: the split-eq factor
    /// contributes only its per-round linear term `ℓ`, so kernels sample the
    /// remaining summand `q` alone and the product is restored per point —
    /// never a full-domain eq-weighted sweep.
    fn checked_round_poly(
        &self,
        q_evals: &mut [F],
        previous_claim: F,
        round: usize,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>>;

    /// `(q(0), q(∞))` of the two-table product summand
    /// `Σ_y E(y) · a(y) · b(y)` over the remaining low-to-high `(lo, hi)`
    /// pairs — the endpoints `gruen_poly_deg_3` completes into the cubic
    /// round message with the running claim.
    fn product_endpoints(&self, a: &Polynomial<F>, b: &Polynomial<F>) -> (F, F);
}

impl<F: JoltField> GruenRoundMessage<F> for GruenSplitEqPolynomial<F> {
    fn checked_round_poly(
        &self,
        q_evals: &mut [F],
        previous_claim: F,
        round: usize,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        debug_assert!(q_evals.len() >= 2);
        let (l_at_0, l_at_1) = self.current_linear_evals();
        let l_step = l_at_1 - l_at_0;
        let mut l_eval = l_at_0;
        for q in q_evals.iter_mut() {
            *q *= l_eval;
            l_eval += l_step;
        }

        let round_sum = q_evals[0] + q_evals[1];
        if round_sum != previous_claim {
            return Err(SumcheckError::RoundCheckFailed {
                round,
                expected: previous_claim,
                actual: round_sum,
            });
        }
        Ok(UnivariatePoly::from_evals(q_evals))
    }

    fn product_endpoints(&self, a: &Polynomial<F>, b: &Polynomial<F>) -> (F, F) {
        let a = a.evals();
        let b = b.evals();
        debug_assert_eq!(
            self.e_out_current_len() * self.e_in_current_len() * 2,
            a.len()
        );
        debug_assert_eq!(a.len(), b.len());
        let [zero, infinity] = self.par_fold_out_in(
            || [F::zero(); 2],
            |accumulator, row, _x_in, e| {
                let (a_low, a_high) = (a[2 * row], a[2 * row + 1]);
                let (b_low, b_high) = (b[2 * row], b[2 * row + 1]);
                accumulator[0] += e * (a_low * b_low);
                accumulator[1] += e * ((a_high - a_low) * (b_high - b_low));
            },
            |_x_out, e_out, accumulator| [e_out * accumulator[0], e_out * accumulator[1]],
            |left, right| [left[0] + right[0], left[1] + right[1]],
        );
        (zero, infinity)
    }
}

/// Assemble a round message from evaluations at `{0, 2, 3, .., degree}`,
/// recovering `s(1) = previous_claim − s(0)` — exactly the evaluation vector
/// the reference tier computes directly (its own round check pins
/// `s(0) + s(1) = previous_claim`), interpolated through the same
/// `UnivariatePoly::from_evals` path, so the coefficient vectors are
/// byte-identical on honest inputs.
pub(crate) fn round_poly_from_skipped_evals<F: JoltField>(
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
pub(crate) fn merge_evals<F: JoltField>(mut left: Vec<F>, right: Vec<F>) -> Vec<F> {
    for (left, right) in left.iter_mut().zip(right) {
        *left += right;
    }
    left
}

/// `s(t)` samples at `t ∈ {0, 2, 3}` of the cubic triple-product summand
/// `Σ_y a(y) · b(y) · c(y)` over the remaining low-to-high `(lo, hi)` pairs,
/// through the deferred-reduction accumulator; `s(1)` comes from the engine's
/// `from_evals_and_hint` recovery.
pub(crate) fn triple_product_round_evals<F: JoltField>(
    half: usize,
    a: impl Fn(usize) -> (F, F) + Send + Sync,
    b: impl Fn(usize) -> (F, F) + Send + Sync,
    c: impl Fn(usize) -> (F, F) + Send + Sync,
) -> [F; 3] {
    let accumulate = |y: usize, acc: &mut [F::Accumulator; 3]| {
        let (a_0, a_1) = a(y);
        let (b_0, b_1) = b(y);
        let (c_0, c_1) = c(y);
        let (a_m, b_m, c_m) = (a_1 - a_0, b_1 - b_0, c_1 - c_0);
        let (a_2, b_2, c_2) = (a_1 + a_m, b_1 + b_m, c_1 + c_m);
        acc[0].fmadd(a_0 * b_0, c_0);
        acc[1].fmadd(a_2 * b_2, c_2);
        acc[2].fmadd((a_2 + a_m) * (b_2 + b_m), c_2 + c_m);
    };

    #[cfg(feature = "parallel")]
    {
        (0..half)
            .into_par_iter()
            .fold(
                || [F::Accumulator::default(); 3],
                |mut acc, y| {
                    accumulate(y, &mut acc);
                    acc
                },
            )
            .map(|acc| acc.map(F::Accumulator::reduce))
            .reduce(
                || [F::zero(); 3],
                |a, b| [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
            )
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut acc = [F::Accumulator::default(); 3];
        for y in 0..half {
            accumulate(y, &mut acc);
        }
        acc.map(F::Accumulator::reduce)
    }
}

/// Sum per-pair-group evaluation contributions over `y = 0..groups` into a
/// `slots`-sized vector — the dense-table round walk of the pair-group
/// kernels ([`pair`] serves the `(lo, hi)` values inside `accumulate`).
pub(crate) fn par_sum_pair_groups<F: JoltField>(
    groups: usize,
    slots: usize,
    accumulate: impl Fn(&mut [F], usize) + Send + Sync,
) -> Vec<F> {
    par_sum_pair_groups_reusing(groups, slots, || (), |acc, (), y| accumulate(acc, y))
}

/// [`par_sum_pair_groups`] with a per-thread scratch buffer, for kernels
/// whose group walk reuses an allocation across groups (the scratch is
/// fully overwritten per group).
pub(crate) fn par_sum_pair_groups_reusing<F: JoltField, S: Send>(
    groups: usize,
    slots: usize,
    scratch: impl Fn() -> S + Send + Sync,
    accumulate: impl Fn(&mut [F], &mut S, usize) + Send + Sync,
) -> Vec<F> {
    #[cfg(feature = "parallel")]
    {
        (0..groups)
            .into_par_iter()
            .fold(
                || (vec![F::zero(); slots], scratch()),
                |(mut acc, mut scratch), y| {
                    accumulate(&mut acc, &mut scratch, y);
                    (acc, scratch)
                },
            )
            .map(|(acc, _)| acc)
            .reduce(|| vec![F::zero(); slots], merge_evals)
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut acc = vec![F::zero(); slots];
        let mut scratch = scratch();
        for y in 0..groups {
            accumulate(&mut acc, &mut scratch, y);
        }
        acc
    }
}

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
#[derive(Clone)]
#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F")
)]
pub(crate) enum SplitLt<F> {
    Split {
        #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
        lt_lo: Vec<F>,
        #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
        lt_hi: Vec<F>,
        #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
        eq_hi: Vec<F>,
    },
    Dense(#[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))] Vec<F>),
}

impl<F: JoltField> SplitLt<F> {
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
                bind_pairs(lt_lo, r);
                if lt_lo.len() == 1 {
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
            Self::Dense(table) => bind_pairs(table, r),
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
