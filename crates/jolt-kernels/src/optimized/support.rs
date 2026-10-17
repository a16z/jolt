use std::marker::PhantomData;
use std::ops::Range;

use jolt_field::{Accumulator, JoltField};
use jolt_poly::{
    BindingOrder, EqPolynomial, GruenSplitEqPolynomial, LtPolynomial, Polynomial, UnivariatePoly,
};
use jolt_sumcheck::SumcheckError;
#[cfg(feature = "parallel")]
use jolt_utils::par_collect_windows;
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, ConcreteSumcheckChallenges, DerivedIdOf, SumcheckInputPoints,
    SumcheckOutputPoints,
};
use jolt_verifier::VerifierError;
use jolt_witness::{
    stream_witnesses, JoltWitnessPlane, RowSource, StreamConsumer, TraceRows, WitnessBundle,
    WitnessError,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::SumcheckKernelError;

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

    pub(crate) fn bound(&self) -> usize {
        self.bound
    }

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
    access: &TraceRows,
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
    if let Some(access) = source.rows() {
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

/// Largest factor count [`accumulate_product_grid`] accepts; `n!` must fit a
/// `u64` for the finite-difference extension. Kernels whose factor count
/// follows the one-hot chunk widths reject a larger one at construction.
pub(crate) const MAX_GRID_FACTORS: usize = 20;

/// Scratch length [`accumulate_product_grid`] needs for `n` factors: two
/// child tables per tree level plus one difference table.
pub(crate) const fn product_grid_scratch_len(n: usize) -> usize {
    6 * n + 2
}

/// Largest factor count whose grid is cheaper to walk directly (every
/// factor multiplied at every point) than through the product tree.
const DIRECT_GRID_FACTORS: usize = 4;

/// Accumulates `Π factors` into `lane`, fusing the last multiply into the
/// deferred-reduction accumulator. Requires at least two factors.
#[inline]
fn accumulate_product<F: JoltField>(factors: &[F], lane: &mut F::Accumulator) {
    let last = factors.len() - 1;
    let mut product = factors[0];
    for factor in &factors[1..last] {
        product *= *factor;
    }
    lane.fmadd(product, factors[last]);
}

/// Walk one row's product grid: with `evals` the `t = 1` values of `n`
/// linear factors and `steps` their slopes, accumulate their product at
/// `t = 1, …, n − 1` into `lanes[t − 1]` and its leading coefficient
/// `Π steps` into `lanes[n − 1]`, where `n = lanes.len() ≥ 2`.
///
/// Above [`DIRECT_GRID_FACTORS`] the product is built as a balanced tree:
/// each node multiplies its two halves pointwise at exactly as many points
/// as its degree, after extending both halves to those points by finite
/// differences (additions only). That costs `O(n log n)` multiplications
/// instead of the `n · (n − 2)` of multiplying every factor at every point;
/// the root's multiplications fuse into the deferred-reduction lanes.
/// Four and eight factors take unrolled forms of the same tree that keep
/// every table on the stack.
/// `scratch` must hold [`product_grid_scratch_len`]`(n)` elements.
#[inline]
pub(crate) fn accumulate_product_grid<F: JoltField>(
    evals: &[F],
    steps: &[F],
    lanes: &mut [F::Accumulator],
    scratch: &mut [F],
) {
    let n = lanes.len();
    debug_assert!((2..=MAX_GRID_FACTORS).contains(&n));
    debug_assert_eq!(evals.len(), n);
    debug_assert_eq!(steps.len(), n);
    match (evals, steps, lanes) {
        ([a, b, c, d], [da, db, dc, dd], lanes @ [_, _, _, _]) => {
            let left = quadratic_product_samples([*a, *b], [*da, *db]);
            let right = quadratic_product_samples([*c, *d], [*dc, *dd]);
            lanes[0].fmadd(left[0], right[0]);
            lanes[1].fmadd(left[1], right[1]);
            lanes[2].fmadd(quadratic_next(left), quadratic_next(right));
            lanes[3].fmadd(left[2], right[2]);
        }
        (
            [a, b, c, d, e, f, g, h],
            [da, db, dc, dd, de, df, dg, dh],
            lanes @ [_, _, _, _, _, _, _, _],
        ) => {
            let left = quartic_product_samples([*a, *b, *c, *d], [*da, *db, *dc, *dd]);
            let right = quartic_product_samples([*e, *f, *g, *h], [*de, *df, *dg, *dh]);
            let mut left_window = [left[0], left[1], left[2], left[3]];
            let mut right_window = [right[0], right[1], right[2], right[3]];
            for ((lane, left), right) in lanes.iter_mut().zip(left_window).zip(right_window) {
                lane.fmadd(left, right);
            }
            let left_six = left[4].mul_u64(6);
            let right_six = right[4].mul_u64(6);
            for lane in &mut lanes[4..7] {
                let left_next = quartic_next(left_window, left_six);
                let right_next = quartic_next(right_window, right_six);
                lane.fmadd(left_next, right_next);
                left_window = [left_window[1], left_window[2], left_window[3], left_next];
                right_window = [
                    right_window[1],
                    right_window[2],
                    right_window[3],
                    right_next,
                ];
            }
            lanes[7].fmadd(left[4], right[4]);
        }
        (evals, steps, lanes) if n <= DIRECT_GRID_FACTORS => {
            let point = &mut scratch[..n];
            point.copy_from_slice(evals);
            accumulate_product(point, &mut lanes[0]);
            for lane in &mut lanes[1..n - 1] {
                for (value, step) in point.iter_mut().zip(steps) {
                    *value += *step;
                }
                accumulate_product(point, lane);
            }
            accumulate_product(steps, &mut lanes[n - 1]);
        }
        (evals, steps, lanes) => {
            let mid = n / 2;
            let (left, rest) = scratch.split_at_mut(n - 1);
            let (right, rest) = rest.split_at_mut(n - 1);
            let left_lead = linear_product(&evals[..mid], &steps[..mid], &mut left[..mid], rest);
            let right_lead =
                linear_product(&evals[mid..], &steps[mid..], &mut right[..n - mid], rest);
            extend_by_differences(left, mid, left_lead, rest);
            extend_by_differences(right, n - mid, right_lead, rest);
            for ((lane, left), right) in lanes[..n - 1].iter_mut().zip(&*left).zip(&*right) {
                lane.fmadd(*left, *right);
            }
            lanes[n - 1].fmadd(left_lead, right_lead);
        }
    }
}

#[inline]
fn quadratic_product_samples<F: JoltField>(values: [F; 2], steps: [F; 2]) -> [F; 3] {
    [
        values[0] * values[1],
        (values[0] + steps[0]) * (values[1] + steps[1]),
        steps[0] * steps[1],
    ]
}

#[inline]
fn quadratic_next<F: JoltField>(samples: [F; 3]) -> F {
    let next = samples[1] + samples[2];
    next + next - samples[0]
}

#[inline]
fn quartic_product_samples<F: JoltField>(values: [F; 4], steps: [F; 4]) -> [F; 5] {
    let left = quadratic_product_samples([values[0], values[1]], [steps[0], steps[1]]);
    let right = quadratic_product_samples([values[2], values[3]], [steps[2], steps[3]]);
    let left_third = quadratic_next(left);
    let right_third = quadratic_next(right);
    let left_next = left_third + left[2];
    let right_next = right_third + right[2];
    [
        left[0] * right[0],
        left[1] * right[1],
        left_third * right_third,
        (left_next + left_next - left[1]) * (right_next + right_next - right[1]),
        left[2] * right[2],
    ]
}

#[inline]
fn quartic_next<F: JoltField>(window: [F; 4], six_leading: F) -> F {
    // The fourth finite difference is 24 times the leading coefficient.
    let mut next = six_leading + window[3] - window[2] + window[1];
    next = next + next - window[2];
    next + next - window[0]
}

/// The product of the linear factors `values[i] + (t − 1) · steps[i]` at
/// `t = 1, …, degree` into `out` (`degree = values.len() = out.len()`);
/// returns its leading coefficient.
fn linear_product<F: JoltField>(values: &[F], steps: &[F], out: &mut [F], scratch: &mut [F]) -> F {
    let degree = values.len();
    if degree == 1 {
        out[0] = values[0];
        return steps[0];
    }
    let mid = degree / 2;
    let (left, rest) = scratch.split_at_mut(degree);
    let (right, rest) = rest.split_at_mut(degree);
    let left_lead = linear_product(&values[..mid], &steps[..mid], &mut left[..mid], rest);
    let right_lead = linear_product(
        &values[mid..],
        &steps[mid..],
        &mut right[..degree - mid],
        rest,
    );
    extend_by_differences(left, mid, left_lead, rest);
    extend_by_differences(right, degree - mid, right_lead, rest);
    for ((out, left), right) in out.iter_mut().zip(&*left).zip(&*right) {
        *out = *left * *right;
    }
    left_lead * right_lead
}

/// Extends a degree-`degree` polynomial known at `t = 1, …, degree`
/// (`values[..degree]`) with leading coefficient `lead` to every
/// `t ≤ values.len()`: the `degree`-th difference is the constant
/// `degree! · lead`, so each new point costs `degree` additions after one
/// backward-difference table at `t = degree`.
fn extend_by_differences<F: JoltField>(
    values: &mut [F],
    degree: usize,
    lead: F,
    scratch: &mut [F],
) {
    const FACTORIALS: [u64; MAX_GRID_FACTORS + 1] = {
        let mut table = [1u64; MAX_GRID_FACTORS + 1];
        let mut i = 1;
        while i <= MAX_GRID_FACTORS {
            table[i] = table[i - 1] * i as u64;
            i += 1;
        }
        table
    };
    if values.len() <= degree {
        return;
    }
    let (known, unknown) = values.split_at_mut(degree);
    if degree == 1 {
        let mut value = known[0];
        for next in unknown {
            value += lead;
            *next = value;
        }
        return;
    }
    // `back[k] = ∇^k p(degree)`: the last entry of each difference order.
    let (work, back) = scratch.split_at_mut(degree);
    let back = &mut back[..degree];
    work.copy_from_slice(known);
    back[0] = work[degree - 1];
    for (order, last) in back.iter_mut().enumerate().skip(1) {
        for i in (order..degree).rev() {
            work[i] -= work[i - 1];
        }
        *last = work[degree - 1];
    }
    let top = if degree == 2 {
        lead + lead
    } else {
        lead.mul_u64(FACTORIALS[degree])
    };
    for next in unknown {
        let mut higher = top;
        for difference in back.iter_mut().rev() {
            *difference += higher;
            higher = *difference;
        }
        *next = higher;
    }
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
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(crate) struct RoundChallenges<F> {
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

    pub(crate) fn bound(&self) -> usize {
        self.challenges.len()
    }

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
    id: DerivedIdOf<F, R>,
    input_points: &SumcheckInputPoints<F, R>,
    output_points: &SumcheckOutputPoints<F, R>,
    challenges: &ConcreteSumcheckChallenges<F, R>,
    got: F,
) -> Result<(), SumcheckKernelError<F>> {
    let expected = relation.derive_output_term(&id, input_points, output_points, challenges)?;
    if got != expected {
        return Err(SumcheckKernelError::DerivedTableDrift {
            id: id.into(),
            expected,
            got,
        });
    }
    Ok(())
}

/// [`pin_derived_term`], passing vacuously when the relation does not derive
/// the term under this proof shape (`MissingStageClaimDerived`).
pub(crate) fn pin_derived_term_if_derived<F: JoltField, R: ConcreteSumcheck<F>>(
    relation: &R,
    id: DerivedIdOf<F, R>,
    input_points: &SumcheckInputPoints<F, R>,
    output_points: &SumcheckOutputPoints<F, R>,
    challenges: &ConcreteSumcheckChallenges<F, R>,
    got: F,
) -> Result<(), SumcheckKernelError<F>> {
    match relation.derive_output_term(&id, input_points, output_points, challenges) {
        Ok(expected) if got != expected => Err(SumcheckKernelError::DerivedTableDrift {
            id: id.into(),
            expected,
            got,
        }),
        Ok(_) | Err(VerifierError::MissingStageClaimDerived { .. }) => Ok(()),
        Err(error) => Err(error.into()),
    }
}

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

    fn checked_cubic(
        &self,
        q_zero: F,
        q_leading: F,
        previous_claim: F,
        round: usize,
        q_at_one: impl FnOnce() -> F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>>;

    fn checked_toom(
        &self,
        q_evals: &[F],
        previous_claim: F,
        round: usize,
        q_at_zero: impl FnOnce() -> F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>>;

    #[cfg(feature = "field-inline")]
    fn checked_linear(
        &self,
        q_one: F,
        previous_claim: F,
        round: usize,
        q_at_zero: impl FnOnce() -> F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>>;

    fn product_at_one(&self, a: &Polynomial<F>, b: &Polynomial<F>) -> F;

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

    fn checked_cubic(
        &self,
        q_zero: F,
        q_leading: F,
        previous_claim: F,
        round: usize,
        q_at_one: impl FnOnce() -> F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        self.gruen_poly_deg_3(q_zero, q_leading, previous_claim, q_at_one)
            .map_err(|actual| SumcheckError::RoundCheckFailed {
                round,
                expected: previous_claim,
                actual,
            })
    }

    fn checked_toom(
        &self,
        q_evals: &[F],
        previous_claim: F,
        round: usize,
        q_at_zero: impl FnOnce() -> F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        self.gruen_poly_from_evals(q_evals, previous_claim, q_at_zero)
            .map_err(|actual| SumcheckError::RoundCheckFailed {
                round,
                expected: previous_claim,
                actual,
            })
    }

    #[cfg(feature = "field-inline")]
    fn checked_linear(
        &self,
        q_one: F,
        previous_claim: F,
        round: usize,
        q_at_zero: impl FnOnce() -> F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        self.gruen_poly_deg_2(q_one, previous_claim, q_at_zero)
            .map_err(|actual| SumcheckError::RoundCheckFailed {
                round,
                expected: previous_claim,
                actual,
            })
    }

    fn product_at_one(&self, a: &Polynomial<F>, b: &Polynomial<F>) -> F {
        self.par_fold_out_in(
            F::zero,
            |sum, row, _, weight| {
                *sum += weight * a.evals()[2 * row + 1] * b.evals()[2 * row + 1];
            },
            |_, weight, sum| weight * sum,
            |left, right| left + right,
        )
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

/// Sum `task(0), …, task(tasks − 1)` elementwise (each yields a `len`-sized
/// vector), failing fast on the first error.
pub(crate) fn try_par_sum_vecs<F: JoltField, E: Send>(
    tasks: usize,
    len: usize,
    task: impl Fn(usize) -> Result<Vec<F>, E> + Send + Sync,
) -> Result<Vec<F>, E> {
    let merge = |mut left: Vec<F>, right: Vec<F>| {
        for (left, right) in left.iter_mut().zip(right) {
            *left += right;
        }
        left
    };
    #[cfg(feature = "parallel")]
    {
        (0..tasks).into_par_iter().map(task).try_reduce(
            || vec![F::zero(); len],
            |left, right| Ok(merge(left, right)),
        )
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut folded = vec![F::zero(); len];
        for index in 0..tasks {
            folded = merge(folded, task(index)?);
        }
        Ok(folded)
    }
}

/// Rows per callback of a task pass's visit.
pub(crate) const VISIT_ROWS: usize = 1 << 12;

/// The rows a streamed kernel's task passes visit: the witness's row handle,
/// or, in in-module parity tests, typed rows that no trace produces
/// (full-range words, `i128::MIN` operands).
pub(crate) enum KernelRows<B> {
    Trace(TraceRows, PhantomData<fn() -> B>),
    #[cfg(test)]
    Synthetic(Vec<B>),
}

impl<B: WitnessBundle> KernelRows<B> {
    /// The witness's row handle, which must span exactly the `2^log_t`-cycle
    /// domain; a witness without one fails instead of collecting rows.
    pub(crate) fn from_witness<F: JoltField>(
        witness: &dyn JoltWitnessPlane<F>,
        log_t: usize,
        label: &'static str,
    ) -> Result<Self, WitnessError> {
        let rows = witness
            .rows()
            .ok_or(WitnessError::UnavailableView { label })?;
        if rows.cycles() != 1 << log_t {
            return Err(WitnessError::InvalidDimensions {
                label,
                reason: format!("{} rows for a 2^{log_t}-cycle stage", rows.cycles()),
            });
        }
        Ok(Self::Trace(rows, PhantomData))
    }

    /// One serial visit of `range` in exact `chunk`-row callbacks
    /// ([`TraceRows::visit_bundles`]); `visit` must not enter Rayon.
    pub(crate) fn visit(
        &self,
        range: Range<usize>,
        chunk: usize,
        visit: impl FnMut(usize, &[B]) -> Result<(), WitnessError>,
    ) -> Result<(), WitnessError> {
        match self {
            Self::Trace(rows, _) => rows.visit_bundles(range, chunk, visit),
            #[cfg(test)]
            Self::Synthetic(rows) => {
                let mut visit = visit;
                range
                    .clone()
                    .step_by(chunk)
                    .try_for_each(|start| visit(start, &rows[start..range.end.min(start + chunk)]))
            }
        }
    }
}

/// Tasks per pool thread in a [`task_ranges`] partition, so work stealing can
/// even out unequal row costs between tasks.
#[cfg(feature = "parallel")]
const TASKS_PER_THREAD: usize = 8;

/// The task partition of one pass over the cycle domain `0..cycles`: units of
/// `U = min(cycles, group)` cycles and `min(TASKS_PER_THREAD · pool threads,
/// cycles / U)` tasks, task `i` holding units `⌊i·n/tasks⌋..⌊(i+1)·n/tasks⌋`
/// of the `n = cycles / U`. Both sizes are powers of two, so every task
/// covers whole `group`-cycle kernel groups. Per-task state allocated before
/// the pass is held `tasks` times at once, up to `TASKS_PER_THREAD` per pool
/// thread.
pub(crate) fn task_ranges(cycles: usize, group: usize) -> Vec<Range<usize>> {
    let unit = cycles.min(group);
    let units = cycles / unit;
    #[cfg(feature = "parallel")]
    let tasks = (TASKS_PER_THREAD * rayon::current_num_threads()).min(units);
    #[cfg(not(feature = "parallel"))]
    let tasks = 1;
    (0..tasks)
        .map(|task| task * units / tasks * unit..(task + 1) * units / tasks * unit)
        .collect()
}

/// Splits every table at the task boundaries of `ranges`, where `entry(cycle)`
/// is the table index at which a boundary cycle's entries start, so each task
/// fills only its own slices of the final tables.
pub(crate) fn split_at_tasks<'a, T, const N: usize>(
    mut tables: [&'a mut [T]; N],
    ranges: &[Range<usize>],
    entry: impl Fn(usize) -> usize,
) -> Vec<[&'a mut [T]; N]> {
    ranges
        .iter()
        .map(|range| {
            let len = entry(range.end) - entry(range.start);
            tables.each_mut().map(|table| {
                let (head, tail) = std::mem::take(table).split_at_mut(len);
                *table = tail;
                head
            })
        })
        .collect()
}

/// The round whose pass materializes a streamed-prefix kernel's dense tables:
/// the first bind depth `k` at which the tables, `table_bytes` before any
/// bind, fit `cap` bytes (`⌈table_bytes / 2^k⌉ ≤ cap`), or `rounds` when even
/// the last round's tables exceed it and every round streams.
pub(crate) fn dense_crossover(table_bytes: usize, cap: usize, rounds: usize) -> usize {
    (0..rounds)
        .find(|&depth| table_bytes.div_ceil(1 << depth) <= cap)
        .unwrap_or(rounds)
}

/// Contracts consecutive rows into the halves of `2^(log_half + 1)`-row
/// pairs: row `t` adds its `M` values to half `(t >> log_half) & 1` of pair
/// `t >> (log_half + 1)`. A walk adds every row of its range in order, so a
/// pair's last row completes it; halves persist across visit callbacks.
pub(crate) struct PairHalves<F, const M: usize> {
    log_half: usize,
    halves: [[F; 2]; M],
}

impl<F: JoltField, const M: usize> PairHalves<F, M> {
    pub(crate) fn new(log_half: usize) -> Self {
        Self {
            log_half,
            halves: [[F::zero(); 2]; M],
        }
    }

    /// Row `t`'s offset within its half: the index of its row weights.
    pub(crate) fn offset(&self, t: usize) -> usize {
        t & ((1 << self.log_half) - 1)
    }

    /// Adds row `t`'s values; the pair's last row returns the pair index and
    /// each value's `[low, high]` halves.
    pub(crate) fn add(&mut self, t: usize, values: [F; M]) -> Option<(usize, [[F; 2]; M])> {
        let half = (t >> self.log_half) & 1;
        for (halves, value) in self.halves.iter_mut().zip(values) {
            halves[half] += value;
        }
        let last = (2 << self.log_half) - 1;
        if t & last != last {
            return None;
        }
        Some((
            t >> (self.log_half + 1),
            std::mem::replace(&mut self.halves, [[F::zero(); 2]; M]),
        ))
    }
}

/// One task's Gruen-grouped sums
/// `Σ_x_out e_out[x_out] · Σ_x_in e_in[x_in] · terms(x_out·|e_in| + x_in)`.
pub(crate) struct GruenSums<'a, F, const N: usize> {
    e_out: &'a [F],
    e_in: &'a [F],
    inner: [F; N],
    pub(crate) total: [F; N],
}

impl<'a, F: JoltField, const N: usize> GruenSums<'a, F, N> {
    pub(crate) fn new(e_out: &'a [F], e_in: &'a [F]) -> Self {
        Self {
            e_out,
            e_in,
            inner: [F::zero(); N],
            total: [F::zero(); N],
        }
    }

    /// Tasks hold whole groups ([`task_ranges`]) and add their pairs in
    /// order, so a group's last pair closes it.
    pub(crate) fn add(&mut self, pair: usize, terms: [F; N]) {
        let x_in = pair & (self.e_in.len() - 1);
        for (inner, term) in self.inner.iter_mut().zip(terms) {
            *inner += self.e_in[x_in] * term;
        }
        if x_in + 1 == self.e_in.len() {
            let e_out = self.e_out[pair / self.e_in.len()];
            for (total, inner) in self.total.iter_mut().zip(&mut self.inner) {
                *total += e_out * *inner;
                *inner = F::zero();
            }
        }
    }
}

/// One task's receiver of a [`gruen_pair_pass`]: writes each completed
/// pair's halves into the task's slices of the tables being filled, if any,
/// and folds the pair's summand terms into the task's Gruen sums.
pub(crate) struct PairSink<'a, F, T, const M: usize, const N: usize> {
    slices: Option<[&'a mut [F]; M]>,
    first_pair: usize,
    terms: &'a T,
    sums: GruenSums<'a, F, N>,
}

impl<F, T, const M: usize, const N: usize> PairSink<'_, F, T, M, N>
where
    F: JoltField,
    T: Fn([[F; 2]; M]) -> [F; N],
{
    /// `halves[m]` is table `m`'s `[low, high]` pair; every pair of the
    /// task's range must arrive once, in order.
    pub(crate) fn emit(&mut self, pair: usize, halves: [[F; 2]; M]) {
        if let Some(slices) = &mut self.slices {
            let entry = 2 * (pair - self.first_pair);
            for (slice, pair_halves) in slices.iter_mut().zip(&halves) {
                slice[entry..entry + 2].copy_from_slice(pair_halves);
            }
        }
        self.sums.add(pair, (self.terms)(halves));
    }
}

/// One pass of a streamed round of the split-eq summand
/// `Σ_pair E(pair) · terms(pair)`, its pairs spanning `2^log_pair_cycles`
/// cycles each: the cycle domain (`|e_out| · |e_in|` pairs) splits into
/// [`task_ranges`] of whole Gruen groups, and each task's `walk` visits its
/// range once, serially, in callbacks of at most `chunk` rows, passing every
/// completed pair to its [`PairSink`] in order. A filling pass splits
/// `tables` (pair `z` at entries `2z, 2z + 1`) at the task boundaries, so
/// each task writes only its own slices. Task results are `N` field
/// elements, so they reduce.
pub(crate) fn gruen_pair_pass<F, T, const M: usize, const N: usize>(
    split_eq: &GruenSplitEqPolynomial<F>,
    log_pair_cycles: usize,
    tables: Option<[&mut [F]; M]>,
    terms: T,
    walk: impl Fn(Range<usize>, usize, &mut PairSink<'_, F, T, M, N>) -> Result<(), WitnessError> + Sync,
) -> Result<[F; N], WitnessError>
where
    F: JoltField,
    T: Fn([[F; 2]; M]) -> [F; N] + Sync,
{
    let e_out = split_eq.e_out_current();
    let e_in = split_eq.e_in_current();
    let group = e_in.len() << log_pair_cycles;
    let ranges = task_ranges(e_out.len() * group, group);
    let slices: Vec<Option<[&mut [F]; M]>> = match tables {
        Some(tables) => split_at_tasks(tables, &ranges, |cycle| 2 * (cycle >> log_pair_cycles))
            .into_iter()
            .map(Some)
            .collect(),
        None => ranges.iter().map(|_| None).collect(),
    };
    let task =
        |(range, slices): (&Range<usize>, Option<[&mut [F]; M]>)| -> Result<[F; N], WitnessError> {
            let mut sink = PairSink {
                slices,
                first_pair: range.start >> log_pair_cycles,
                terms: &terms,
                sums: GruenSums::new(e_out, e_in),
            };
            walk(range.clone(), VISIT_ROWS.min(group), &mut sink)?;
            Ok(sink.sums.total)
        };
    let add = |left: [F; N], right: [F; N]| std::array::from_fn(|i| left[i] + right[i]);
    #[cfg(feature = "parallel")]
    return ranges
        .par_iter()
        .zip(slices)
        .map(task)
        .try_reduce(|| [F::zero(); N], |left, right| Ok(add(left, right)));
    #[cfg(not(feature = "parallel"))]
    ranges
        .iter()
        .zip(slices)
        .try_fold([F::zero(); N], |sum, task_input| {
            Ok(add(sum, task(task_input)?))
        })
}

/// `(q(0), q(∞))` terms of one pair of a two-factor product summand `a·b`,
/// from the factors' `[low, high]` halves.
pub(crate) fn product_endpoint_terms<F: JoltField>([a, b]: [[F; 2]; 2]) -> [F; 2] {
    [a[0] * b[0], (a[1] - a[0]) * (b[1] - b[0])]
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

#[cfg(feature = "parallel")]
pub(crate) fn merge_evals<F: JoltField>(mut left: Vec<F>, right: Vec<F>) -> Vec<F> {
    for (left, right) in left.iter_mut().zip(right) {
        *left += right;
    }
    left
}

/// Adds one low-to-high pair of the cubic `a(t) · b(t) · c(t)`, each factor
/// given by its `(lo, hi)` values, to the deferred-reduction lanes of
/// `t = 0, 2, 3`.
#[inline]
pub(crate) fn accumulate_triple_product<F: JoltField>(
    acc: &mut [F::Accumulator; 3],
    (a_0, a_1): (F, F),
    (b_0, b_1): (F, F),
    (c_0, c_1): (F, F),
) {
    let (a_m, b_m, c_m) = (a_1 - a_0, b_1 - b_0, c_1 - c_0);
    let (a_2, b_2, c_2) = (a_1 + a_m, b_1 + b_m, c_1 + c_m);
    acc[0].fmadd(a_0 * b_0, c_0);
    acc[1].fmadd(a_2 * b_2, c_2);
    acc[2].fmadd((a_2 + a_m) * (b_2 + b_m), c_2 + c_m);
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
        accumulate_triple_product(acc, a(y), b(y), c(y));
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
    #[cfg(feature = "parallel")]
    {
        (0..groups)
            .into_par_iter()
            .fold(
                || vec![F::zero(); slots],
                |mut acc, y| {
                    accumulate(&mut acc, y);
                    acc
                },
            )
            .reduce(|| vec![F::zero(); slots], merge_evals)
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut acc = vec![F::zero(); slots];
        for y in 0..groups {
            accumulate(&mut acc, y);
        }
        acc
    }
}

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
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(crate) enum SplitLt<F> {
    Split {
        lt_lo: Vec<F>,
        lt_hi: Vec<F>,
        eq_hi: Vec<F>,
    },
    Dense(Vec<F>),
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

/// Where a kernel's typed rows live: a slice-backed witness serves an
/// owning handle and every pass re-extracts its windows on the fly — the
/// materialized row vector never exists; re-emulating sources retain the
/// collected rows. The generic twin of the spartan-outer kernel's store,
/// for every carry-style typed-row consumer.
#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "B")
)]
pub(crate) enum BundleStore<B> {
    /// The witness plane owns these rows; it reports them itself.
    #[cfg_attr(feature = "allocative", allocative(skip))]
    Owned(TraceRows),
    Retained(
        #[cfg_attr(feature = "allocative", allocative(visit = crate::backend::visit_heap_free_elements))]
         Vec<B>,
    ),
}

impl<B: WitnessBundle + Copy + Send + Sync> BundleStore<B> {
    /// Resolve for a witness plane: the owning handle when the source is
    /// slice-backed (and covers the cycle domain), a materialized collect
    /// otherwise.
    pub(crate) fn resolve<F: JoltField>(
        witness: &dyn JoltWitnessPlane<F>,
        cycles: usize,
    ) -> Result<Self, crate::KernelError<F>> {
        match witness.rows() {
            Some(rows) if cycles <= rows.cycles() => Ok(Self::Owned(rows)),
            _ => Ok(Self::Retained(collect_rows(witness, cycles)?)),
        }
    }

    pub(crate) fn access(&self) -> BundleAccess<'_, B> {
        match self {
            Self::Owned(rows) => BundleAccess::View(rows),
            Self::Retained(rows) => BundleAccess::Retained(rows),
        }
    }
}

pub(crate) enum BundleAccess<'a, B> {
    View(&'a TraceRows),
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

#[inline(always)]
pub(crate) fn mul_0_optimized<F: JoltField>(left: F, right: F) -> F {
    if left.is_zero() || right.is_zero() {
        F::zero()
    } else {
        left * right
    }
}

/// First-bind value at half-domain index `y` of a raw column:
/// `raw(2y) + r1·(raw(2y+1) − raw(2y))`.
#[inline]
pub(crate) fn bound_pair<F: JoltField>(raw: impl Fn(usize) -> F, r1: F, y: usize) -> F {
    let lo = raw(2 * y);
    let hi = raw(2 * y + 1);
    lo + mul_0_optimized(r1, hi - lo)
}

/// Two LSB binds of a raw `len`-entry column materialized directly at
/// `len / 4`, skipping the half-size intermediate.
pub(crate) fn bind_raw_twice<F: JoltField>(
    raw: impl Fn(usize) -> F + Sync,
    len: usize,
    r1: F,
    r2: F,
) -> Polynomial<F> {
    let pair = |z: usize| {
        let lo = bound_pair(&raw, r1, 2 * z);
        let hi = bound_pair(&raw, r1, 2 * z + 1);
        lo + mul_0_optimized(r2, hi - lo)
    };
    #[cfg(feature = "parallel")]
    let bound: Vec<F> = (0..len / 4).into_par_iter().map(pair).collect();
    #[cfg(not(feature = "parallel"))]
    let bound: Vec<F> = (0..len / 4).map(pair).collect();
    Polynomial::new(bound)
}

#[cfg(test)]
mod product_grid_tests {
    #[cfg(feature = "akita")]
    use jolt_field::Prime128OffsetA7F7;
    use jolt_field::{Accumulator, Fr, JoltField};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    use super::{accumulate_product_grid, product_grid_scratch_len, MAX_GRID_FACTORS};

    /// Accumulates several rows into shared lanes and compares every lane to
    /// the factors multiplied one by one at that lane's point. The scratch is
    /// refilled with random elements before each row, and the first two rows
    /// have all-zero slopes and all-zero values.
    fn assert_product_grid<F: JoltField>(factors: usize) {
        let mut rng = ChaCha20Rng::seed_from_u64(431);
        let mut lanes = vec![F::Accumulator::default(); factors];
        let mut expected = vec![F::zero(); factors];
        let mut scratch = vec![F::zero(); product_grid_scratch_len(factors)];
        let mut random = |zero: bool| -> Vec<F> {
            (0..factors)
                .map(|_| {
                    let value = F::random(&mut rng);
                    if zero {
                        F::zero()
                    } else {
                        value
                    }
                })
                .collect()
        };
        for row in 0..32 {
            let values = random(row == 1);
            let steps = random(row == 0);
            for (slot, value) in scratch.iter_mut().zip(random(false).iter().cycle()) {
                *slot = *value;
            }
            accumulate_product_grid(&values, &steps, &mut lanes, &mut scratch);
            for (index, expected) in expected.iter_mut().enumerate() {
                *expected += if index == factors - 1 {
                    steps.iter().copied().product::<F>()
                } else {
                    let point = F::from_u64(index as u64);
                    values
                        .iter()
                        .zip(&steps)
                        .map(|(value, step)| *value + point * *step)
                        .product::<F>()
                };
            }
        }
        for (index, (lane, expected)) in lanes.iter().zip(expected).enumerate() {
            assert_eq!(lane.reduce(), expected, "{factors} factors, lane {index}");
        }
    }

    /// Covers the direct walk (2, 3), the unrolled forms (4, 8), and the
    /// balanced tree at every other length up to the largest supported one,
    /// odd splits included.
    #[test]
    fn products_match_direct_evaluations_at_every_supported_length() {
        for factors in 2..=MAX_GRID_FACTORS {
            assert_product_grid::<Fr>(factors);
        }
    }

    #[cfg(feature = "akita")]
    #[test]
    fn products_match_direct_evaluations_at_every_supported_length_fp128() {
        for factors in 2..=MAX_GRID_FACTORS {
            assert_product_grid::<Prime128OffsetA7F7>(factors);
        }
    }
}

#[cfg(test)]
mod task_partition_tests {
    use super::task_ranges;
    #[cfg(feature = "parallel")]
    use super::TASKS_PER_THREAD;

    /// Tasks tile the domain in order, each a positive whole number of
    /// aligned units with the units spread evenly, and never outnumber
    /// `TASKS_PER_THREAD` per pool thread (one task without `parallel`).
    #[test]
    fn tasks_tile_the_domain_in_whole_aligned_units() {
        #[cfg(feature = "parallel")]
        let most_tasks = TASKS_PER_THREAD * rayon::current_num_threads();
        #[cfg(not(feature = "parallel"))]
        let most_tasks = 1;
        for (cycles, group) in [
            (1usize << 12, 1usize << 4),
            (1 << 12, 1 << 2),
            (1 << 12, 1 << 12),
            (1 << 6, 1 << 8),
            (1 << 20, 1),
            (1, 1),
        ] {
            let unit = cycles.min(group);
            let units = cycles / unit;
            let ranges = task_ranges(cycles, group);
            assert_eq!(ranges.len(), most_tasks.min(units));
            let (fewest, most) = (units / ranges.len(), units.div_ceil(ranges.len()));
            let mut next = 0;
            for range in &ranges {
                assert_eq!(range.start, next);
                assert_eq!(range.start % unit, 0);
                assert_eq!(range.len() % unit, 0);
                assert!((fewest..=most).contains(&(range.len() / unit)) && !range.is_empty());
                next = range.end;
            }
            assert_eq!(next, cycles);
        }
    }
}
