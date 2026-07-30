//! Small shared primitives of the optimized kernels.

use jolt_field::{Field, RingAccumulator};
use jolt_poly::{BindingOrder, EqPolynomial, LtPolynomial, Polynomial, UnivariatePoly};
use jolt_witness::{
    collect_bundles_par, stream_witnesses, RowSource, StreamConsumer, WitnessBundle, WitnessError,
};

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
pub(crate) fn collect_rows<B: WitnessBundle + Clone + Send + Sync>(
    source: &(impl RowSource + ?Sized),
    cycles: usize,
) -> Result<Vec<B>, WitnessError> {
    if let Some(access) = source.random_access() {
        if cycles <= access.cycles {
            return collect_bundles_par(&access, cycles);
        }
    }
    struct Presized<B> {
        rows: Vec<B>,
    }
    impl<B: WitnessBundle + Clone + Send + Sync> StreamConsumer for Presized<B> {
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
