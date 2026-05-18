//! Source abstractions for commitment backends.
//!
//! A source describes committed data and the traversal shapes a backend may
//! exploit. It does not prescribe the backend's commitment algorithm or
//! parallel schedule.

use std::iter::repeat_n;

use jolt_field::Field;
use jolt_poly::{MultilinearPoly, OneHotPolynomial, Polynomial, RlcSource};

use crate::claims::LinearSourceTerm;

/// Stable identifier for a committed source inside a batch commitment source.
///
/// In the Dory/Jolt trace path this can be a logical committed polynomial id.
/// In a packed PCS path this can instead identify a packed witness group. The
/// id names what the PCS commits to; it does not have to be one logical Jolt
/// polynomial.
pub trait SourceId: Copy + Eq + Ord + Send + Sync + 'static {}

impl<T> SourceId for T where T: Copy + Eq + Ord + Send + Sync + 'static {}

/// A compact coordinate into a one-hot domain.
///
/// The value is the hot basis-vector index `k` in `e_k`. The surrounding
/// [`OneHotRow`] carries the domain size, so this type only stores the
/// coordinate. Current Jolt one-hot chunks have at most `2^8` entries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(transparent)]
pub struct OneHotIndex(u8);

impl OneHotIndex {
    /// Creates a one-hot coordinate when `index < 2^log_domain_size`.
    pub fn new(index: u8, log_domain_size: u8) -> Option<Self> {
        (log_domain_size <= 8 && (index as usize) < (1usize << log_domain_size))
            .then_some(Self(index))
    }

    /// Returns the coordinate as an array/vector index.
    pub fn get(self) -> usize {
        self.0 as usize
    }
}

/// A row of one-hot entries, one entry per trace column in the current chunk.
///
/// `log_domain_size` says that every hot coordinate lives in a one-hot domain
/// of size `2^log_domain_size`. The entries record whether each trace column
/// has a required hot coordinate or may be zero. Dory consumes this as the
/// current streaming one-hot chunk shape: it builds one row commitment per hot
/// coordinate, with columns contributing to the row for their hot coordinate.
pub struct OneHotRow<'a> {
    pub log_domain_size: u8,
    pub entries: OneHotEntries<'a>,
}

/// Per-column one-hot data for a [`OneHotRow`].
///
/// This enum avoids forcing all one-hot rows through `Option`. Rows such as
/// instruction and bytecode RA have one hot coordinate for every trace column.
/// Rows such as RAM RA can have no committed address for a column after address
/// remapping, so they need the zero-or-one representation.
pub enum OneHotEntries<'a> {
    /// Every trace column contributes exactly one one-hot basis vector.
    ///
    /// Entry `indices[col] = k` means column `col` contributes `e_k`.
    OnePerColumn(&'a [OneHotIndex]),

    /// Each trace column contributes either zero or one one-hot basis vector.
    ///
    /// Entry `indices[col] = Some(k)` means column `col` contributes `e_k`.
    /// Entry `indices[col] = None` means column `col` contributes zero.
    MaybeZero(&'a [Option<OneHotIndex>]),
}

impl OneHotEntries<'_> {
    /// Number of trace columns represented by this row.
    pub fn len(&self) -> usize {
        match self {
            Self::OnePerColumn(indices) => indices.len(),
            Self::MaybeZero(indices) => indices.len(),
        }
    }

    /// Returns `true` when this row has no trace columns.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A borrowed row view of a polynomial source.
///
/// This is a traversal hint, not the core polynomial abstraction. Backends that
/// can exploit row structure consume these rows directly. Backends that do not
/// care about the encoding may interpret the row as field evaluations.
pub enum SourceRow<'a, F> {
    /// A row whose entries occupy evenly spaced columns.
    ///
    /// `column_stride` is the distance between consecutive occupied columns in
    /// the backend's row view. For example, values with stride `4` occupy
    /// columns `0, 4, 8, ...`; all intervening columns are zero.
    /// Use stride `1` for a dense row.
    StridedFieldElements {
        values: &'a [F],
        column_stride: usize,
    },

    /// A row of signed integers embedded canonically into the field.
    ///
    /// This preserves small-scalar MSM paths without first materializing field
    /// elements.
    /// Use stride `1` for a dense row.
    StridedI128 {
        values: &'a [i128],
        column_stride: usize,
    },

    /// A row of unsigned 64-bit integers embedded canonically into the field.
    ///
    /// This preserves the common compact-polynomial benchmark and commitment
    /// path without paying the cost of first converting every row entry into a
    /// full-width field element.
    /// Use stride `1` for a dense row.
    StridedU64 {
        values: &'a [u64],
        column_stride: usize,
    },

    /// A streaming one-hot chunk whose entries are one-hot vectors over a small
    /// domain.
    ///
    /// This is included for Jolt's RA commitments, where Dory can preserve the
    /// existing grouped-addition path without materializing a dense `{0,1}`
    /// table. Backends that do not exploit this shape can expand it explicitly
    /// in the same hot-coordinate-major order.
    OneHot(OneHotRow<'a>),
}

/// A single polynomial-like object that a PCS can commit to and open.
///
/// The source owns semantic operations: evaluate at a point, traverse rows, and
/// fold rows for opening-time vector/matrix products. It may be materialized or
/// lazy; for example, it can be backed by an execution trace.
pub trait CommitmentSource<F: Field>: Send + Sync {
    /// Number of multilinear variables in the source.
    fn num_vars(&self) -> usize;

    /// Evaluates the source at a multilinear point.
    fn evaluate(&self, point: &[F]) -> F;

    /// Preferred row length for commitment traversal, when the source has one.
    ///
    /// This is a source traversal fact, not a PCS-specific partition. Dory
    /// interprets it as its row width and derives its private matrix split
    /// internally. Other backends may ignore it or use it as a tiling hint.
    fn natural_chunk_len(&self) -> Option<usize> {
        None
    }

    /// Visits row-shaped chunks of the source using `chunk_len` columns.
    ///
    /// The borrowed row only has to remain valid for the duration of the visit
    /// call, which lets trace-backed sources allocate temporary row buffers and
    /// avoid ownership wrappers such as `Cow`.
    fn for_each_row<V>(&self, chunk_len: usize, visit: V)
    where
        V: for<'row> FnMut(usize, SourceRow<'row, F>);

    /// Maps row-shaped chunks of the source into owned backend results.
    ///
    /// This is the performance-oriented companion to
    /// [`for_each_row`](Self::for_each_row). The default implementation is a
    /// sequential traversal, which is sufficient for lazy sources that produce
    /// temporary row buffers. Materialized sources can override this method to
    /// parallelize over borrowed row chunks without first copying them into an
    /// owned staging buffer.
    fn map_rows<R, V>(&self, chunk_len: usize, visit: V) -> Vec<R>
    where
        R: Send,
        V: for<'row> Fn(usize, SourceRow<'row, F>) -> R + Send + Sync,
    {
        let mut rows = Vec::new();
        self.for_each_row(chunk_len, |row_index, row| {
            rows.push(visit(row_index, row));
        });
        rows
    }

    /// Whether the whole source is a unit-valued one-hot polynomial.
    ///
    /// This preserves existing commitment fast paths that only need hot
    /// coordinates instead of row materialization. Sources that expose richer
    /// per-row one-hot structure can use [`SourceRow::OneHot`] instead.
    fn is_one_hot(&self) -> bool {
        false
    }

    /// Visits the hot flat indices when [`is_one_hot`](Self::is_one_hot) is true.
    ///
    /// Backends use this for current Dory-style one-hot commitment, where each
    /// hot index maps directly to one SRS basis addition. Non-one-hot sources
    /// may leave the default empty traversal.
    fn for_each_one<V>(&self, _visit: V)
    where
        V: FnMut(usize),
    {
    }

    /// Folds rows against the left-side weights used by opening algorithms.
    fn fold_rows(&self, left: &[F], chunk_len: usize) -> Vec<F>;
}

impl<F, S> CommitmentSource<F> for &S
where
    F: Field,
    S: CommitmentSource<F> + ?Sized,
{
    fn num_vars(&self) -> usize {
        (**self).num_vars()
    }

    fn evaluate(&self, point: &[F]) -> F {
        (**self).evaluate(point)
    }

    fn natural_chunk_len(&self) -> Option<usize> {
        (**self).natural_chunk_len()
    }

    fn for_each_row<V>(&self, chunk_len: usize, visit: V)
    where
        V: for<'row> FnMut(usize, SourceRow<'row, F>),
    {
        (**self).for_each_row(chunk_len, visit);
    }

    fn map_rows<R, V>(&self, chunk_len: usize, visit: V) -> Vec<R>
    where
        R: Send,
        V: for<'row> Fn(usize, SourceRow<'row, F>) -> R + Send + Sync,
    {
        (**self).map_rows(chunk_len, visit)
    }

    fn is_one_hot(&self) -> bool {
        (**self).is_one_hot()
    }

    fn for_each_one<V>(&self, visit: V)
    where
        V: FnMut(usize),
    {
        (**self).for_each_one(visit);
    }

    fn fold_rows(&self, left: &[F], chunk_len: usize) -> Vec<F> {
        (**self).fold_rows(left, chunk_len)
    }
}

fn chunk_len_to_sigma(chunk_len: usize) -> usize {
    assert!(
        chunk_len.is_power_of_two(),
        "commitment source chunk length ({chunk_len}) must be a power of two",
    );
    chunk_len.trailing_zeros() as usize
}

fn multilinear_num_vars<F, T>(source: &T) -> usize
where
    F: Field,
    T: MultilinearPoly<F> + ?Sized,
{
    MultilinearPoly::num_vars(source)
}

fn multilinear_evaluate<F, T>(source: &T, point: &[F]) -> F
where
    F: Field,
    T: MultilinearPoly<F> + ?Sized,
{
    MultilinearPoly::evaluate(source, point)
}

fn multilinear_for_each_row<F, T, V>(source: &T, chunk_len: usize, mut visit: V)
where
    F: Field,
    T: MultilinearPoly<F> + ?Sized,
    V: for<'row> FnMut(usize, SourceRow<'row, F>),
{
    let sigma = chunk_len_to_sigma(chunk_len);
    MultilinearPoly::for_each_row(source, sigma, &mut |row_index, row| {
        visit(
            row_index,
            SourceRow::StridedFieldElements {
                values: row,
                column_stride: 1,
            },
        );
    });
}

fn multilinear_is_one_hot<F, T>(source: &T) -> bool
where
    F: Field,
    T: MultilinearPoly<F> + ?Sized,
{
    MultilinearPoly::is_one_hot(source)
}

fn multilinear_for_each_one<F, T, V>(source: &T, mut visit: V)
where
    F: Field,
    T: MultilinearPoly<F> + ?Sized,
    V: FnMut(usize),
{
    MultilinearPoly::for_each_one(source, &mut visit);
}

fn multilinear_fold_rows<F, T>(source: &T, left: &[F], chunk_len: usize) -> Vec<F>
where
    F: Field,
    T: MultilinearPoly<F> + ?Sized,
{
    let sigma = chunk_len_to_sigma(chunk_len);
    MultilinearPoly::fold_rows(source, left, sigma)
}

macro_rules! impl_commitment_source_for_multilinear {
    ($ty:ty) => {
        impl<F: Field> CommitmentSource<F> for $ty {
            fn num_vars(&self) -> usize {
                multilinear_num_vars(self)
            }

            fn evaluate(&self, point: &[F]) -> F {
                multilinear_evaluate(self, point)
            }

            fn for_each_row<V>(&self, chunk_len: usize, visit: V)
            where
                V: for<'row> FnMut(usize, SourceRow<'row, F>),
            {
                multilinear_for_each_row(self, chunk_len, visit);
            }

            fn is_one_hot(&self) -> bool {
                multilinear_is_one_hot(self)
            }

            fn for_each_one<V>(&self, visit: V)
            where
                V: FnMut(usize),
            {
                multilinear_for_each_one(self, visit);
            }

            fn fold_rows(&self, left: &[F], chunk_len: usize) -> Vec<F> {
                multilinear_fold_rows(self, left, chunk_len)
            }
        }
    };
}

impl_commitment_source_for_multilinear!(Polynomial<F>);
impl_commitment_source_for_multilinear!(Vec<F>);
impl_commitment_source_for_multilinear!([F]);

impl<F, S> CommitmentSource<F> for RlcSource<F, S>
where
    F: Field,
    S: MultilinearPoly<F>,
{
    fn num_vars(&self) -> usize {
        multilinear_num_vars(self)
    }

    fn evaluate(&self, point: &[F]) -> F {
        multilinear_evaluate(self, point)
    }

    fn for_each_row<V>(&self, chunk_len: usize, visit: V)
    where
        V: for<'row> FnMut(usize, SourceRow<'row, F>),
    {
        multilinear_for_each_row(self, chunk_len, visit);
    }

    fn is_one_hot(&self) -> bool {
        multilinear_is_one_hot(self)
    }

    fn for_each_one<V>(&self, visit: V)
    where
        V: FnMut(usize),
    {
        multilinear_for_each_one(self, visit);
    }

    fn fold_rows(&self, left: &[F], chunk_len: usize) -> Vec<F> {
        multilinear_fold_rows(self, left, chunk_len)
    }
}

impl<F: Field> CommitmentSource<F> for OneHotPolynomial {
    fn num_vars(&self) -> usize {
        multilinear_num_vars::<F, _>(self)
    }

    fn evaluate(&self, point: &[F]) -> F {
        multilinear_evaluate(self, point)
    }

    fn for_each_row<V>(&self, chunk_len: usize, visit: V)
    where
        V: for<'row> FnMut(usize, SourceRow<'row, F>),
    {
        multilinear_for_each_row(self, chunk_len, visit);
    }

    fn is_one_hot(&self) -> bool {
        multilinear_is_one_hot::<F, _>(self)
    }

    fn for_each_one<V>(&self, visit: V)
    where
        V: FnMut(usize),
    {
        multilinear_for_each_one::<F, _, _>(self, visit);
    }

    fn fold_rows(&self, left: &[F], chunk_len: usize) -> Vec<F> {
        multilinear_fold_rows(self, left, chunk_len)
    }
}

/// Materializes a commitment source into its canonical multilinear-evaluation vector.
///
/// This helper is the shared fallback for PCS backends that do not have a native
/// streaming path for a source. It centralizes `SourceRow` expansion so all
/// materializing backends agree on strided rows and one-hot layout. One-hot rows
/// are accumulated across chunks and flushed in hot-index-major order, matching
/// the polynomial layout used by Jolt's one-hot committed sources.
pub fn materialize_source_evaluations<F, S>(source: &S) -> Vec<F>
where
    F: Field,
    S: CommitmentSource<F> + ?Sized,
{
    fn flush_one_hot<F: Field>(
        evaluations: &mut Vec<F>,
        pending: &mut Option<(usize, Vec<Vec<Option<usize>>>)>,
    ) {
        let Some((domain_size, chunks)) = pending.take() else {
            return;
        };

        let trace_len = chunks.iter().map(Vec::len).sum::<usize>();
        let start = evaluations.len();
        evaluations.resize(start + trace_len * domain_size, F::zero());

        let mut chunk_offset = 0;
        for chunk in chunks {
            for (column, hot_index) in chunk.iter().enumerate() {
                if let Some(hot_index) = hot_index {
                    evaluations[start + hot_index * trace_len + chunk_offset + column] =
                        F::from_u64(1);
                }
            }
            chunk_offset += chunk.len();
        }
    }

    let mut evaluations = Vec::with_capacity(1usize << source.num_vars());
    let mut one_hot_chunks = None;
    let chunk_len = source
        .natural_chunk_len()
        .unwrap_or_else(|| 1usize << source.num_vars());
    source.for_each_row(chunk_len, |_, row| match row {
        SourceRow::StridedFieldElements {
            values,
            column_stride,
        } => {
            flush_one_hot(&mut evaluations, &mut one_hot_chunks);
            for value in values {
                evaluations.push(*value);
                evaluations.extend(repeat_n(F::zero(), column_stride.saturating_sub(1)));
            }
        }
        SourceRow::StridedI128 {
            values,
            column_stride,
        } => {
            flush_one_hot(&mut evaluations, &mut one_hot_chunks);
            for value in values {
                evaluations.push(F::from_i128(*value));
                evaluations.extend(repeat_n(F::zero(), column_stride.saturating_sub(1)));
            }
        }
        SourceRow::StridedU64 {
            values,
            column_stride,
        } => {
            flush_one_hot(&mut evaluations, &mut one_hot_chunks);
            for value in values {
                evaluations.push(F::from_u64(*value));
                evaluations.extend(repeat_n(F::zero(), column_stride.saturating_sub(1)));
            }
        }
        SourceRow::OneHot(row) => {
            let domain_size = 1usize << row.log_domain_size;
            let chunk = match row.entries {
                OneHotEntries::OnePerColumn(indices) => {
                    indices.iter().map(|index| Some(index.get())).collect()
                }
                OneHotEntries::MaybeZero(indices) => indices
                    .iter()
                    .map(|index| index.map(OneHotIndex::get))
                    .collect(),
            };

            match &mut one_hot_chunks {
                Some((existing_domain_size, chunks)) => {
                    assert_eq!(
                        *existing_domain_size, domain_size,
                        "one source changed one-hot domain size during materialization",
                    );
                    chunks.push(chunk);
                }
                None => {
                    one_hot_chunks = Some((domain_size, vec![chunk]));
                }
            }
        }
    });
    flush_one_hot(&mut evaluations, &mut one_hot_chunks);
    evaluations
}

/// A batch of committed sources that can share one traversal.
///
/// This is the no-regression traversal hook for current CycleMajor Dory
/// commitment. The default PCS implementation can ignore it and commit sources
/// one at a time through [`source`](Self::source).
pub trait BatchCommitmentSource<F: Field>: Send + Sync {
    type Id: SourceId;

    /// Borrowed single-source adapter for a source in this batch.
    type Source<'a>: CommitmentSource<F> + 'a
    where
        Self: 'a;

    /// All source ids this batch can expose, in natural protocol order.
    fn source_ids(&self) -> &[Self::Id];

    /// Number of multilinear variables in the selected source.
    fn num_vars(&self, id: Self::Id) -> usize;

    /// Preferred shared row length for committing the selected sources.
    fn natural_chunk_len(&self, _ids: &[Self::Id]) -> Option<usize> {
        None
    }

    /// Returns a single-source view for backends that do not use batch traversal.
    fn source(&self, id: Self::Id) -> Self::Source<'_>;

    /// Maps a row visitor over many sources while sharing source traversal.
    ///
    /// The returned vector is row-major: `output[row_index][id_index]`.
    fn map_rows<R, V>(&self, chunk_len: usize, ids: &[Self::Id], visit: V) -> Vec<Vec<R>>
    where
        R: Send,
        V: for<'row> Fn(Self::Id, SourceRow<'row, F>) -> R + Send + Sync;
}

/// A batch of already-committed sources available for opening.
///
/// Commitment and opening have different traversal needs. Commitment wants to
/// stream rows for many sources at once. Opening needs a registry that can
/// recover individual committed sources and borrow their backend-owned opening
/// hints. Algebraic combinations of several sources are separate capabilities
/// rather than part of this generic registry.
pub trait BatchOpeningSource<F: Field, OpeningHint>: Send + Sync {
    type Id: SourceId;

    /// Borrowed single-source adapter for a source in this batch.
    type Source<'a>: CommitmentSource<F> + 'a
    where
        Self: 'a;

    /// Returns a single-source view for the selected committed source.
    fn source(&self, id: Self::Id) -> Self::Source<'_>;

    /// Borrows the backend-owned opening hint produced with this source's
    /// commitment.
    ///
    /// The borrow is deliberate: production Dory hints contain row commitments,
    /// and source-backed Stage 8 opening must combine those hints without
    /// cloning the row-commitment vectors in the hot path.
    fn opening_hint(&self, id: Self::Id) -> &OpeningHint;
}

/// Opening-source capability for linear combinations of committed sources.
///
/// This is the natural capability for homomorphic/RLC-style batch openings:
/// after a PCS samples batching coefficients, the prover can expose the source
/// `Σ coefficient_i * source_i` as an ordinary [`CommitmentSource`]. Backends
/// with a different batching strategy can ignore this trait and provide their
/// own PCS-specific opening extension instead.
pub trait LinearCombinationOpeningSource<F: Field, OpeningHint>:
    BatchOpeningSource<F, OpeningHint>
{
    /// Source representing the requested linear combination.
    type LinearCombination<'a>: CommitmentSource<F> + 'a
    where
        Self: 'a;

    /// Builds a source for `Σ terms[i].coefficient * source(terms[i].source_id)`.
    ///
    /// The mutable receiver lets streaming implementations move one-shot state
    /// such as advice polynomials into the combined source without interior
    /// mutability or hidden caches.
    fn linear_combination<'a>(
        &'a mut self,
        terms: &[LinearSourceTerm<F, Self::Id>],
    ) -> Self::LinearCombination<'a>;
}

/// Fallback linear-combination source for callers without a streaming path.
///
/// This helper eagerly materializes each input source, combines the evaluations,
/// and then exposes the result as a normal [`CommitmentSource`]. It is useful
/// for tests and simple backends. Hot streaming paths should usually implement
/// [`LinearCombinationOpeningSource`] with a borrowed or one-shot source that
/// preserves their native traversal.
pub struct MaterializedLinearCombination<F: Field> {
    evaluations: Vec<F>,
}

impl<F: Field> MaterializedLinearCombination<F> {
    /// Eagerly materializes a linear combination using individual source views.
    pub fn new<B, OpeningHint>(source_batch: &B, terms: &[LinearSourceTerm<F, B::Id>]) -> Self
    where
        B: BatchOpeningSource<F, OpeningHint>,
    {
        let Some((first, rest)) = terms.split_first() else {
            return Self {
                evaluations: Vec::new(),
            };
        };

        let mut evaluations = materialize_source_evaluations(&source_batch.source(first.source_id));
        for value in &mut evaluations {
            *value *= first.coefficient;
        }

        for term in rest {
            let next = materialize_source_evaluations(&source_batch.source(term.source_id));
            assert_eq!(
                evaluations.len(),
                next.len(),
                "cannot linearly combine sources with different materialized lengths",
            );
            for (acc, value) in evaluations.iter_mut().zip(next) {
                *acc += term.coefficient * value;
            }
        }

        Self { evaluations }
    }
}

impl<F: Field> CommitmentSource<F> for MaterializedLinearCombination<F> {
    fn num_vars(&self) -> usize {
        if self.evaluations.is_empty() {
            0
        } else {
            assert!(
                self.evaluations.len().is_power_of_two(),
                "materialized linear-combination source length must be a power of two",
            );
            self.evaluations.len().trailing_zeros() as usize
        }
    }

    fn evaluate(&self, point: &[F]) -> F {
        Polynomial::new(self.evaluations.clone()).evaluate(point)
    }

    fn for_each_row<V>(&self, chunk_len: usize, visit: V)
    where
        V: for<'row> FnMut(usize, SourceRow<'row, F>),
    {
        multilinear_for_each_row::<F, _, _>(&self.evaluations, chunk_len, visit);
    }

    fn fold_rows(&self, left: &[F], chunk_len: usize) -> Vec<F> {
        multilinear_fold_rows::<F, _>(&self.evaluations, left, chunk_len)
    }
}
