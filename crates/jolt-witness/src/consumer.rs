//! The fused trace→bundles pass: one row walk drives a statically-known set
//! of consumers.

use std::ops::Range;

use jolt_program::execution::TraceRow;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::bundle::WitnessBundle;
use crate::witnesses::WitnessEnv;
use crate::WitnessError;

/// One consumer of a bundle stream. `Option<C>` is also a consumer:
/// membership in a set is static, presence is runtime.
pub trait StreamConsumer: Send + Sync {
    type Witness: WitnessBundle + Send;

    fn consume(&mut self, chunk: &[Self::Witness]);

    /// Whether this consumer wants the bundles at all. An inactive consumer
    /// is skipped *before* extraction, so a runtime-absent slot costs nothing
    /// beyond the branch — extraction is the expensive half of the pass.
    fn is_active(&self) -> bool {
        true
    }
}

impl<C: StreamConsumer> StreamConsumer for Option<C> {
    type Witness = C::Witness;

    fn consume(&mut self, chunk: &[Self::Witness]) {
        if let Some(consumer) = self {
            consumer.consume(chunk);
        }
    }

    fn is_active(&self) -> bool {
        self.as_ref().is_some_and(C::is_active)
    }
}

/// A statically-known set of consumers (a tuple) fanned out over one walk.
/// The caller owns the tuple and lends `&mut`.
pub trait ConsumerSet {
    fn consume_chunk(
        &mut self,
        rows: &[TraceRow],
        next_after: Option<&TraceRow>,
        env: &WitnessEnv<'_>,
    ) -> Result<(), WitnessError>;
}

/// Buffers below this size extract serially — rayon dispatch would cost more
/// than the extraction itself.
#[cfg(feature = "parallel")]
const PAR_EXTRACT_THRESHOLD: usize = 128;

fn deliver<C: StreamConsumer>(
    consumer: &mut C,
    rows: &[TraceRow],
    next_after: Option<&TraceRow>,
    env: &WitnessEnv<'_>,
) -> Result<(), WitnessError> {
    if !consumer.is_active() {
        return Ok(());
    }
    // Extraction is pure per cycle window, so buffers extract in parallel;
    // chunk order (the consumer's contract) is unchanged.
    let extract = |(index, row): (usize, &TraceRow)| {
        C::Witness::from_row(row, rows.get(index + 1).or(next_after), env)
    };
    #[cfg(feature = "parallel")]
    let bundles: Vec<C::Witness> = if rows.len() >= PAR_EXTRACT_THRESHOLD {
        rows.par_iter()
            .enumerate()
            .map(extract)
            .collect::<Result<_, _>>()?
    } else {
        rows.iter()
            .enumerate()
            .map(extract)
            .collect::<Result<_, _>>()?
    };
    #[cfg(not(feature = "parallel"))]
    let bundles: Vec<C::Witness> = rows
        .iter()
        .enumerate()
        .map(extract)
        .collect::<Result<_, _>>()?;
    consumer.consume(&bundles);
    Ok(())
}

macro_rules! consumer_set_tuple {
    ($($name:ident : $index:tt),+) => {
        impl<$($name: StreamConsumer),+> ConsumerSet for ($($name,)+) {
            fn consume_chunk(
                &mut self,
                rows: &[TraceRow],
                next_after: Option<&TraceRow>,
                env: &WitnessEnv<'_>,
            ) -> Result<(), WitnessError> {
                $(deliver(&mut self.$index, rows, next_after, env)?;)+
                Ok(())
            }
        }
    };
}

consumer_set_tuple!(A: 0);
consumer_set_tuple!(A: 0, B: 1);
consumer_set_tuple!(A: 0, B: 1, C: 2);
consumer_set_tuple!(A: 0, B: 1, C: 2, D: 3);
consumer_set_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4);
consumer_set_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4, G: 5);
consumer_set_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4, G: 5, H: 6);
consumer_set_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4, G: 5, H: 6, I: 7);

/// The per-chunk callback of a [`RowSource`] walk: a row buffer, the
/// lookahead row following it (`None` only at the end of the cycle domain),
/// and the extraction environment.
pub type ChunkVisitor<'a> =
    dyn FnMut(&[TraceRow], Option<&TraceRow>, &WitnessEnv<'_>) -> Result<(), WitnessError> + 'a;

/// Sequential row access for the pass: trace-backed today, segment-backed
/// later. Random access is deliberately inexpressible — except through
/// [`RowSource::random_access`], the measured escape hatch for
/// order-insensitive whole-range collection: the chunked walk serializes on
/// per-chunk staging and consume copies, and at 2^25 cycles on a 64-thread
/// host the collection walks alone were most of the prover's wall time.
pub trait RowSource {
    /// Visits the half-open cycle `range` in order as buffers of at most
    /// `chunk_size` rows; `[0, T)` today, segments later.
    fn visit_chunks(
        &self,
        range: Range<usize>,
        chunk_size: usize,
        visitor: &mut ChunkVisitor<'_>,
    ) -> Result<(), WitnessError>;

    /// The random-access view of a slice-backed source: `Some` lets
    /// whole-range collectors ([`collect_bundles`], the kernels' presized
    /// twins) take the index-parallel path over the physical rows. `None`
    /// (the default, and every re-emulating source) keeps the sequential
    /// chunk walk as the only access shape.
    fn random_access(&self) -> Option<RandomAccessRows<'_>> {
        None
    }

    /// An owning counterpart of [`RowSource::random_access`], for consumers
    /// that outlive their borrow of the source (sumcheck kernels holding
    /// state across rounds): re-deriving per-cycle windows from the handle
    /// replaces retaining materialized row vectors at gigabyte scale. `None`
    /// whenever `random_access` is.
    fn owned_rows(&self) -> Option<OwnedRows> {
        None
    }
}

/// A shared owning handle to a slice-backed source's rows and extraction
/// context: [`OwnedRows::view`] re-creates the borrowed random-access view
/// on demand.
pub struct OwnedRows {
    rows: std::sync::Arc<Vec<TraceRow>>,
    cycles: usize,
    preprocessing: std::sync::Arc<jolt_program::preprocess::JoltProgramPreprocessing>,
}

impl OwnedRows {
    pub(crate) fn new(
        rows: std::sync::Arc<Vec<TraceRow>>,
        cycles: usize,
        preprocessing: std::sync::Arc<jolt_program::preprocess::JoltProgramPreprocessing>,
    ) -> Self {
        Self {
            rows,
            cycles,
            preprocessing,
        }
    }

    /// The padded cycle-domain size the handle serves.
    pub fn cycles(&self) -> usize {
        self.cycles
    }

    /// The borrowed random-access view over the held rows.
    pub fn view(&self) -> RandomAccessRows<'_> {
        RandomAccessRows::new(
            &self.rows,
            self.cycles,
            WitnessEnv {
                preprocessing: &self.preprocessing,
            },
        )
    }
}

/// A slice-backed source's rows and extraction context, for index-parallel
/// collection. Cycles at or beyond the physical rows are padding (default)
/// rows, exactly as the sequential walk serves them.
pub struct RandomAccessRows<'a> {
    /// The physical trace rows.
    rows: &'a [TraceRow],
    /// The padded cycle-domain size (`2^log_t`); the lookahead window ends
    /// here regardless of the collected range.
    cycles: usize,
    /// The extraction environment shared by every window.
    env: WitnessEnv<'a>,
    /// The padding row served at and beyond the physical trace, shared by
    /// every window (and every thread) of the view.
    padding: TraceRow,
}

impl<'a> RandomAccessRows<'a> {
    pub(crate) fn new(rows: &'a [TraceRow], cycles: usize, env: WitnessEnv<'a>) -> Self {
        Self {
            rows,
            cycles,
            env,
            padding: TraceRow::default(),
        }
    }

    /// The padded cycle-domain size the view serves.
    pub fn cycles(&self) -> usize {
        self.cycles
    }

    /// Extracts the bundle at `index` with the sequential walk's one-row
    /// lookahead window (padding rows at and beyond the physical trace, no
    /// lookahead at the end of the cycle domain). Pure per index — callers
    /// extract from any thread in any order.
    #[inline]
    pub fn window<B: WitnessBundle>(&self, index: usize) -> Result<B, WitnessError> {
        let current = self.rows.get(index).unwrap_or(&self.padding);
        let next =
            (index + 1 < self.cycles).then(|| self.rows.get(index + 1).unwrap_or(&self.padding));
        B::from_row(current, next, &self.env)
    }
}

/// The parallel scatter grain of the index-parallel collectors: big enough
/// to amortize rayon dispatch, small enough to load-balance skewed
/// extraction.
#[cfg(feature = "parallel")]
const COLLECT_PAR_CHUNK: usize = 1 << 12;

/// In-place parallel collection of `window(0..count)` into a fresh vector:
/// workers write straight into the destination's spare capacity — no
/// per-thread segment buffers or concatenation (rayon's `Result` collect
/// loses indexedness and stages every segment). First error wins; the
/// partially-written buffer is abandoned without drops (elements are plain
/// data across every caller).
#[cfg(feature = "parallel")]
pub(crate) fn par_collect_windows<V: Send>(
    count: usize,
    window: impl Fn(usize) -> Result<V, WitnessError> + Send + Sync,
) -> Result<Vec<V>, WitnessError> {
    let mut out: Vec<V> = Vec::with_capacity(count);
    let spare = &mut out.spare_capacity_mut()[..count];
    let error = std::sync::Mutex::new(None);
    spare
        .par_chunks_mut(COLLECT_PAR_CHUNK)
        .enumerate()
        .for_each(|(chunk_index, destination)| {
            let base = chunk_index * COLLECT_PAR_CHUNK;
            for (offset, slot) in destination.iter_mut().enumerate() {
                match window(base + offset) {
                    Ok(value) => {
                        let _ = slot.write(value);
                    }
                    Err(failure) => {
                        if let Ok(mut guard) = error.try_lock() {
                            let _ = guard.get_or_insert(failure);
                        }
                        return;
                    }
                }
            }
        });
    #[expect(clippy::unwrap_used, reason = "no lock user can panic")]
    if let Some(failure) = error.into_inner().unwrap() {
        return Err(failure);
    }
    // SAFETY: the error latch is empty, so every chunk ran to completion and
    // initialized all `count` slots of the spare capacity above.
    unsafe { out.set_len(count) };
    Ok(out)
}

/// Index-parallel bundle collection over a random-access view, mapped
/// through `pack` element by element (so packed forms never stage the wide
/// bundle): values are identical to the sequential pass's — extraction is
/// pure per cycle window, and the lookahead row at the end of the range
/// matches the chunk walk's `next_after`.
pub fn collect_par_map<B: WitnessBundle, V: Send>(
    access: &RandomAccessRows<'_>,
    cycles: usize,
    pack: impl Fn(B) -> V + Send + Sync,
) -> Result<Vec<V>, WitnessError> {
    let window = |index: usize| access.window::<B>(index).map(&pack);
    #[cfg(feature = "parallel")]
    return par_collect_windows(cycles, window);
    #[cfg(not(feature = "parallel"))]
    (0..cycles).map(window).collect()
}

/// [`collect_par_map`] without the packing step: index-parallel collection
/// of the bundles themselves.
pub fn collect_bundles_par<B: WitnessBundle + Send>(
    access: &RandomAccessRows<'_>,
    cycles: usize,
) -> Result<Vec<B>, WitnessError> {
    collect_par_map(access, cycles, |bundle: B| bundle)
}

/// Index-parallel collection of one cycle sub-range into a reusable buffer
/// (cleared first, allocation kept): the pipelining collector's shape — a
/// caller overlapping extraction with downstream work re-fills two buffers
/// alternately instead of allocating per delivery. Elements must be plain
/// data (every witness bundle is); on error the buffer is left cleared.
pub fn collect_range_into<B: WitnessBundle + Send>(
    access: &RandomAccessRows<'_>,
    range: Range<usize>,
    out: &mut Vec<B>,
) -> Result<(), WitnessError> {
    let start = range.start;
    let count = range.end.saturating_sub(start);
    out.clear();
    #[cfg(feature = "parallel")]
    {
        out.reserve(count);
        let spare = &mut out.spare_capacity_mut()[..count];
        let error = std::sync::Mutex::new(None);
        spare
            .par_chunks_mut(COLLECT_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_index, destination)| {
                let base = start + chunk_index * COLLECT_PAR_CHUNK;
                for (offset, slot) in destination.iter_mut().enumerate() {
                    match access.window::<B>(base + offset) {
                        Ok(bundle) => {
                            let _ = slot.write(bundle);
                        }
                        Err(failure) => {
                            if let Ok(mut guard) = error.try_lock() {
                                let _ = guard.get_or_insert(failure);
                            }
                            return;
                        }
                    }
                }
            });
        #[expect(clippy::unwrap_used, reason = "no lock user can panic")]
        if let Some(failure) = error.into_inner().unwrap() {
            return Err(failure);
        }
        // SAFETY: the error latch is empty, so every chunk ran to completion
        // and initialized all `count` slots of the spare capacity above.
        unsafe { out.set_len(count) };
        Ok(())
    }
    #[cfg(not(feature = "parallel"))]
    {
        for index in range {
            out.push(access.window::<B>(index)?);
        }
        Ok(())
    }
}

/// The fused pass: walk `range` once and deliver each chunk to every
/// consumer in the set.
#[tracing::instrument(
    skip_all,
    name = "stream_witnesses",
    fields(cycles = range.end.saturating_sub(range.start))
)]
pub fn stream_witnesses<S: RowSource + ?Sized, C: ConsumerSet>(
    source: &S,
    range: Range<usize>,
    chunk_size: usize,
    consumers: &mut C,
) -> Result<(), WitnessError> {
    if chunk_size == 0 {
        return Err(WitnessError::InvalidDimensions {
            label: crate::JOLT_VM_LABEL,
            reason: "pass chunk size must be nonzero".to_owned(),
        });
    }
    source.visit_chunks(range, chunk_size, &mut |rows, next_after, env| {
        consumers.consume_chunk(rows, next_after, env)
    })
}

/// The chunk size of a single-consumer bundle-collection pass.
const BUNDLE_PASS_CHUNK: usize = 1 << 12;

/// Materialize one bundle type over `0..cycles` from a row source. The
/// object-safe counterpart of [`crate::BundleSource::bundles`] — `&dyn
/// RowSource` consumers (kernels behind the witness plane) collect their
/// typed rows through this.
#[tracing::instrument(
    skip_all,
    name = "collect_bundles",
    fields(bundle = core::any::type_name::<B>(), cycles)
)]
pub fn collect_bundles<B: WitnessBundle + Clone + Send + Sync>(
    source: &(impl RowSource + ?Sized),
    cycles: usize,
) -> Result<Vec<B>, WitnessError> {
    // Whole-range collection over a slice-backed source skips the chunked
    // walk (out-of-range requests fall through to it for its validation).
    if let Some(access) = source.random_access() {
        if cycles <= access.cycles() {
            return collect_bundles_par(&access, cycles);
        }
    }
    let mut consumers = (CollectBundles::<B>::default(),);
    stream_witnesses(source, 0..cycles, BUNDLE_PASS_CHUNK, &mut consumers)?;
    Ok(consumers.0.into_rows())
}

/// The collecting consumer: accumulates one bundle type across the pass.
/// Backends materialize bundle vectors through this, so the pass driver is
/// the live path, not speculative API.
#[derive(Clone, Debug)]
pub struct CollectBundles<W> {
    rows: Vec<W>,
}

impl<W> Default for CollectBundles<W> {
    fn default() -> Self {
        Self { rows: Vec::new() }
    }
}

impl<W> CollectBundles<W> {
    pub fn into_rows(self) -> Vec<W> {
        self.rows
    }
}

impl<W: WitnessBundle + Clone + Send + Sync> StreamConsumer for CollectBundles<W> {
    type Witness = W;

    fn consume(&mut self, chunk: &[W]) {
        self.rows.extend_from_slice(chunk);
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use super::*;
    use crate::testing::with_sample_backend;
    use crate::witnesses::{Extract, NextUnexpandedPc, ToField, UnexpandedPc};
    use crate::BundleSource;
    use jolt_claims::protocols::jolt::JoltPolynomialId;
    use jolt_field::Fr;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A hand-implemented bundle carrying a lookahead witness, so chunk
    /// boundaries are observable.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct WindowBundle {
        pc: UnexpandedPc,
        next_pc: NextUnexpandedPc,
    }

    impl WitnessBundle for WindowBundle {
        fn from_row(
            row: &TraceRow,
            next: Option<&TraceRow>,
            env: &WitnessEnv<'_>,
        ) -> Result<Self, WitnessError> {
            Ok(Self {
                pc: UnexpandedPc::extract(row, next, env)?,
                next_pc: NextUnexpandedPc::extract(row, next, env)?,
            })
        }

        fn annotated_ids() -> Vec<JoltPolynomialId> {
            Vec::new()
        }
    }

    /// Counts its own extractions, so a skipped consumer is observable.
    #[derive(Clone, Copy, Debug)]
    struct CountingBundle;

    static EXTRACTIONS: AtomicUsize = AtomicUsize::new(0);

    impl WitnessBundle for CountingBundle {
        fn from_row(
            _row: &TraceRow,
            _next: Option<&TraceRow>,
            _env: &WitnessEnv<'_>,
        ) -> Result<Self, WitnessError> {
            let _ = EXTRACTIONS.fetch_add(1, Ordering::Relaxed);
            Ok(Self)
        }

        fn annotated_ids() -> Vec<JoltPolynomialId> {
            Vec::new()
        }
    }

    fn collect_with_chunk_size(chunk_size: usize) -> Vec<WindowBundle> {
        with_sample_backend(|backend| {
            let mut consumers = (CollectBundles::<WindowBundle>::default(),);
            stream_witnesses(backend, 0..4, chunk_size, &mut consumers).unwrap();
            consumers.0.into_rows()
        })
    }

    #[test]
    fn lookahead_crosses_chunk_boundaries() {
        let whole = collect_with_chunk_size(4);
        for chunk_size in [1, 2, 3] {
            assert_eq!(collect_with_chunk_size(chunk_size), whole);
        }
        // The shifted column: next_pc[t] == pc[t + 1], 0 at the end.
        for (index, bundle) in whole.iter().enumerate() {
            let expected = whole.get(index + 1).map_or(0, |next| next.pc.0);
            assert_eq!(bundle.next_pc.0, expected);
        }
    }

    #[test]
    fn random_access_collection_matches_the_chunked_walk() {
        with_sample_backend(|backend| {
            // The routed path (index-parallel over the slice-backed trace).
            let routed: Vec<WindowBundle> = collect_bundles(backend, 4).unwrap();
            // The chunked walk, forced.
            let mut consumers = (CollectBundles::<WindowBundle>::default(),);
            stream_witnesses(backend, 0..4, 2, &mut consumers).unwrap();
            assert_eq!(routed, consumers.0.into_rows());
            // The mapped variant packs the same windows.
            let access = backend.random_access().unwrap();
            let packed: Vec<u64> =
                collect_par_map(&access, 4, |bundle: WindowBundle| bundle.pc.0).unwrap();
            let expected: Vec<u64> = routed.iter().map(|bundle| bundle.pc.0).collect();
            assert_eq!(packed, expected);
        });
    }

    #[test]
    fn one_walk_feeds_every_consumer_and_absent_slots_skip() {
        with_sample_backend(|backend| {
            let mut consumers = (
                CollectBundles::<WindowBundle>::default(),
                Some(CollectBundles::<WindowBundle>::default()),
                None::<CollectBundles<WindowBundle>>,
            );
            stream_witnesses(backend, 0..4, 2, &mut consumers).unwrap();
            let first = consumers.0.into_rows();
            assert_eq!(first.len(), 4);
            assert_eq!(consumers.1.unwrap().into_rows(), first);
            assert!(consumers.2.is_none());
        });
    }

    #[test]
    fn absent_slots_skip_extraction_too() {
        with_sample_backend(|backend| {
            EXTRACTIONS.store(0, Ordering::Relaxed);
            let mut consumers = (
                None::<CollectBundles<CountingBundle>>,
                CollectBundles::<WindowBundle>::default(),
            );
            stream_witnesses(backend, 0..4, 2, &mut consumers).unwrap();
            assert_eq!(consumers.1.into_rows().len(), 4);
            assert_eq!(EXTRACTIONS.load(Ordering::Relaxed), 0);

            let mut consumers = (Some(CollectBundles::<CountingBundle>::default()),);
            stream_witnesses(backend, 0..4, 2, &mut consumers).unwrap();
            assert_eq!(EXTRACTIONS.load(Ordering::Relaxed), 4);
        });
    }

    #[test]
    fn bundle_columns_match_the_oracle_walk() {
        with_sample_backend(|backend| {
            let rows: Vec<WindowBundle> = backend.bundles().unwrap();
            let column: Vec<Fr> = rows.iter().map(|bundle| bundle.pc.to_field()).collect();
            let table = crate::JoltWitnessOracle::<Fr>::oracle_table(
                backend,
                JoltPolynomialId::Virtual(
                    jolt_claims::protocols::jolt::JoltVirtualPolynomial::UnexpandedPC,
                ),
            )
            .unwrap();
            assert_eq!(column, table);
        });
    }
}
