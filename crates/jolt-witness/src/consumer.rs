//! The fused trace→bundles pass: one row walk drives a statically-known set
//! of consumers.

use std::ops::Range;
use std::sync::Arc;

use jolt_program::preprocess::JoltProgramPreprocessing;
use jolt_riscv::JoltTraceRow as TraceRow;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::bundle::WitnessBundle;
use crate::witnesses::WitnessEnv;
use crate::{WitnessError, JOLT_VM_LABEL};

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
    let extract_span = tracing::info_span!(
        "stream_extract",
        bundle = core::any::type_name::<C::Witness>(),
        rows = rows.len()
    )
    .entered();
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
    drop(extract_span);
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

/// Sequential row access, with an optional random-access fast path.
pub trait RowSource: Sync {
    /// Visits the half-open cycle `range` in order as buffers of at most
    /// `chunk_size` rows; `[0, T)` today, segments later.
    fn visit_chunks(
        &self,
        range: Range<usize>,
        chunk_size: usize,
        visitor: &mut ChunkVisitor<'_>,
    ) -> Result<(), WitnessError>;

    /// Returns shared random access when the source can provide it.
    fn random_access(&self) -> Option<RandomAccessRows> {
        None
    }
}

/// Shared random access to compact rows and their extraction context.
#[derive(Clone)]
pub struct RandomAccessRows {
    rows: Arc<Vec<TraceRow>>,
    cycles: usize,
    preprocessing: Arc<JoltProgramPreprocessing>,
    padding: TraceRow,
}

impl RandomAccessRows {
    pub(crate) fn new(
        rows: Arc<Vec<TraceRow>>,
        cycles: usize,
        preprocessing: Arc<JoltProgramPreprocessing>,
    ) -> Result<Self, WitnessError> {
        if rows.len() > cycles {
            return Err(WitnessError::InvalidWitnessData {
                label: JOLT_VM_LABEL,
                reason: format!(
                    "physical trace has {} rows but the cycle domain has {cycles}",
                    rows.len()
                ),
            });
        }
        Ok(Self {
            rows,
            cycles,
            preprocessing,
            padding: TraceRow::default(),
        })
    }

    /// Padded cycle-domain size.
    pub fn cycles(&self) -> usize {
        self.cycles
    }

    /// Extracts one bundle with padding and one-row lookahead semantics.
    #[inline]
    pub fn window<B: WitnessBundle>(&self, index: usize) -> Result<B, WitnessError> {
        let current = self.rows.get(index).unwrap_or(&self.padding);
        let next =
            (index + 1 < self.cycles).then(|| self.rows.get(index + 1).unwrap_or(&self.padding));
        B::from_row(current, next, &WitnessEnv::new(&self.preprocessing))
    }
}

/// Collects one bundle per cycle directly into an indexed destination.
pub fn collect_bundles_par<B: WitnessBundle + Copy + Send>(
    access: &RandomAccessRows,
    cycles: usize,
) -> Result<Vec<B>, WitnessError> {
    collect_bundles_par_map(access, cycles, |bundle: B| bundle)
}

/// Fuse a representation change into random-access collection so callers
/// do not stage the wider bundle form.
pub fn collect_bundles_par_map<B: WitnessBundle, T: Copy + Send>(
    access: &RandomAccessRows,
    cycles: usize,
    map: impl Fn(B) -> T + Send + Sync,
) -> Result<Vec<T>, WitnessError> {
    let window = |index| access.window::<B>(index).map(&map);
    #[cfg(feature = "parallel")]
    return jolt_utils::par_collect_windows(cycles, window);
    #[cfg(not(feature = "parallel"))]
    (0..cycles).map(window).collect()
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
    // Out-of-range requests fall through to the validated chunk walk.
    if let Some(access) = source.random_access() {
        if cycles <= access.cycles() {
            let window = |index| access.window::<B>(index);
            #[cfg(feature = "parallel")]
            return (0..cycles).into_par_iter().map(window).collect();
            #[cfg(not(feature = "parallel"))]
            return (0..cycles).map(window).collect();
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
    use std::sync::Arc;

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

            let access = backend.random_access().unwrap();
            let packed: Vec<u64> =
                collect_bundles_par_map(&access, 4, |bundle: WindowBundle| bundle.pc.0).unwrap();
            assert_eq!(
                packed,
                routed.iter().map(|bundle| bundle.pc.0).collect::<Vec<_>>()
            );
        });
    }

    #[test]
    fn random_access_rejects_rows_beyond_cycle_domain() {
        with_sample_backend(|backend| {
            let rows = Arc::new(vec![TraceRow::default(); 2]);
            assert!(matches!(
                RandomAccessRows::new(rows, 1, Arc::clone(&backend.preprocessing)),
                Err(WitnessError::InvalidWitnessData { .. })
            ));
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
