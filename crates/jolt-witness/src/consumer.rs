//! The fused trace→bundles pass: one row walk drives a statically-known set
//! of consumers.

use jolt_program::execution::TraceRow;

use crate::bundle::WitnessBundle;
use crate::witnesses::WitnessEnv;
use crate::WitnessError;

/// Half-open range of cycles walked by one pass; `[0, T)` today, segments
/// later.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CycleRange {
    pub start: usize,
    pub end: usize,
}

impl CycleRange {
    pub const fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    pub const fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// One consumer of a bundle stream. `Option<C>` is also a consumer:
/// membership in a set is static, presence is runtime.
pub trait StreamConsumer: Send + Sync {
    type Witness: WitnessBundle;

    fn consume(&mut self, chunk: &[Self::Witness]);
}

impl<C: StreamConsumer> StreamConsumer for Option<C> {
    type Witness = C::Witness;

    fn consume(&mut self, chunk: &[Self::Witness]) {
        if let Some(consumer) = self {
            consumer.consume(chunk);
        }
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

fn deliver<C: StreamConsumer>(
    consumer: &mut C,
    rows: &[TraceRow],
    next_after: Option<&TraceRow>,
    env: &WitnessEnv<'_>,
) -> Result<(), WitnessError> {
    let mut bundles = Vec::with_capacity(rows.len());
    for (index, row) in rows.iter().enumerate() {
        let next = rows.get(index + 1).or(next_after);
        bundles.push(C::Witness::from_row(row, next, env)?);
    }
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
/// later. Random access is deliberately inexpressible.
pub trait RowSource {
    /// Visits `range` in order as buffers of at most `chunk_size` rows.
    fn visit_chunks(
        &self,
        range: CycleRange,
        chunk_size: usize,
        visitor: &mut ChunkVisitor<'_>,
    ) -> Result<(), WitnessError>;
}

/// The fused pass: walk `range` once and deliver each chunk to every
/// consumer in the set.
pub fn stream_witnesses<S: RowSource + ?Sized, C: ConsumerSet>(
    source: &S,
    range: CycleRange,
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

    fn collect_with_chunk_size(chunk_size: usize) -> Vec<WindowBundle> {
        with_sample_backend(|backend| {
            let mut consumers = (CollectBundles::<WindowBundle>::default(),);
            stream_witnesses(backend, CycleRange::new(0, 4), chunk_size, &mut consumers).unwrap();
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
    fn one_walk_feeds_every_consumer_and_absent_slots_skip() {
        with_sample_backend(|backend| {
            let mut consumers = (
                CollectBundles::<WindowBundle>::default(),
                Some(CollectBundles::<WindowBundle>::default()),
                None::<CollectBundles<WindowBundle>>,
            );
            stream_witnesses(backend, CycleRange::new(0, 4), 2, &mut consumers).unwrap();
            let first = consumers.0.into_rows();
            assert_eq!(first.len(), 4);
            assert_eq!(consumers.1.unwrap().into_rows(), first);
            assert!(consumers.2.is_none());
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
