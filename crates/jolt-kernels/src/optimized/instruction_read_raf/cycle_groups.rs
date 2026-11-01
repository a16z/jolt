use std::num::NonZeroU16;

use jolt_field::JoltField;
use jolt_poly::GruenSplitEqPolynomial;

use super::{ClaimBytes, LookupIndexChunks};
use crate::optimized::support::{repeat_hash, repeats_often, RepeatCache};

const MAX_PAIRS: usize = 1024;

#[derive(Clone, Copy, PartialEq, Eq)]
struct PairKey {
    indices: [u128; 2],
    classes: u16,
}

impl PairKey {
    fn hash(self) -> u64 {
        let [lo, hi] = self.indices;
        repeat_hash([
            lo as u64,
            (lo >> 64) as u64,
            hi as u64,
            (hi >> 64) as u64,
            u64::from(self.classes),
        ])
    }
}

/// Only the first cycle round: both source rows and their claim bytes determine
/// all factors except the equality weight. Groups stay inside one Gruen block.
pub(super) struct CyclePairGroups<'a> {
    source: &'a LookupIndexChunks,
    claims: &'a ClaimBytes,
    batch_size: usize,
}

impl<'a> CyclePairGroups<'a> {
    pub(super) fn new(
        source: &'a LookupIndexChunks,
        claims: &'a ClaimBytes,
        prefix_pairs: usize,
        inner_len: usize,
    ) -> Option<Self> {
        let groups = Self {
            source,
            claims,
            batch_size: inner_len.min(MAX_PAIRS),
        };
        repeats_often(
            prefix_pairs,
            groups.batch_size,
            |row| groups.key(row).hash(),
            |a, b| groups.key(a) == groups.key(b),
        )
        .then_some(groups)
    }

    fn key(&self, row: usize) -> PairKey {
        PairKey {
            indices: [
                self.source.packed_index(2 * row),
                self.source.packed_index(2 * row + 1),
            ],
            classes: u16::from(self.claims.0[2 * row])
                | (u16::from(self.claims.0[2 * row + 1]) << 8),
        }
    }

    pub(super) fn fold<F, A, B>(
        &self,
        eq: &GruenSplitEqPolynomial<F>,
        len: usize,
        make_inner: impl Fn() -> A + Send + Sync,
        inner_step: impl Fn(&mut A, usize, F) + Send + Sync,
        outer_step: impl Fn(usize, F, A) -> B + Send + Sync,
        merge: impl Fn(B, B) -> B + Send + Sync,
    ) -> B
    where
        F: JoltField,
        A: Send,
        B: Send,
    {
        eq.par_fold_out_in_prefix(
            len,
            || GroupedScratch::new(make_inner(), self.batch_size),
            |scratch, row, _, weight| {
                scratch.push(self, row, weight);
                if scratch.pairs_seen == self.batch_size {
                    scratch.flush(&inner_step);
                }
            },
            |out, weight, mut scratch| {
                scratch.flush(&inner_step);
                outer_step(out, weight, scratch.inner)
            },
            merge,
        )
    }
}

struct GroupedScratch<F, A> {
    inner: A,
    rows: Vec<usize>,
    weights: Vec<F>,
    cache: RepeatCache<NonZeroU16>,
    pairs_seen: usize,
}

impl<F: JoltField, A> GroupedScratch<F, A> {
    fn new(inner: A, batch_size: usize) -> Self {
        Self {
            inner,
            rows: Vec::with_capacity(batch_size),
            weights: Vec::with_capacity(batch_size),
            cache: RepeatCache::new(batch_size),
            pairs_seen: 0,
        }
    }

    fn push(&mut self, groups: &CyclePairGroups<'_>, row: usize, weight: F) {
        self.pairs_seen += 1;
        let key = groups.key(row);
        let rows = &self.rows;
        let representative = self
            .cache
            .find_or_insert(key.hash(), rows.len(), |previous| {
                groups.key(rows[previous]) == key
            });
        if let Some(previous) = representative {
            self.weights[previous] += weight;
        } else {
            self.rows.push(row);
            self.weights.push(weight);
        }
    }

    fn flush(&mut self, inner_step: &impl Fn(&mut A, usize, F)) {
        if self.rows.is_empty() {
            return;
        }
        for (&row, &weight) in self.rows.iter().zip(&self.weights) {
            inner_step(&mut self.inner, row, weight);
        }
        self.rows.clear();
        self.weights.clear();
        self.cache.clear();
        self.pairs_seen = 0;
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    #[cfg(feature = "akita")]
    use jolt_witness::witnesses::FusedInc;

    use super::super::{ClaimBytes, InstructionCycleRow, LookupIndexChunks};
    use super::CyclePairGroups;

    fn engages(period: usize) -> bool {
        let cycles = 1 << 14;
        let rows: Vec<_> = (0..cycles)
            .map(|cycle| {
                InstructionCycleRow::new(
                    (cycle % period) as u128,
                    None,
                    false,
                    0,
                    None,
                    #[cfg(feature = "akita")]
                    FusedInc::default(),
                )
            })
            .collect();
        let source = LookupIndexChunks::new(Arc::new(rows.into()), 16, 4);
        let claims = ClaimBytes(Arc::new(vec![0u8; cycles].into()));
        CyclePairGroups::new(&source, &claims, cycles / 2, 1 << 6).is_some()
    }

    #[test]
    fn grouping_engages_only_for_repeated_pairs() {
        assert!(engages(10));
        assert!(!engages(1 << 14));
    }
}
