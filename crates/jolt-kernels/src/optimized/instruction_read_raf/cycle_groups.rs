use super::{ConstantSuffix, LookupIndexChunks};
use jolt_field::JoltField;
use jolt_poly::GruenSplitEqPolynomial;
use std::num::NonZeroU16;

const MAX_PAIRS: usize = 1024;
const SAMPLE_BLOCKS: usize = 32;

#[derive(Clone, Copy, PartialEq, Eq)]
struct PairKey {
    indices: [u128; 2],
    classes: u16,
}

impl PairKey {
    fn slot(self, slots: usize) -> usize {
        let [lo, hi] = self.indices;
        let mut hash = (lo as u64)
            ^ ((lo >> 64) as u64).rotate_left(11)
            ^ (hi as u64).rotate_left(23)
            ^ ((hi >> 64) as u64).rotate_left(37)
            ^ u64::from(self.classes).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        hash = (hash ^ (hash >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        hash = (hash ^ (hash >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((hash ^ (hash >> 31)) as usize) & (slots - 1)
    }
}

/// Only the first cycle round: both source rows and their claim bytes determine
/// all factors except the equality weight. Groups stay inside one Gruen block.
pub(super) struct CyclePairGroups<'a> {
    source: &'a LookupIndexChunks,
    claims: &'a ConstantSuffix<u8>,
    batch_size: usize,
}

impl<'a> CyclePairGroups<'a> {
    pub(super) fn new(
        source: &'a LookupIndexChunks,
        claims: &'a ConstantSuffix<u8>,
        prefix_pairs: usize,
        inner_len: usize,
    ) -> Option<Self> {
        let batch_size = inner_len.min(MAX_PAIRS);
        if prefix_pairs < SAMPLE_BLOCKS * batch_size {
            return None;
        }
        let groups = Self {
            source,
            claims,
            batch_size,
        };
        let blocks = prefix_pairs.div_ceil(batch_size);
        let mut cache = vec![None::<NonZeroU16>; 4 * batch_size];
        let mut hits = 0;
        let mut sampled = 0;
        for sample in 0..SAMPLE_BLOCKS {
            cache.fill(None);
            let start = sample * (blocks - 1) / (SAMPLE_BLOCKS - 1) * batch_size;
            let len = batch_size.min(prefix_pairs - start);
            sampled += len;
            for offset in 0..len {
                let key = groups.key(start + offset);
                let slot = key.slot(cache.len());
                if cache[slot].is_some_and(|previous| {
                    groups.key(start + usize::from(previous.get() - 1)) == key
                }) {
                    hits += 1;
                } else {
                    cache[slot] = NonZeroU16::new(offset as u16 + 1);
                }
            }
        }
        (4 * hits >= sampled).then_some(groups)
    }

    fn key(&self, row: usize) -> PairKey {
        PairKey {
            indices: [
                self.source.packed_index(2 * row),
                self.source.packed_index(2 * row + 1),
            ],
            classes: u16::from(self.claims[2 * row]) | (u16::from(self.claims[2 * row + 1]) << 8),
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
    cache: Vec<Option<NonZeroU16>>,
    pairs_seen: usize,
}

impl<F: JoltField, A> GroupedScratch<F, A> {
    fn new(inner: A, batch_size: usize) -> Self {
        Self {
            inner,
            rows: Vec::with_capacity(batch_size),
            weights: Vec::with_capacity(batch_size),
            cache: vec![None; 4 * batch_size],
            pairs_seen: 0,
        }
    }

    fn push(&mut self, groups: &CyclePairGroups<'_>, row: usize, weight: F) {
        self.pairs_seen += 1;
        let key = groups.key(row);
        let slot = key.slot(self.cache.len());
        if let Some(previous) = self.cache[slot] {
            let previous = usize::from(previous.get() - 1);
            if groups.key(self.rows[previous]) == key {
                self.weights[previous] += weight;
                return;
            }
        }
        self.rows.push(row);
        self.weights.push(weight);
        self.cache[slot] = NonZeroU16::new(self.rows.len() as u16);
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
        self.cache.fill(None);
        self.pairs_seen = 0;
    }
}
