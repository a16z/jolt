//! Direct-mapped detection of rows that repeat a recent key.
//!
//! Kernels whose per-row contribution is linear in a weight may merge rows
//! with equal keys and evaluate one representative. The cache only finds
//! candidates: callers compare full keys before merging, so a hash collision
//! leaves both rows as separate representatives and results stay exact.

use std::num::NonZeroU16;

const SAMPLE_BLOCKS: usize = 32;
/// Slots per cached entry; keeps the direct-mapped load at one quarter.
const SLOTS_PER_ENTRY: usize = 4;

/// Folds `words` and applies the splitmix64 finalizer.
#[inline]
pub(crate) fn repeat_hash(words: impl IntoIterator<Item = u64>) -> u64 {
    let mut hash = words.into_iter().fold(0, |hash: u64, word| {
        (hash.rotate_left(23) ^ word).wrapping_mul(0x9e37_79b9_7f4a_7c15)
    });
    hash = (hash ^ (hash >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    hash = (hash ^ (hash >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    hash ^ (hash >> 31)
}

pub(crate) struct RepeatCache<V> {
    slots: Box<[Option<V>]>,
}

impl<V> RepeatCache<V> {
    /// Sized for up to `entries` distinct keys between clears.
    pub(crate) fn new(entries: usize) -> Self {
        Self {
            slots: (0..(SLOTS_PER_ENTRY * entries).next_power_of_two())
                .map(|_| None)
                .collect(),
        }
    }

    #[inline]
    pub(crate) fn entry(&mut self, hash: u64) -> &mut Option<V> {
        let mask = self.slots.len() - 1;
        &mut self.slots[hash as usize & mask]
    }

    pub(crate) fn clear(&mut self) {
        self.slots.iter_mut().for_each(|slot| *slot = None);
    }

    pub(crate) fn drain(&mut self) -> impl Iterator<Item = V> + '_ {
        self.slots.iter_mut().filter_map(Option::take)
    }
}

impl RepeatCache<NonZeroU16> {
    /// Returns the cached representative when `same` accepts it. Otherwise
    /// caches `candidate` in its slot and returns `None`; a displaced
    /// representative stays valid for the caller, it is only no longer found.
    #[inline]
    pub(crate) fn find_or_insert(
        &mut self,
        hash: u64,
        candidate: usize,
        same: impl FnOnce(usize) -> bool,
    ) -> Option<usize> {
        let slot = self.entry(hash);
        if let Some(previous) = *slot {
            let previous = usize::from(previous.get() - 1);
            if same(previous) {
                return Some(previous);
            }
        }
        // An index past u16 is simply not cached: callers then keep it as
        // its own representative.
        *slot = u16::try_from(candidate + 1).ok().and_then(NonZeroU16::new);
        None
    }
}

/// Whether at least a quarter of the rows in up to 32 evenly spaced blocks of
/// `block_len` rows repeat a key cached earlier in their block. This only
/// chooses between a merged and a plain evaluation that are both exact.
pub(crate) fn repeats_often(
    len: usize,
    block_len: usize,
    hash: impl Fn(usize) -> u64,
    same: impl Fn(usize, usize) -> bool,
) -> bool {
    repeats_often_in_window(len, block_len, block_len, hash, same)
}

/// Samples complete windows with the caller's cache capacity. A cache can
/// recognize loops longer than its nominal entry budget because it has four
/// slots per entry; clearing the sample sooner hides those repeats.
pub(crate) fn repeats_often_in_window(
    len: usize,
    entries: usize,
    window_len: usize,
    hash: impl Fn(usize) -> u64,
    same: impl Fn(usize, usize) -> bool,
) -> bool {
    if entries == 0 || window_len == 0 || len < window_len {
        return false;
    }
    let blocks = len.div_ceil(window_len);
    // Spend the same sample budget on longer windows, rather than increasing
    // the serial work used to select an evaluation path.
    let samples = blocks.min((SAMPLE_BLOCKS * entries / window_len).max(1));
    let mut cache = RepeatCache::new(entries);
    let mut hits = 0;
    let mut sampled = 0;
    for sample in 0..samples {
        cache.clear();
        let start = sample * blocks / samples * window_len;
        for row in start..(start + window_len).min(len) {
            sampled += 1;
            hits += usize::from(
                cache
                    .find_or_insert(hash(row), row - start, |previous| {
                        same(start + previous, row)
                    })
                    .is_some(),
            );
        }
    }
    4 * hits >= sampled
}

#[cfg(test)]
mod tests {
    use super::{repeats_often, repeats_often_in_window, RepeatCache};

    #[test]
    fn window_sampling_preserves_cache_capacity() {
        let period = 3072;
        let sample = |entries| {
            repeats_often_in_window(
                1 << 17,
                entries,
                16384,
                |row| (row % period) as u64,
                |a, b| a % period == b % period,
            )
        };
        // The long window repeats each key, but the small direct-mapped cache
        // evicts every representative before the key appears again.
        assert!(sample(2048));
        assert!(!sample(64));
    }

    #[test]
    fn colliding_keys_stay_separate_representatives() {
        let mut cache = RepeatCache::new(1);
        let keys = [7u64, 9, 7, 9];
        let mut found = Vec::new();
        for (row, key) in keys.iter().enumerate() {
            // One hash for every key forces each lookup into the same slot.
            found.push(cache.find_or_insert(0, row, |previous| keys[previous] == *key));
        }
        assert_eq!(found, [None, None, None, None]);
        assert_eq!(
            cache.find_or_insert(0, 4, |previous| previous == 3),
            Some(3)
        );
    }

    #[test]
    fn sampled_repeat_rate_threshold() {
        let period = |period: usize| {
            move |len: usize| {
                repeats_often(
                    len,
                    64,
                    |row| (row % period) as u64,
                    |a, b| a % period == b % period,
                )
            }
        };
        // Keys below the 256 slots never collide. In 64-row blocks, period 48
        // repeats 16/64 rows and period 49 repeats 15/64.
        assert!(period(48)(1 << 12));
        assert!(!period(49)(1 << 12));
        assert!(period(1)(1 << 20));
        assert!(!period(1)(63));
    }
}
