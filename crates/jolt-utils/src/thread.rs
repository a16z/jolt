//! Rayon-backed threading utilities: background drops and deterministic-error
//! index-parallel collection.

use std::sync::Mutex;

use rayon::prelude::*;

/// Drops `data` in a background rayon task to avoid blocking the caller.
pub fn drop_in_background_thread<T: Send + 'static>(data: T) {
    rayon::spawn(move || drop(data));
}

/// The parallel scatter grain of [`par_collect_windows`]: big enough to
/// amortize rayon dispatch, small enough to load-balance skewed work.
const COLLECT_PAR_CHUNK: usize = 1 << 12;

/// Deterministic error latch for index-parallel collectors: keeps the
/// failure at the LOWEST index, so a failing input reports the same error
/// the sequential walk would hit first, independent of thread timing.
///
/// Blocking lock throughout: contention exists only on error paths, which
/// never overlap the happy path's cost (each worker chunk stops at its first
/// failure).
pub struct FirstErrorLatch<E> {
    slot: Mutex<Option<(usize, E)>>,
}

#[expect(
    clippy::unwrap_used,
    reason = "no lock user can panic while holding the latch"
)]
impl<E> FirstErrorLatch<E> {
    pub fn new() -> Self {
        Self {
            slot: Mutex::new(None),
        }
    }

    /// Record `failure` at `index` if it precedes the held one.
    pub fn record(&self, index: usize, failure: E) {
        let mut guard = self.slot.lock().unwrap();
        if guard.as_ref().is_none_or(|(held, _)| index < *held) {
            *guard = Some((index, failure));
        }
    }

    /// The lowest-index failure, if any worker recorded one.
    pub fn take(self) -> Option<E> {
        self.slot.into_inner().unwrap().map(|(_, failure)| failure)
    }
}

impl<E> Default for FirstErrorLatch<E> {
    fn default() -> Self {
        Self::new()
    }
}

/// In-place parallel collection of `window(0..count)` into a fresh vector:
/// workers write straight into the destination's spare capacity — no
/// per-thread segment buffers or concatenation (rayon's `Result` collect
/// loses indexedness and stages every segment). The lowest-index error wins
/// (deterministic across runs); on error the partially-written buffer is
/// abandoned without drops — sound because `V: Copy` rules out drop
/// obligations by construction.
pub fn par_collect_windows<V: Copy + Send, E: Send>(
    count: usize,
    window: impl Fn(usize) -> Result<V, E> + Send + Sync,
) -> Result<Vec<V>, E> {
    let mut out: Vec<V> = Vec::with_capacity(count);
    let spare = &mut out.spare_capacity_mut()[..count];
    let error = FirstErrorLatch::new();
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
                        error.record(base + offset, failure);
                        return;
                    }
                }
            }
        });
    if let Some(failure) = error.take() {
        return Err(failure);
    }
    // SAFETY: the error latch is empty, so every chunk ran to completion and
    // initialized all `count` slots of the spare capacity above.
    unsafe { out.set_len(count) };
    Ok(out)
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use super::*;

    #[test]
    fn latch_keeps_lowest_index() {
        let latch = FirstErrorLatch::new();
        latch.record(7, "seven");
        latch.record(3, "three");
        latch.record(5, "five");
        assert_eq!(latch.take(), Some("three"));
        assert_eq!(FirstErrorLatch::<&str>::new().take(), None);
    }

    #[test]
    fn par_collect_windows_collects_in_order() {
        // Spans multiple scatter chunks so cross-chunk indexing is exercised.
        let count = COLLECT_PAR_CHUNK * 2 + 17;
        let out: Vec<usize> = par_collect_windows(count, Ok::<usize, ()>).unwrap();
        assert_eq!(out.len(), count);
        assert!(out.iter().enumerate().all(|(index, &value)| index == value));
    }

    #[test]
    fn par_collect_windows_reports_lowest_index_error() {
        let count = COLLECT_PAR_CHUNK * 4;
        let result: Result<Vec<usize>, usize> = par_collect_windows(count, |index| {
            // Every chunk fails somewhere; the lowest failing index must win.
            if index % COLLECT_PAR_CHUNK == 13 {
                Err(index)
            } else {
                Ok(index)
            }
        });
        assert_eq!(result.unwrap_err(), 13);
    }
}
