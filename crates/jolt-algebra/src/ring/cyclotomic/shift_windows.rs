use super::*;
use jolt_field::WithCommitAccumulator;

/// Shifts summed per destination pass. Each pass streams the destination
/// once and reads one contiguous window per shift, so a batch of `n` shifts
/// costs `⌊n / GROUP⌋` destination read-modify-writes plus at most three for
/// the remainder, instead of `n`.
const GROUP: usize = 8;

/// Every negacyclic shift of one ring element, prepared for batched
/// accumulation into wide destinations.
///
/// Holds the [`WithCommitAccumulator::CommitLanes`] of
/// `[-a_0, …, -a_{D-1}, a_0, …, a_{D-1}]`, so coefficient `j` of `a · X^k` is
/// entry `D + j - k` for every `k < D`. A batch of shifts into one
/// destination then reads each shift as a contiguous window of half-width
/// lanes and sums up to eight of them per destination coefficient before
/// writing it back, instead of one full read-modify-write of the destination
/// per shift.
/// Loading costs two lane splits per coefficient, so a load pays off when the
/// element is accumulated into several destinations or with several shifts.
#[derive(Debug, Clone)]
pub struct NegacyclicShiftWindows<F: WithCommitAccumulator, const D: usize> {
    lanes: Vec<F::CommitLanes>,
}

impl<F: WithCommitAccumulator, const D: usize> Default for NegacyclicShiftWindows<F, D> {
    fn default() -> Self {
        Self {
            lanes: vec![F::CommitLanes::default(); 2 * D],
        }
    }
}

impl<F: WithCommitAccumulator, const D: usize> NegacyclicShiftWindows<F, D> {
    /// Replaces the held element with `src`.
    #[inline]
    pub fn load(&mut self, src: &CyclotomicRing<F, D>) {
        let (negative, positive) = self.lanes.split_at_mut(D);
        for ((negative, positive), &value) in negative.iter_mut().zip(positive).zip(&src.coeffs) {
            *negative = F::CommitLanes::from(-value);
            *positive = F::CommitLanes::from(value);
        }
    }

    /// `dst += a · Σ_{k ∈ shifts} X^k` for the held element `a`.
    ///
    /// Shifts may repeat and come in any order. Each shift is one addition
    /// against [`WithCommitAccumulator::MAX_COMMIT_ACCUMULATIONS`] for `dst`.
    ///
    /// # Panics
    ///
    /// Panics if any shift is at least `D`.
    #[inline]
    pub fn accumulate_shifts_into(
        &self,
        dst: &mut WideCyclotomicRing<F::Wide, D>,
        shifts: &[usize],
    ) {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: runtime feature detection guarantees AVX2.
            unsafe { self.accumulate_shifts_into_avx2(dst, shifts) };
            return;
        }
        self.accumulate_shifts_into_portable(dst, shifts);
    }

    /// Baseline x86-64 widens `u16` lanes with SSE2 unpacks; AVX2 widens
    /// eight lanes per `vpmovzxwd`.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn accumulate_shifts_into_avx2(
        &self,
        dst: &mut WideCyclotomicRing<F::Wide, D>,
        shifts: &[usize],
    ) {
        self.accumulate_shifts_into_portable(dst, shifts);
    }

    #[inline(always)]
    fn accumulate_shifts_into_portable(
        &self,
        dst: &mut WideCyclotomicRing<F::Wide, D>,
        shifts: &[usize],
    ) {
        // Flat lanes widen and add one vector at a time; per-coefficient lane
        // structs would make the vectorizer shuffle lanes across coefficients.
        let lanes = F::flatten_commit_lanes(&self.lanes);
        let out = F::flatten_wide_mut(&mut dst.coeffs);
        let len = out.len();
        debug_assert_eq!(lanes.len(), 2 * len);
        let width = len / D;
        let window = |shift: usize| {
            // `shift == D` would otherwise read the negative half as `-a`.
            assert!(
                shift < D,
                "negacyclic shift {shift} out of range for D = {D}"
            );
            &lanes[(D - shift) * width..][..len]
        };
        let mut rest = shifts;
        while let Some((group, tail)) = rest.split_first_chunk::<GROUP>() {
            add_windows(out, group.map(window));
            rest = tail;
        }
        // The remainder is below `GROUP`, so at most one pass of each
        // smaller power of two covers it.
        if let Some((group, tail)) = rest.split_first_chunk::<4>() {
            add_windows(out, group.map(window));
            rest = tail;
        }
        if let Some((group, tail)) = rest.split_first_chunk::<2>() {
            add_windows(out, group.map(window));
            rest = tail;
        }
        if let Some((group, _)) = rest.split_first_chunk::<1>() {
            add_windows(out, group.map(window));
        }
    }
}

/// `out[i] += Σ_w w[i]` over `G` lane windows as long as `out`.
#[inline(always)]
fn add_windows<const G: usize>(out: &mut [i32], windows: [&[u16]; G]) {
    let windows = windows.map(|window| &window[..out.len()]);
    for (i, out) in out.iter_mut().enumerate() {
        let mut sum = *out;
        for window in &windows {
            sum += i32::from(window[i]);
        }
        *out = sum;
    }
}
