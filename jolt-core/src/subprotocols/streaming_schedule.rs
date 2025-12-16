use allocative::Allocative;

/// Trait for controlling the streaming prover's schedule during sumcheck.
///
/// The streaming prover trades memory for computation by computing prover messages
/// directly from the trace without materializing large intermediate polynomials.
/// This trait defines when to switch from streaming mode to linear-time / materialized mode,
/// and how to partition rounds into "windows" for the streaming data structure.
pub trait StreamingSchedule: Send + Sync + Allocative {
    /// Returns the round at which we switch from streaming to linear-time mode.
    /// Prior to this round, prover messages are computed directly from the trace.
    /// At and after this round, polynomials are materialized in memory.
    fn switch_over_point(&self) -> usize;

    /// Returns `true` if we are starting a new streaming window at this round.
    ///
    /// At the start of each window, the streaming data structure (multiquadratic
    /// polynomial over the window variables) is recomputed from the trace.
    fn is_window_start(&self, round: usize) -> bool;

    /// Returns the total number of rounds for sumcheck.
    fn num_rounds(&self) -> usize;

    /// Returns the number of unbound variables remaining in the current window.
    fn num_unbound_vars(&self, round: usize) -> usize;
}

/// A streaming schedule that uses cost-optimal window sizes in the first half,
/// then single-round windows in the second half.
///
/// # Streaming Phase (first half)
///
/// Window sizes are chosen to keep the grid computation cost approximately constant.
/// For a sumcheck of degree `d`, the cost for a window of size `w` starting at
/// round `i` is proportional to:
///
/// ```text
/// (d+1)^w / 2^(w+i) × T
/// ```
///
/// Setting this ≈ 1 gives the optimal window size: `w ≈ ln(2) / ln((d+1)/2) × i`.
///
/// For degree 2 (multiquadratic), this ratio is ≈ 1.71, giving windows at
/// rounds 0, 1, 3, 8, 22, ... with sizes 1, 2, 5, 14, 38, ...
///
/// The last window before the halfway point may be truncated.
///
/// # Linear Phase (second half)
///
/// After the switch-over point (at or just past the halfway mark), each round
/// is its own window with size 1. At this point, Az and Bz polynomials are
/// fully materialized in memory.
///
/// # Cost Analysis
///
/// The ratio `ln(2) / ln((d+1)/2)` ensures each streaming window has
/// approximately the same computational cost, avoiding the exponential blowup
/// that would occur with fixed doubling (1, 2, 4, 8, ...).
///
/// | Degree | Ratio  | Example windows (sizes)   |
/// |--------|--------|---------------------------|
/// | 2      | 1.71   | 1, 2, 5, 14, 38, ...      |
/// | 3      | 1.00   | 1, 1, 2, 3, 4, ...        |
/// | 4      | 0.76   | 1, 1, 1, 2, 2, 3, ...     |
///
/// # Example
///
/// For `num_rounds = 20` with degree 2:
/// - Halfway point: 10
/// - Streaming windows: [0,1), [1,3), [3,8), [8,10) with sizes 1, 2, 5, 2
/// - Switch-over at round 10
/// - Linear windows: [10,11), [11,12), ..., [19,20) each with size 1
#[derive(Debug, Clone, Allocative)]
pub struct HalfSplitSchedule {
    num_rounds: usize,
    switch_over_point: usize,
    window_starts: Vec<usize>,
}

impl HalfSplitSchedule {
    /// Computes the optimal window ratio for a given polynomial degree.
    ///
    /// For a sumcheck of degree `d`, the cost is `(d+1)^w / 2^(w+i)`.
    /// Setting this ≈ 1 gives the ratio `ln(2) / ln((d+1)/2)`.
    #[inline]
    pub fn compute_window_ratio(degree: usize) -> f64 {
        let d_plus_1 = (degree + 1) as f64;
        2.0_f64.ln() / (d_plus_1 / 2.0).ln()
    }

    /// Computes the optimal window size for a given round and degree.
    ///
    /// Returns `max(1, round(ratio × round))` where `ratio = ln(2) / ln((d+1)/2)`.
    #[inline]
    pub fn optimal_window_size(round: usize, degree: usize) -> usize {
        let ratio = Self::compute_window_ratio(degree);
        (((round as f64) * ratio).round() as usize).max(1)
    }

    /// Creates a new `HalfSplitSchedule` for the given number of rounds and
    /// polynomial degree.
    ///
    /// The schedule automatically computes cost-optimal window sizes for the
    /// streaming phase (first half) and single-round windows for the linear
    /// phase (second half).
    ///
    /// # Arguments
    ///
    /// * `num_rounds` - Total number of sumcheck rounds
    /// * `degree` - Degree of the sumcheck polynomial (e.g., 2 for multiquadratic)
    ///
    /// # Window Sizing
    ///
    /// Window sizes follow `w = round(ratio × i)` where:
    /// - `i` is the current round
    /// - `ratio = ln(2) / ln((degree+1)/2)`
    ///
    /// This keeps the grid computation cost `(degree+1)^w / 2^(w+i)` approximately
    /// constant per window.
    pub fn new(num_rounds: usize, degree: usize) -> Self {
        let window_ratio = Self::compute_window_ratio(degree);
        let halfway = num_rounds / 2;
        let mut window_starts = Vec::new();

        // Generate cost-optimal windows using w = round(ratio * i)
        // This keeps (d+1)^w / 2^(w+i) ≈ 1 for each window
        let mut round = 0usize;
        while round < halfway {
            window_starts.push(round);

            // Optimal window size: w = round(ratio * round), minimum 1
            let optimal_width = ((round as f64) * window_ratio).round() as usize;
            let width = optimal_width.max(1);

            let remaining = halfway - round;
            // Truncate the last window if it would exceed halfway
            let actual_width = width.min(remaining);
            round += actual_width;
        }

        // Switch-over point is where we transition to linear mode
        let switch_over_point = round;

        // Add single-round windows from switch-over to end
        for i in switch_over_point..num_rounds {
            window_starts.push(i);
        }

        Self {
            num_rounds,
            switch_over_point,
            window_starts,
        }
    }
}

impl StreamingSchedule for HalfSplitSchedule {
    fn switch_over_point(&self) -> usize {
        self.switch_over_point
    }

    fn is_window_start(&self, round: usize) -> bool {
        // Use binary search since window_starts is sorted
        self.window_starts.binary_search(&round).is_ok()
    }

    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn num_unbound_vars(&self, round: usize) -> usize {
        // Binary search to find the window containing this round
        let current_idx = match self.window_starts.binary_search(&round) {
            Ok(idx) => idx,                    // Exact match: round is a window start
            Err(idx) => idx.saturating_sub(1), // Round is within window starting at idx-1
        };

        // Get the next window start (or num_rounds if this is the last window)
        let next_window_start = self
            .window_starts
            .get(current_idx + 1)
            .copied()
            .unwrap_or(self.num_rounds);

        next_window_start - round
    }
}

/// A schedule that disables streaming and runs all sumcheck rounds in
/// the linear-time mode.
#[derive(Debug, Clone, Allocative)]
pub struct LinearOnlySchedule {
    num_rounds: usize,
}

impl LinearOnlySchedule {
    pub fn new(num_rounds: usize) -> Self {
        Self { num_rounds }
    }
}

impl StreamingSchedule for LinearOnlySchedule {
    fn is_window_start(&self, _round: usize) -> bool {
        true
    }

    fn switch_over_point(&self) -> usize {
        0
    }

    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn num_unbound_vars(&self, _round: usize) -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_window_ratio_and_size() {
        // ratio = ln(2) / ln((d+1)/2)
        assert!((HalfSplitSchedule::compute_window_ratio(2) - 1.71).abs() < 0.01);
        assert!((HalfSplitSchedule::compute_window_ratio(3) - 1.0).abs() < 0.01);

        // For degree 2: w = round(1.71 * i), minimum 1
        assert_eq!(HalfSplitSchedule::optimal_window_size(0, 2), 1);
        assert_eq!(HalfSplitSchedule::optimal_window_size(1, 2), 2);
        assert_eq!(HalfSplitSchedule::optimal_window_size(3, 2), 5);
    }

    #[test]
    fn test_schedule_construction() {
        // 20 rounds, degree 2: windows at 0,1,3,8 then switch at 10
        let schedule = HalfSplitSchedule::new(20, 2);
        assert_eq!(schedule.num_rounds(), 20);
        assert_eq!(schedule.switch_over_point(), 10);
        assert_eq!(
            schedule.window_starts,
            vec![0, 1, 3, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        );

        // Window starts
        assert!(schedule.is_window_start(0));
        assert!(schedule.is_window_start(3));
        assert!(!schedule.is_window_start(2));
        assert!(!schedule.is_window_start(5));

        // Unbound vars decrease within window
        assert_eq!(schedule.num_unbound_vars(3), 5);
        assert_eq!(schedule.num_unbound_vars(7), 1);

        // After switch-over: all size 1
        for round in 10..20 {
            assert_eq!(schedule.num_unbound_vars(round), 1);
        }
    }

    #[test]
    fn test_edge_cases() {
        // Very small: 2 rounds
        let small = HalfSplitSchedule::new(2, 2);
        assert_eq!(small.switch_over_point(), 1);

        // Large: 40 rounds
        let large = HalfSplitSchedule::new(40, 2);
        assert_eq!(large.switch_over_point(), 20);
        assert_eq!(large.num_unbound_vars(8), 12); // truncated window
    }

    #[test]
    fn test_cost_model() {
        // Cost factor (d+1)^w / 2^(w+i) should stay near 1
        let schedule = HalfSplitSchedule::new(100, 2);
        let mut round = 0usize;
        let mut idx = 0usize;
        while round < 50 {
            let next = schedule.window_starts.get(idx + 1).copied().unwrap_or(50);
            let w = next - round;
            if w > 0 && next < 50 {
                let cost = 3.0_f64.powi(w as i32) / 2.0_f64.powi((w + round) as i32);
                assert!((0.5..=2.0).contains(&cost));
            }
            round = next;
            idx += 1;
        }
    }
}
