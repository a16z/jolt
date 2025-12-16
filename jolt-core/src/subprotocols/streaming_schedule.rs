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

    // Degree 2 is the common case (multiquadratic for Spartan outer sumcheck)
    const DEGREE_2: usize = 2;

    #[test]
    fn test_window_ratio() {
        // For degree d, ratio = ln(2) / ln((d+1)/2)
        let ratio_deg2 = HalfSplitSchedule::compute_window_ratio(2);
        assert!((ratio_deg2 - 1.70951).abs() < 0.001); // ln(2)/ln(1.5) ≈ 1.71

        let ratio_deg3 = HalfSplitSchedule::compute_window_ratio(3);
        assert!((ratio_deg3 - 1.0).abs() < 0.001); // ln(2)/ln(2) = 1.0

        let ratio_deg4 = HalfSplitSchedule::compute_window_ratio(4);
        assert!((ratio_deg4 - 0.756).abs() < 0.01); // ln(2)/ln(2.5) ≈ 0.756
    }

    #[test]
    fn test_optimal_window_size() {
        // For degree 2: w = round(1.71 * i), minimum 1
        assert_eq!(HalfSplitSchedule::optimal_window_size(0, DEGREE_2), 1); // round(0) = 0, min 1
        assert_eq!(HalfSplitSchedule::optimal_window_size(1, DEGREE_2), 2); // round(1.71) = 2
        assert_eq!(HalfSplitSchedule::optimal_window_size(3, DEGREE_2), 5); // round(5.13) = 5
        assert_eq!(HalfSplitSchedule::optimal_window_size(8, DEGREE_2), 14); // round(13.68) = 14
        assert_eq!(HalfSplitSchedule::optimal_window_size(22, DEGREE_2), 38); // round(37.61) = 38
    }

    #[test]
    fn test_basic_construction() {
        // num_rounds = 20, degree = 2, halfway = 10
        // Cost-optimal windows:
        //   - round 0: w = max(1, round(0)) = 1, next = 1
        //   - round 1: w = round(1.71) = 2, next = 3
        //   - round 3: w = round(5.13) = 5, next = 8
        //   - round 8: w = round(13.68) = 14, but 8+14=22 > 10, truncated to 2, next = 10
        // switch_over_point = 10
        let schedule = HalfSplitSchedule::new(20, DEGREE_2);
        assert_eq!(schedule.num_rounds(), 20);
        assert_eq!(schedule.switch_over_point, 10);
        assert_eq!(
            schedule.window_starts,
            vec![0, 1, 3, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        );
    }

    #[test]
    fn test_switch_over_point() {
        let schedule = HalfSplitSchedule::new(20, DEGREE_2);
        let switch_over = schedule.switch_over_point();

        assert_eq!(switch_over, 10);
    }

    #[test]
    fn test_window_starts() {
        let schedule = HalfSplitSchedule::new(20, DEGREE_2);

        // Streaming phase - cost-optimal windows at 0, 1, 3, 8
        assert!(schedule.is_window_start(0)); // width 1
        assert!(schedule.is_window_start(1)); // width 2
        assert!(!schedule.is_window_start(2)); // inside [1,3)
        assert!(schedule.is_window_start(3)); // width 5
        assert!(!schedule.is_window_start(4)); // inside [3,8)
        assert!(!schedule.is_window_start(5));
        assert!(!schedule.is_window_start(6));
        assert!(!schedule.is_window_start(7));
        assert!(schedule.is_window_start(8)); // truncated

        // After switch-over - every round is a window start
        assert!(schedule.is_window_start(10));
        assert!(schedule.is_window_start(11));
        assert!(schedule.is_window_start(19));
    }

    #[test]
    fn test_num_unbound_vars_optimal_windows() {
        let schedule = HalfSplitSchedule::new(20, DEGREE_2);

        // Window [0, 1): width 1
        assert_eq!(schedule.num_unbound_vars(0), 1);

        // Window [1, 3): width 2
        assert_eq!(schedule.num_unbound_vars(1), 2);
        assert_eq!(schedule.num_unbound_vars(2), 1);

        // Window [3, 8): width 5
        assert_eq!(schedule.num_unbound_vars(3), 5);
        assert_eq!(schedule.num_unbound_vars(4), 4);
        assert_eq!(schedule.num_unbound_vars(5), 3);
        assert_eq!(schedule.num_unbound_vars(6), 2);
        assert_eq!(schedule.num_unbound_vars(7), 1);

        // Window [8, 10): truncated width 2
        assert_eq!(schedule.num_unbound_vars(8), 2);
        assert_eq!(schedule.num_unbound_vars(9), 1);
    }

    #[test]
    fn test_num_unbound_vars_after_switchover() {
        let schedule = HalfSplitSchedule::new(20, DEGREE_2);

        // After switch-over, each window is size 1
        for round in 10..20 {
            assert_eq!(schedule.num_unbound_vars(round), 1);
        }
    }

    #[test]
    fn test_small_num_rounds() {
        // num_rounds = 4, halfway = 2
        // Windows: [0,1), [1,2) -> switch at 2
        let schedule = HalfSplitSchedule::new(4, DEGREE_2);
        assert_eq!(schedule.switch_over_point, 2);
        assert_eq!(schedule.window_starts, vec![0, 1, 2, 3]);

        assert_eq!(schedule.num_unbound_vars(0), 1);
        assert_eq!(schedule.num_unbound_vars(1), 1);
        assert_eq!(schedule.num_unbound_vars(2), 1);
        assert_eq!(schedule.num_unbound_vars(3), 1);
    }

    #[test]
    fn test_large_num_rounds() {
        // num_rounds = 40, halfway = 20
        // Cost-optimal windows:
        //   - round 0: w = 1, next = 1
        //   - round 1: w = 2, next = 3
        //   - round 3: w = 5, next = 8
        //   - round 8: w = 14, next = 22 > 20, truncated to 12, next = 20
        // switch_over_point = 20
        let schedule = HalfSplitSchedule::new(40, DEGREE_2);
        assert_eq!(schedule.switch_over_point, 20);

        // Check window starts in streaming phase
        assert!(schedule.is_window_start(0));
        assert!(schedule.is_window_start(1));
        assert!(schedule.is_window_start(3));
        assert!(schedule.is_window_start(8));

        // Window [3, 8): width 5
        assert_eq!(schedule.num_unbound_vars(3), 5);
        assert_eq!(schedule.num_unbound_vars(5), 3);
        assert_eq!(schedule.num_unbound_vars(7), 1);

        // Window [8, 20): truncated width 12
        assert_eq!(schedule.num_unbound_vars(8), 12);
        assert_eq!(schedule.num_unbound_vars(15), 5);
        assert_eq!(schedule.num_unbound_vars(19), 1);
    }

    #[test]
    fn test_very_small_num_rounds() {
        // num_rounds = 2, halfway = 1
        // Windows: [0,1) -> switch at 1
        let schedule = HalfSplitSchedule::new(2, DEGREE_2);
        assert_eq!(schedule.switch_over_point, 1);
        assert_eq!(schedule.window_starts, vec![0, 1]);

        assert_eq!(schedule.num_unbound_vars(0), 1);
        assert_eq!(schedule.num_unbound_vars(1), 1);
    }

    #[test]
    fn test_cost_factors_near_one() {
        // Verify that the cost factor (d+1)^w / 2^(w+i) stays near 1 for degree 2
        // (except for the last window which may be truncated)
        let schedule = HalfSplitSchedule::new(100, DEGREE_2);
        let halfway = 50;

        let mut round = 0usize;
        let mut window_idx = 0usize;
        while round < halfway {
            let next_start = schedule
                .window_starts
                .get(window_idx + 1)
                .copied()
                .unwrap_or(halfway);
            let w = next_start - round;

            // Skip the last window before switch-over (may be truncated)
            let is_last_streaming_window = next_start >= halfway;

            if w > 0 && !is_last_streaming_window {
                // For degree 2: cost = 3^w / 2^(w+i)
                let cost_factor = 3.0_f64.powi(w as i32) / 2.0_f64.powi((w + round) as i32);
                // Allow cost factor between 0.5 and 2.0 (close to 1)
                assert!(
                    (0.5..=2.0).contains(&cost_factor),
                    "Cost factor {cost_factor} out of range for round {round} with window size {w}"
                );
            }

            round = next_start;
            window_idx += 1;
        }
    }

    #[test]
    fn test_different_degrees() {
        // Degree 3: ratio = 1.0, so windows grow linearly
        let schedule_deg3 = HalfSplitSchedule::new(20, 3);
        // Windows should be at 0, 1, 2, 4, 7, ... (slower growth)
        assert!(schedule_deg3.is_window_start(0));
        assert!(schedule_deg3.is_window_start(1));

        // Degree 4: ratio = 0.76, so windows grow even slower
        let schedule_deg4 = HalfSplitSchedule::new(20, 4);
        assert!(schedule_deg4.is_window_start(0));
        assert!(schedule_deg4.is_window_start(1));
    }
}
