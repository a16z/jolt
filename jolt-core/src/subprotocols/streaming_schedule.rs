use allocative::Allocative;

/// Trait for controlling the streaming prover's schedule during sumcheck.
///
/// The streaming prover trades memory for computation by computing prover messages
/// directly from the trace without materializing large intermediate polynomials.
/// This trait defines when to switch from streaming mode to linear-time / materialized mode,
/// and how to partition rounds into "windows" for the streaming data structure.
pub trait StreamingSchedule: Send + Sync {
    /// Returns `true` if this round is the switch-over round where we finally
    /// materialise the sumcheck polynomials (Az, Bz) in memory.
    ///
    /// Prior to this round, prover messages are computed directly from the trace
    /// using a streaming data structure.
    fn is_switch_over_point(&self, round: usize) -> bool;

    /// Returns `true` if the given round is before the switch-over point
    /// (i.e., still in streaming mode).
    fn before_switch_over_point(&self, round: usize) -> bool;

    /// Returns `true` if the given round is after the switch-over point
    /// (i.e., in linear-time mode with materialized polynomials).
    fn after_switch_over_point(&self, round: usize) -> bool;

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
    fn is_switch_over_point(&self, round: usize) -> bool {
        self.switch_over_point == round
    }

    fn after_switch_over_point(&self, round: usize) -> bool {
        round > self.switch_over_point
    }

    fn before_switch_over_point(&self, round: usize) -> bool {
        round < self.switch_over_point
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

    fn after_switch_over_point(&self, round: usize) -> bool {
        round > 0
    }

    fn before_switch_over_point(&self, _round: usize) -> bool {
        false
    }

    fn is_switch_over_point(&self, round: usize) -> bool {
        round == 0
    }

    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn num_unbound_vars(&self, _round: usize) -> usize {
        1
    }
}

/// Experimental streaming schedule with configurable window sizes.
///
/// This schedule allows full control over the streaming phase by specifying
/// exact window sizes. After the streaming windows, it switches to linear mode.
///
/// # Example: 3 Passes for 25 Variables
///
/// For 2^24 cycles (25 variables with streaming index):
/// - `num_rounds = 24` (excluding univariate-skip first round)
/// - `streaming_window_sizes = [1, 2, 9]` (12 vars in 3 passes)
///
/// ```text
/// Pass 1: round 0,  size 1  → bind streaming index
/// Pass 2: round 1,  size 2  → bind vars 1-2
/// Pass 3: round 3,  size 9  → bind vars 3-11
/// Materialize at round 12 (4th pass through trace)
/// Linear: rounds 12-23, each size 1
/// ```
///
/// # Cost Analysis
///
/// For each streaming window of size `w` at round `i`:
/// - Grid computation: O(3^w) per (x_out, x_in) pair
/// - Number of pairs: O(2^(remaining vars))
/// - Trace accesses: full pass through trace
///
/// Larger windows amortize trace access cost but increase grid computation.
#[derive(Debug, Clone, Allocative)]
pub struct ExperimentalSchedule {
    num_rounds: usize,
    switch_over_point: usize,
    window_starts: Vec<usize>,
    window_sizes: Vec<usize>,
}

impl ExperimentalSchedule {
    /// Creates a new schedule with explicit streaming window sizes.
    ///
    /// # Arguments
    ///
    /// * `num_rounds` - Total number of sumcheck rounds
    /// * `streaming_window_sizes` - Sizes of each streaming window (one per pass)
    ///
    /// # Panics
    ///
    /// Panics if the sum of window sizes exceeds `num_rounds`.
    pub fn new(num_rounds: usize, streaming_window_sizes: &[usize]) -> Self {
        let total_streaming_vars: usize = streaming_window_sizes.iter().sum();
        assert!(
            total_streaming_vars <= num_rounds,
            "Streaming window sizes sum to {total_streaming_vars} but only {num_rounds} rounds available"
        );

        let mut window_starts = Vec::new();
        let mut window_sizes = Vec::new();

        // Build streaming windows
        let mut round = 0usize;
        for &size in streaming_window_sizes {
            window_starts.push(round);
            window_sizes.push(size);
            round += size;
        }

        let switch_over_point = round;

        // Add single-round windows for linear phase
        for i in switch_over_point..num_rounds {
            window_starts.push(i);
            window_sizes.push(1);
        }

        Self {
            num_rounds,
            switch_over_point,
            window_starts,
            window_sizes,
        }
    }

    /// Creates a schedule that limits to N streaming passes, using cost-optimal
    /// window sizes from HalfSplitSchedule's formula.
    ///
    /// # Arguments
    ///
    /// * `num_rounds` - Total number of sumcheck rounds
    /// * `max_streaming_passes` - Maximum number of passes through trace before materializing
    /// * `degree` - Polynomial degree (2 for multiquadratic)
    pub fn with_max_passes(num_rounds: usize, max_streaming_passes: usize, degree: usize) -> Self {
        let ratio = HalfSplitSchedule::compute_window_ratio(degree);

        let mut streaming_window_sizes = Vec::new();
        let mut round = 0usize;

        for _ in 0..max_streaming_passes {
            if round >= num_rounds {
                break;
            }
            let optimal_size = ((round as f64) * ratio).round() as usize;
            let size = optimal_size.max(1).min(num_rounds - round);
            streaming_window_sizes.push(size);
            round += size;
        }

        Self::new(num_rounds, &streaming_window_sizes)
    }

    /// Creates a schedule with exactly 3 streaming passes that cover half the rounds.
    ///
    /// Window sizes grow exponentially: `[1, r, r²]` where `1 + r + r² = H` and
    /// `H = num_rounds / 2`. This gives a larger growth factor than cost-optimal,
    /// trading more grid computation per pass for fewer total passes.
    ///
    /// # Warning
    ///
    /// **This schedule is designed for degree-2 (multiquadratic) sumchecks only.**
    /// For higher degrees, the cost-optimal ratio is smaller and 3 passes may not
    /// be sufficient to cover half the rounds efficiently.
    ///
    /// # Arguments
    ///
    /// * `num_rounds` - Total number of sumcheck rounds
    ///
    /// # Example
    ///
    /// For 24 rounds (25 variables including streaming index):
    /// - H = 12, solve 1 + r + r² = 12 → r ≈ 2.85
    /// - Window sizes: [1, 3, 8]
    /// - Materialize at round 12
    ///
    /// ```text
    /// Pass 1: round 0, size 1  → streaming index
    /// Pass 2: round 1, size 3  → vars 1-3
    /// Pass 3: round 4, size 8  → vars 4-11
    /// Materialize at round 12 (4th pass, O(n) memory)
    /// Linear: rounds 12-23, each size 1
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `num_rounds < 6` (need at least 3 rounds for streaming phase).
    pub fn three_pass_half_split(num_rounds: usize) -> Self {
        assert!(
            num_rounds >= 6,
            "three_pass_half_split requires at least 6 rounds, got {num_rounds}"
        );

        let half = num_rounds / 2;

        // Solve 1 + r + r² = H for r
        // r² + r + (1 - H) = 0
        // r = (-1 + sqrt(1 - 4(1-H))) / 2 = (-1 + sqrt(4H - 3)) / 2
        let discriminant = 4.0 * (half as f64) - 3.0;
        let r = (-1.0 + discriminant.sqrt()) / 2.0;

        // Window sizes: 1, round(r), remainder to hit exactly half
        let size1 = 1usize;
        let size2 = r.round() as usize;
        // Adjust size3 to hit exactly half (absorb rounding errors from r²)
        let size3 = half.saturating_sub(size1 + size2);

        // Sanity check: sizes should be increasing
        debug_assert!(
            size1 <= size2 && size2 <= size3,
            "Window sizes should be increasing: [{size1}, {size2}, {size3}]"
        );

        let streaming_window_sizes = vec![size1, size2, size3];

        #[cfg(debug_assertions)]
        {
            let total: usize = streaming_window_sizes.iter().sum();
            eprintln!(
                "three_pass_half_split: {num_rounds} rounds, H={half}, r={r:.2}, sizes={streaming_window_sizes:?}, total={total}"
            );
        }

        Self::new(num_rounds, &streaming_window_sizes)
    }

    /// Creates a schedule optimized for a target number of streaming variables.
    ///
    /// Tries to bind `target_streaming_vars` variables in `max_passes` passes.
    ///
    /// # Arguments
    ///
    /// * `num_rounds` - Total number of sumcheck rounds
    /// * `target_streaming_vars` - How many variables to bind before materializing
    /// * `max_passes` - Maximum number of streaming passes
    pub fn with_streaming_target(
        num_rounds: usize,
        target_streaming_vars: usize,
        max_passes: usize,
    ) -> Self {
        let target = target_streaming_vars.min(num_rounds);

        // Distribute variables across passes
        // Strategy: start small, grow exponentially, but cap at target
        let mut streaming_window_sizes = Vec::new();
        let mut total = 0usize;

        // Use sizes: 1, 2, 4, 8, ... until we hit target or max_passes
        let mut size = 1usize;
        for _ in 0..max_passes {
            if total >= target {
                break;
            }
            let actual_size = size.min(target - total);
            streaming_window_sizes.push(actual_size);
            total += actual_size;
            size *= 2;
        }

        Self::new(num_rounds, &streaming_window_sizes)
    }

    /// Returns the streaming window sizes for debugging.
    pub fn streaming_window_sizes(&self) -> Vec<usize> {
        let streaming_count = self
            .window_starts
            .iter()
            .take_while(|&&r| r < self.switch_over_point)
            .count();
        self.window_sizes[..streaming_count].to_vec()
    }

    /// Returns summary statistics for the schedule.
    pub fn summary(&self) -> String {
        let streaming_sizes = self.streaming_window_sizes();
        let total_streaming: usize = streaming_sizes.iter().sum();
        format!(
            "ExperimentalSchedule: {} rounds, {} streaming passes ({} vars: {:?}), materialize at round {}, {} linear rounds",
            self.num_rounds,
            streaming_sizes.len(),
            total_streaming,
            streaming_sizes,
            self.switch_over_point,
            self.num_rounds - self.switch_over_point
        )
    }
}

impl StreamingSchedule for ExperimentalSchedule {
    fn is_switch_over_point(&self, round: usize) -> bool {
        self.switch_over_point == round
    }

    fn after_switch_over_point(&self, round: usize) -> bool {
        round > self.switch_over_point
    }

    fn before_switch_over_point(&self, round: usize) -> bool {
        round < self.switch_over_point
    }

    fn is_window_start(&self, round: usize) -> bool {
        self.window_starts.binary_search(&round).is_ok()
    }

    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn num_unbound_vars(&self, round: usize) -> usize {
        let idx = match self.window_starts.binary_search(&round) {
            Ok(idx) => idx,
            Err(idx) => idx.saturating_sub(1),
        };

        let window_start = self.window_starts[idx];
        let window_size = self.window_sizes[idx];
        window_size - (round - window_start)
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

        // Before switch-over (streaming phase)
        assert!(schedule.before_switch_over_point(0));
        assert!(schedule.before_switch_over_point(8));
        assert!(!schedule.before_switch_over_point(10));

        // At switch-over
        assert!(schedule.is_switch_over_point(10));
        assert!(!schedule.is_switch_over_point(8));
        assert!(!schedule.is_switch_over_point(11));

        // After switch-over (linear phase)
        assert!(schedule.after_switch_over_point(11));
        assert!(schedule.after_switch_over_point(19));
        assert!(!schedule.after_switch_over_point(10));
        assert!(!schedule.after_switch_over_point(8));
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
