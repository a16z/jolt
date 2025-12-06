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

/// A streaming schedule that uses exponentially increasing window sizes in the
/// first half, then single-round windows in the second half.
///
/// # Streaming Phase (first half)
///
/// Window sizes follow powers of two: 1, 2, 4, 8, ...
/// This creates windows at rounds 0, 1, 3, 7, 15, ... with sizes 1, 2, 4, 8, ...
/// The last window before the halfway point may be truncated.
///
/// # Linear Phase (second half)
///
/// After the switch-over point (at or just past the halfway mark), each round
/// is its own window with size 1. At this point, Az and Bz polynomials are
/// fully materialized in memory.
///
/// # Example
///
/// For `num_rounds = 16`:
/// - Halfway point: 8
/// - Streaming windows: [0,1), [1,3), [3,7), [7,8) with sizes 1, 2, 4, 1
/// - Switch-over at round 8
/// - Linear windows: [8,9), [9,10), ..., [15,16) each with size 1
#[derive(Debug, Clone, Allocative)]
pub struct HalfSplitSchedule {
    num_rounds: usize,
    switch_over_point: usize,
    window_starts: Vec<usize>,
}

impl HalfSplitSchedule {
    /// Creates a new `HalfSplitSchedule` for the given number of rounds.
    ///
    /// The schedule automatically computes exponentially increasing window sizes
    /// for the streaming phase (first half) and single-round windows for the
    /// linear phase (second half).
    pub fn new(num_rounds: usize) -> Self {
        let halfway = num_rounds / 2;
        let mut window_starts = Vec::new();

        // Generate exponentially increasing windows: 1, 2, 4, 8, ...
        // Window i starts at round (2^i - 1) with width 2^i
        let mut round = 0usize;
        let mut width = 1usize;
        while round < halfway {
            window_starts.push(round);
            let remaining = halfway - round;
            // Truncate the last window if it would exceed halfway
            let actual_width = width.min(remaining);
            round += actual_width;
            width *= 2;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_construction() {
        // num_rounds = 16, halfway = 8
        // Exponential windows: [0,1), [1,3), [3,7), [7,8)
        //   - round 0: width 1, next = 1
        //   - round 1: width 2, next = 3
        //   - round 3: width 4, next = 7
        //   - round 7: width would be 8, but 7+8=15 > 8, so truncated to 1, next = 8
        // switch_over_point = 8
        let schedule = HalfSplitSchedule::new(16);
        assert_eq!(schedule.num_rounds(), 16);
        assert_eq!(schedule.switch_over_point, 8);
        assert_eq!(
            schedule.window_starts,
            vec![0, 1, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        );
    }

    #[test]
    fn test_switch_over_point() {
        let schedule = HalfSplitSchedule::new(16);

        // Before switch-over (streaming phase)
        assert!(schedule.before_switch_over_point(0));
        assert!(schedule.before_switch_over_point(7));
        assert!(!schedule.before_switch_over_point(8));

        // At switch-over
        assert!(schedule.is_switch_over_point(8));
        assert!(!schedule.is_switch_over_point(7));
        assert!(!schedule.is_switch_over_point(9));

        // After switch-over (linear phase)
        assert!(schedule.after_switch_over_point(9));
        assert!(schedule.after_switch_over_point(15));
        assert!(!schedule.after_switch_over_point(8));
        assert!(!schedule.after_switch_over_point(7));
    }

    #[test]
    fn test_window_starts() {
        let schedule = HalfSplitSchedule::new(16);

        // Streaming phase - exponentially spaced windows
        assert!(schedule.is_window_start(0)); // width 1
        assert!(schedule.is_window_start(1)); // width 2
        assert!(!schedule.is_window_start(2)); // inside [1,3)
        assert!(schedule.is_window_start(3)); // width 4
        assert!(!schedule.is_window_start(4)); // inside [3,7)
        assert!(!schedule.is_window_start(5));
        assert!(!schedule.is_window_start(6));
        assert!(schedule.is_window_start(7)); // truncated width 1

        // After switch-over - every round is a window start
        assert!(schedule.is_window_start(8));
        assert!(schedule.is_window_start(9));
        assert!(schedule.is_window_start(15));
    }

    #[test]
    fn test_num_unbound_vars_exponential_windows() {
        let schedule = HalfSplitSchedule::new(16);

        // Window [0, 1): width 1
        assert_eq!(schedule.num_unbound_vars(0), 1);

        // Window [1, 3): width 2
        assert_eq!(schedule.num_unbound_vars(1), 2);
        assert_eq!(schedule.num_unbound_vars(2), 1);

        // Window [3, 7): width 4
        assert_eq!(schedule.num_unbound_vars(3), 4);
        assert_eq!(schedule.num_unbound_vars(4), 3);
        assert_eq!(schedule.num_unbound_vars(5), 2);
        assert_eq!(schedule.num_unbound_vars(6), 1);

        // Window [7, 8): truncated width 1
        assert_eq!(schedule.num_unbound_vars(7), 1);
    }

    #[test]
    fn test_num_unbound_vars_after_switchover() {
        let schedule = HalfSplitSchedule::new(16);

        // After switch-over, each window is size 1
        for round in 8..16 {
            assert_eq!(schedule.num_unbound_vars(round), 1);
        }
    }

    #[test]
    fn test_odd_num_rounds() {
        // num_rounds = 15, halfway = 7
        // Windows: [0,1), [1,3), [3,7) -> switch at 7
        let schedule = HalfSplitSchedule::new(15);
        assert_eq!(schedule.switch_over_point, 7);
        assert_eq!(
            schedule.window_starts,
            vec![0, 1, 3, 7, 8, 9, 10, 11, 12, 13, 14]
        );
    }

    #[test]
    fn test_small_num_rounds() {
        // num_rounds = 4, halfway = 2
        // Windows: [0,1), [1,2) -> switch at 2
        let schedule = HalfSplitSchedule::new(4);
        assert_eq!(schedule.switch_over_point, 2);
        assert_eq!(schedule.window_starts, vec![0, 1, 2, 3]);

        assert_eq!(schedule.num_unbound_vars(0), 1);
        assert_eq!(schedule.num_unbound_vars(1), 1);
        assert_eq!(schedule.num_unbound_vars(2), 1);
        assert_eq!(schedule.num_unbound_vars(3), 1);
    }

    #[test]
    fn test_large_num_rounds() {
        // num_rounds = 32, halfway = 16
        // Windows: [0,1), [1,3), [3,7), [7,15), [15,16) -> switch at 16
        let schedule = HalfSplitSchedule::new(32);
        assert_eq!(schedule.switch_over_point, 16);

        // Check window starts in streaming phase
        assert!(schedule.is_window_start(0));
        assert!(schedule.is_window_start(1));
        assert!(schedule.is_window_start(3));
        assert!(schedule.is_window_start(7));
        assert!(schedule.is_window_start(15));

        // Window [7, 15): width 8
        assert_eq!(schedule.num_unbound_vars(7), 8);
        assert_eq!(schedule.num_unbound_vars(10), 5);
        assert_eq!(schedule.num_unbound_vars(14), 1);

        // Window [15, 16): truncated width 1
        assert_eq!(schedule.num_unbound_vars(15), 1);
    }

    #[test]
    fn test_very_small_num_rounds() {
        // num_rounds = 2, halfway = 1
        // Windows: [0,1) -> switch at 1
        let schedule = HalfSplitSchedule::new(2);
        assert_eq!(schedule.switch_over_point, 1);
        assert_eq!(schedule.window_starts, vec![0, 1]);

        assert_eq!(schedule.num_unbound_vars(0), 1);
        assert_eq!(schedule.num_unbound_vars(1), 1);
    }
}
