use allocative::Allocative;
//TODO: Robustify half split
pub trait StreamingSchedule: Send + Sync {
    /// If this round is the switch-over round then we finally
    /// materialise the sum-check polynomials in memory.
    /// Prior, to this round, the streaming data structure
    /// use to respond to verifier messages is constructed directly
    /// from the trace.
    fn is_switch_over_point(&self, round: usize) -> bool;
    fn before_switch_over_point(&self, round: usize) -> bool;
    fn after_switch_over_point(&self, round: usize) -> bool;

    /// Returns true if we are starting a new streaming window.
    /// This will lead to recomputation of the streaming data structure
    /// storing the multi-variable polynomial used to computer prover messages.  
    fn is_window_start(&self, round: usize) -> bool;

    /// Get the total number of rounds for sumcheck
    fn num_rounds(&self) -> usize;

    /// Get  the number of unbound variables in given window
    fn num_unbound_vars(&self, round: usize) -> usize;
}

#[derive(Debug, Clone, Allocative)]
pub struct HalfSplitSchedule {
    num_rounds: usize,
    constant_window_width: usize,
    switch_over_point: usize,
    window_starts: Vec<usize>,
}

impl HalfSplitSchedule {
    pub fn new(num_rounds: usize, window_width: usize) -> Self {
        let m = num_rounds / 2;
        let mut window_starts = Vec::new();

        // Generate constant-width windows until we exceed m
        let mut j = 0;
        loop {
            let window_start = j * window_width;
            if window_start >= m {
                break;
            }
            window_starts.push(window_start);
            j += 1;
        }

        // Switch-over point is the first window that would exceed m
        let switch_over_point = j * window_width;

        // Add single-round windows from switch-over to end
        for i in switch_over_point..num_rounds {
            window_starts.push(i);
        }

        Self {
            num_rounds,
            constant_window_width: window_width,
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
        self.window_starts.contains(&round)
    }

    fn num_rounds(&self) -> usize {
        self.num_rounds
    }
    fn num_unbound_vars(&self, round: usize) -> usize {
        // Find the index of the window start that is <= round
        let current_idx = self
            .window_starts
            .iter()
            .rposition(|&w| w <= round)
            .expect("no window start found for round");

        // Get the next window start value
        let next_window_start = if current_idx + 1 < self.window_starts.len() {
            self.window_starts[current_idx + 1]
        } else {
            // If this is the last window, assume num_rounds or some default
            self.num_rounds
        };

        // Return the difference
        next_window_start - round
    }
}

//#[derive(Debug, Clone, Allocative)]
//pub struct IncreasingWindowSchedule {
//    pub(crate) num_rounds: usize,
//    pub(crate) linear_start: usize,
//    pub(crate) window_starts: Vec<usize>,
//}
//
//impl IncreasingWindowSchedule {
//    pub fn new(num_rounds: usize) -> Self {
//        let linear_start = num_rounds.div_ceil(2);
//
//        let mut window_starts = Vec::new();
//        let mut round = 0usize;
//        let mut width = 1usize;
//        while round < linear_start {
//            window_starts.push(round);
//            let remaining = linear_start - round;
//            let w = core::cmp::min(width, remaining);
//            round += w;
//            width += 1;
//        }
//
//        Self {
//            num_rounds,
//            linear_start,
//            window_starts,
//        }
//    }
//}
//
//impl StreamingSchedule for IncreasingWindowSchedule {
//    fn is_streaming(&self, round: usize) -> bool {
//        round < self.linear_start
//    }
//
//    fn is_window_start(&self, round: usize) -> bool {
//        self.window_starts.contains(&round)
//    }
//
//    fn is_first_linear(&self, round: usize) -> bool {
//        round == self.linear_start
//    }
//
//    fn num_rounds(&self) -> usize {
//        self.num_rounds
//    }
//
//    fn num_unbound_vars(&self, round: usize) -> usize {
//        if round >= self.num_rounds {
//            return 0;
//        }
//        if self.is_streaming(round) {
//            let next_boundary = self
//                .window_starts
//                .iter()
//                .find(|&&start| start > round)
//                .copied()
//                .unwrap_or(self.linear_start);
//            next_boundary - round
//        } else {
//            self.num_rounds - round
//        }
//    }
//}

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
        let schedule = HalfSplitSchedule::new(16, 3);
        assert_eq!(schedule.num_rounds(), 16);
        assert_eq!(schedule.switch_over_point, 9);
        assert_eq!(
            schedule.window_starts,
            vec![0, 3, 6, 9, 10, 11, 12, 13, 14, 15]
        );
    }

    #[test]
    fn test_switch_over_point() {
        let schedule = HalfSplitSchedule::new(16, 3);

        // Before switch-over
        assert!(schedule.before_switch_over_point(0));
        assert!(schedule.before_switch_over_point(8));
        assert!(!schedule.before_switch_over_point(9));

        // At switch-over
        assert!(schedule.is_switch_over_point(9));
        assert!(!schedule.is_switch_over_point(8));
        assert!(!schedule.is_switch_over_point(10));

        // After switch-over
        assert!(schedule.after_switch_over_point(10));
        assert!(schedule.after_switch_over_point(15));
        assert!(!schedule.after_switch_over_point(9));
        assert!(!schedule.after_switch_over_point(8));
    }

    #[test]
    fn test_window_starts() {
        let schedule = HalfSplitSchedule::new(16, 3);

        // First half - constant width windows
        assert!(schedule.is_window_start(0));
        assert!(schedule.is_window_start(3));
        assert!(schedule.is_window_start(6));
        assert!(!schedule.is_window_start(1));
        assert!(!schedule.is_window_start(2));
        assert!(!schedule.is_window_start(4));

        // After switch-over - every round is a window start
        assert!(schedule.is_window_start(9));
        assert!(schedule.is_window_start(10));
        assert!(schedule.is_window_start(11));
        assert!(schedule.is_window_start(15));
    }

    #[test]
    fn test_num_unbound_vars_first_window() {
        let schedule = HalfSplitSchedule::new(16, 3);

        // Window [0, 3): 3 unbound vars initially
        assert_eq!(schedule.num_unbound_vars(0), 3);
        assert_eq!(schedule.num_unbound_vars(1), 2);
        assert_eq!(schedule.num_unbound_vars(2), 1);
    }

    #[test]
    fn test_num_unbound_vars_middle_window() {
        let schedule = HalfSplitSchedule::new(16, 3);

        // Window [3, 6): 3 unbound vars
        assert_eq!(schedule.num_unbound_vars(3), 3);
        assert_eq!(schedule.num_unbound_vars(4), 2);
        assert_eq!(schedule.num_unbound_vars(5), 1);

        // Window [6, 9): 3 unbound vars
        assert_eq!(schedule.num_unbound_vars(6), 3);
        assert_eq!(schedule.num_unbound_vars(7), 2);
        assert_eq!(schedule.num_unbound_vars(8), 1);
    }

    #[test]
    fn test_num_unbound_vars_after_switchover() {
        let schedule = HalfSplitSchedule::new(16, 3);

        // After switch-over, each window is size 1
        assert_eq!(schedule.num_unbound_vars(9), 1);
        assert_eq!(schedule.num_unbound_vars(10), 1);
        assert_eq!(schedule.num_unbound_vars(11), 1);
        assert_eq!(schedule.num_unbound_vars(14), 1);

        // Last round
        assert_eq!(schedule.num_unbound_vars(15), 1);
    }

    #[test]
    fn test_odd_num_rounds() {
        let schedule = HalfSplitSchedule::new(15, 3);

        // linear_start = 15.div_ceil(2) = 8
        // switch_over_point = 9
        assert_eq!(schedule.switch_over_point, 9);
        assert_eq!(schedule.window_starts, vec![0, 3, 6, 9, 10, 11, 12, 13, 14]);
    }

    #[test]
    fn test_small_num_rounds() {
        let schedule = HalfSplitSchedule::new(4, 2);

        assert_eq!(schedule.switch_over_point, 2);
        assert_eq!(schedule.window_starts, vec![0, 2, 3]);

        assert_eq!(schedule.num_unbound_vars(0), 2);
        assert_eq!(schedule.num_unbound_vars(1), 1);
        assert_eq!(schedule.num_unbound_vars(2), 1);
        assert_eq!(schedule.num_unbound_vars(3), 1);
    }

    #[test]
    fn test_large_window_width() {
        let schedule = HalfSplitSchedule::new(20, 10);

        assert_eq!(schedule.switch_over_point, 10);
        assert_eq!(
            schedule.window_starts,
            vec![0, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        );

        // First window covers rounds 0-9
        assert_eq!(schedule.num_unbound_vars(0), 10);
        assert_eq!(schedule.num_unbound_vars(5), 5);
        assert_eq!(schedule.num_unbound_vars(9), 1);

        // Second window is just round 10
        assert_eq!(schedule.num_unbound_vars(10), 1);
    }
}
