use allocative::Allocative;

// TODO: Clean up this streaming schedule docstring.
pub trait StreamingSchedule: Send + Sync {
    fn is_streaming(&self, round: usize) -> bool;
    /// Returns true if we are starting a new streaming window.
    /// This will lead to recomputation of the streaming data structure
    /// storing the multi-variable polynomial used to computer prover messages.  
    fn is_window_start(&self, round: usize) -> bool;
    /// Returns true of round is the first round of linear proving
    fn is_first_linear(&self, round: usize) -> bool;
    /// Get the total number of rounds for sumcheck
    fn num_rounds(&self) -> usize;
    /// Get  the number of unbound variables in given round
    /// If still in streaming phase, this should be in terms of how many rounds left in
    /// given window.
    fn num_unbound_vars(&self, round: usize) -> usize;
}

#[derive(Debug, Clone, Allocative)]
pub struct HalfSplitSchedule {
    num_rounds: usize,
    constant_window_width: usize,
    linear_start: usize,
    window_starts: Vec<usize>,
}

impl HalfSplitSchedule {
    pub fn new(num_rounds: usize, window_width: usize) -> Self {
        let linear_start = num_rounds.div_ceil(2);

        let window_starts = (0..linear_start).step_by(window_width).collect();

        Self {
            num_rounds,
            constant_window_width: window_width,
            linear_start,
            window_starts,
        }
    }
}

impl StreamingSchedule for HalfSplitSchedule {
    fn is_streaming(&self, round: usize) -> bool {
        round < self.linear_start
    }

    fn is_window_start(&self, round: usize) -> bool {
        self.window_starts.contains(&round)
    }

    fn is_first_linear(&self, round: usize) -> bool {
        round == self.linear_start
    }

    fn num_rounds(&self) -> usize {
        self.num_rounds
    }
    fn num_unbound_vars(&self, round: usize) -> usize {
        if round >= self.num_rounds {
            return 0;
        }
        if self.is_streaming(round) {
            // Find which window this round belongs to
            // and how many rounds are left in that window

            // Find the next window start after this round (or linear_start)
            let next_boundary = self
                .window_starts
                .iter()
                .find(|&&start| start > round)
                .copied()
                .unwrap_or(self.linear_start);

            // Number of unbound vars = rounds left in current window
            next_boundary - round
        } else {
            // In linear phase: standard sumcheck, one variable at a time
            // Number of unbound = total remaining rounds
            self.num_rounds - round
        }
    }
}

/// Streaming schedule where window sizes increase as 1, 2, 3, ... until the
/// streaming phase (first half of the rounds) is filled. The final window is
/// truncated so that the total number of streaming rounds is still roughly half.
#[derive(Debug, Clone, Allocative)]
pub struct IncreasingWindowSchedule {
    pub(crate) num_rounds: usize,
    pub(crate) linear_start: usize,
    pub(crate) window_starts: Vec<usize>,
}

impl IncreasingWindowSchedule {
    pub fn new(num_rounds: usize) -> Self {
        let linear_start = num_rounds.div_ceil(2);

        let mut window_starts = Vec::new();
        let mut round = 0usize;
        let mut width = 1usize;
        while round < linear_start {
            window_starts.push(round);
            let remaining = linear_start - round;
            let w = core::cmp::min(width, remaining);
            round += w;
            width += 1;
        }

        Self {
            num_rounds,
            linear_start,
            window_starts,
        }
    }
}

impl StreamingSchedule for IncreasingWindowSchedule {
    fn is_streaming(&self, round: usize) -> bool {
        round < self.linear_start
    }

    fn is_window_start(&self, round: usize) -> bool {
        self.window_starts.contains(&round)
    }

    fn is_first_linear(&self, round: usize) -> bool {
        round == self.linear_start
    }

    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn num_unbound_vars(&self, round: usize) -> usize {
        if round >= self.num_rounds {
            return 0;
        }
        if self.is_streaming(round) {
            let next_boundary = self
                .window_starts
                .iter()
                .find(|&&start| start > round)
                .copied()
                .unwrap_or(self.linear_start);
            next_boundary - round
        } else {
            self.num_rounds - round
        }
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
    fn is_streaming(&self, _round: usize) -> bool {
        false
    }

    fn is_window_start(&self, _round: usize) -> bool {
        false
    }

    fn is_first_linear(&self, round: usize) -> bool {
        round == 0
    }

    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn num_unbound_vars(&self, round: usize) -> usize {
        self.num_rounds.saturating_sub(round)
        //if round >= self.num_rounds {
        //    0
        //} else {
        //    self.num_rounds - round
        //}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test basic schedule properties with 10 rounds
    #[test]
    fn test_half_split_schedule_basic() {
        let schedule = HalfSplitSchedule::new(10, 2);

        // First 5 rounds are streaming
        assert!(schedule.is_streaming(0));
        assert!(schedule.is_streaming(4));
        assert!(!schedule.is_streaming(5));
        assert!(!schedule.is_streaming(9));

        // Window starts at 0, 2, 4
        assert!(schedule.is_window_start(0));
        assert!(schedule.is_window_start(2));
        assert!(schedule.is_window_start(4));
        assert!(!schedule.is_window_start(1));
        assert!(!schedule.is_window_start(3));

        // First linear round is 5
        assert!(schedule.is_first_linear(5));
        assert!(!schedule.is_first_linear(4));
        assert!(!schedule.is_first_linear(6));
    }

    /// Test schedule with 8 rounds and window width 2
    ///
    /// # Schedule Layout (8 rounds, window_width=2):
    /// ```text
    /// Rounds:     0   1   2   3   4   5   6   7
    /// Phase:      [--Streaming--] [---Linear---]
    /// Windows:    [W0---] [W1---]
    /// Window#:    0       1       
    /// ```
    ///
    /// Streaming phase: rounds 0-3 (first half)
    /// Linear phase: rounds 4-7 (second half)
    #[test]
    fn test_8_rounds_window_2() {
        let schedule = HalfSplitSchedule::new(8, 2);

        assert_eq!(schedule.num_rounds(), 8);
        assert_eq!(schedule.linear_start, 4);
        assert_eq!(schedule.window_starts, vec![0, 2]);

        // Phase checks
        for round in 0..=3 {
            assert!(
                schedule.is_streaming(round),
                "Round {round} should be streaming",
            );
        }
        for round in 4..=7 {
            assert!(
                !schedule.is_streaming(round),
                "Round {round} should be linear",
            );
        }

        // Window start checks
        assert!(schedule.is_window_start(0), "Round 0 starts window 0");
        assert!(!schedule.is_window_start(1), "Round 1 is mid-window");
        assert!(schedule.is_window_start(2), "Round 2 starts window 1");
        assert!(!schedule.is_window_start(3), "Round 3 is mid-window");

        // First linear check
        assert!(schedule.is_first_linear(4), "Round 4 is first linear");
        for round in [0, 1, 2, 3, 5, 6, 7] {
            assert!(
                !schedule.is_first_linear(round),
                "Round {round} is not first linear",
            );
        }
    }

    /// Test num_unbound_vars with 8 rounds and window width 2
    ///
    /// # Unbound Variables Per Round:
    /// ```text
    /// Round 0 (streaming, window start): 2 unbound (rounds 0,1 left in window)
    /// Round 1 (streaming):                1 unbound (round 1 left in window)
    /// Round 2 (streaming, window start): 2 unbound (rounds 2,3 left in window)
    /// Round 3 (streaming):                1 unbound (round 3 left in window)
    /// Round 4 (linear):                   4 unbound (4 rounds left total)
    /// Round 5 (linear):                   3 unbound (3 rounds left total)
    /// Round 6 (linear):                   2 unbound (2 rounds left total)
    /// Round 7 (linear):                   1 unbound (1 round left total)
    /// ```
    #[test]
    fn test_num_unbound_vars_8_rounds_window_3() {
        let schedule = HalfSplitSchedule::new(8, 2);

        // Streaming phase unbound vars
        assert_eq!(
            schedule.num_unbound_vars(0),
            2,
            "Round 0: start of window 0"
        );
        assert_eq!(schedule.num_unbound_vars(1), 1, "Round 1: end of window 0");
        assert_eq!(
            schedule.num_unbound_vars(2),
            2,
            "Round 2: start of window 1"
        );
        assert_eq!(schedule.num_unbound_vars(3), 1, "Round 3: end of window 1");

        // Linear phase unbound vars
        assert_eq!(
            schedule.num_unbound_vars(4),
            4,
            "Round 4: 4 rounds remaining"
        );
        assert_eq!(
            schedule.num_unbound_vars(5),
            3,
            "Round 5: 3 rounds remaining"
        );
        assert_eq!(
            schedule.num_unbound_vars(6),
            2,
            "Round 6: 2 rounds remaining"
        );
        assert_eq!(
            schedule.num_unbound_vars(7),
            1,
            "Round 7: 1 round remaining"
        );
    }

    /// Test schedule with 8 rounds and window width 3
    ///
    /// # Schedule Layout:
    /// ```text
    /// Rounds:     0   1   2   3   4   5   6   7
    /// Phase:      [--Streaming--] [---Linear---]
    /// Windows:    [W0-------] [W1]
    /// ```
    ///
    /// Window 0: rounds 0-2 (3 rounds)
    /// Window 1: round 3 only (truncated at linear boundary)
    #[test]
    fn test_8_rounds_window_3() {
        let schedule = HalfSplitSchedule::new(8, 3);

        assert_eq!(schedule.window_starts, vec![0, 3]);

        // Unbound vars in streaming phase
        assert_eq!(
            schedule.num_unbound_vars(0),
            3,
            "Round 0: 3 rounds in window"
        );
        assert_eq!(schedule.num_unbound_vars(1), 2, "Round 1: 2 rounds left");
        assert_eq!(schedule.num_unbound_vars(2), 1, "Round 2: 1 round left");
        assert_eq!(schedule.num_unbound_vars(3), 1, "Round 3: truncated window");

        // Unbound vars in linear phase
        assert_eq!(schedule.num_unbound_vars(4), 4);
        assert_eq!(schedule.num_unbound_vars(7), 1);
    }

    /// Test edge case: window width equals streaming phase length
    #[test]
    fn test_single_window() {
        let schedule = HalfSplitSchedule::new(8, 4);

        // Only one window in streaming phase
        assert_eq!(schedule.window_starts, vec![0]);

        assert!(schedule.is_window_start(0));
        assert!(!schedule.is_window_start(1));
        assert!(!schedule.is_window_start(2));
        assert!(!schedule.is_window_start(3));

        // All streaming rounds in single window
        assert_eq!(schedule.num_unbound_vars(0), 4);
        assert_eq!(schedule.num_unbound_vars(1), 3);
        assert_eq!(schedule.num_unbound_vars(2), 2);
        assert_eq!(schedule.num_unbound_vars(3), 1);
    }

    /// Test edge case: window width larger than streaming phase
    #[test]
    fn test_oversized_window() {
        let schedule = HalfSplitSchedule::new(8, 10);

        // Still only one window, even though width > streaming phase
        assert_eq!(schedule.window_starts, vec![0]);

        // Unbound vars limited by linear_start boundary
        assert_eq!(schedule.num_unbound_vars(0), 4, "Limited by linear_start");
        assert_eq!(schedule.num_unbound_vars(1), 3);
        assert_eq!(schedule.num_unbound_vars(2), 2);
        assert_eq!(schedule.num_unbound_vars(3), 1);
    }

    /// Test odd number of rounds
    #[test]
    fn test_odd_rounds() {
        let schedule = HalfSplitSchedule::new(7, 2);

        // (7 + 1) / 2 = 4, so streaming is 0-3, linear is 4-6
        assert_eq!(schedule.linear_start, 4);

        assert!(schedule.is_streaming(3));
        assert!(!schedule.is_streaming(4));
        assert_eq!(schedule.num_rounds(), 7);
    }

    /// Test schedule properties are consistent
    #[test]
    fn test_schedule_invariants() {
        for num_rounds in [4, 8, 16, 32] {
            for window_width in [1, 2, 3, 4, 8] {
                let schedule = HalfSplitSchedule::new(num_rounds, window_width);

                // First window always starts at 0
                assert!(schedule.is_window_start(0));

                // Linear start is roughly half
                assert_eq!(schedule.linear_start, num_rounds.div_ceil(2));
                //assert_eq!(schedule.linear_start, (num_rounds + 1) / 2);

                // All window starts are in streaming phase
                for &start in &schedule.window_starts {
                    assert!(schedule.is_streaming(start));
                }

                // Unbound vars decrease monotonically within windows
                for round in 0..schedule.linear_start - 1 {
                    if !schedule.is_window_start(round + 1) {
                        assert!(
                            schedule.num_unbound_vars(round) > schedule.num_unbound_vars(round + 1),
                            "Unbound vars should decrease within window"
                        );
                    }
                }

                // Linear phase unbound vars decrease by 1 each round
                for round in schedule.linear_start..num_rounds - 1 {
                    assert_eq!(
                        schedule.num_unbound_vars(round) - schedule.num_unbound_vars(round + 1),
                        1,
                        "Linear phase should decrease by 1 each round"
                    );
                }
            }
        }
    }
}
