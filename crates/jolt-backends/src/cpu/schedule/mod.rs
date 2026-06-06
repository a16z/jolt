pub trait StreamingSchedule: Send + Sync {
    fn switch_over_point(&self) -> usize;

    fn is_window_start(&self, round: usize) -> bool;

    fn num_rounds(&self) -> usize;

    fn num_unbound_vars(&self, round: usize) -> usize;
}

#[derive(Debug, Clone)]
pub struct HalfSplitSchedule {
    num_rounds: usize,
    switch_over_point: usize,
    window_starts: Vec<usize>,
}

impl HalfSplitSchedule {
    #[inline]
    pub fn compute_window_ratio(degree: usize) -> f64 {
        let d_plus_1 = (degree + 1) as f64;
        2.0_f64.ln() / (d_plus_1 / 2.0).ln()
    }

    #[inline]
    pub fn optimal_window_size(round: usize, degree: usize) -> usize {
        let ratio = Self::compute_window_ratio(degree);
        (((round as f64) * ratio).round() as usize).max(1)
    }

    pub fn new(num_rounds: usize, degree: usize) -> Self {
        let window_ratio = Self::compute_window_ratio(degree);
        let halfway = num_rounds / 2;
        let mut window_starts = Vec::new();

        let mut round = 0usize;
        while round < halfway {
            window_starts.push(round);

            let optimal_width = ((round as f64) * window_ratio).round() as usize;
            let width = optimal_width.max(1);
            let remaining = halfway - round;
            let actual_width = width.min(remaining);
            round += actual_width;
        }

        let switch_over_point = round;
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
        self.window_starts.binary_search(&round).is_ok()
    }

    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn num_unbound_vars(&self, round: usize) -> usize {
        let current_idx = match self.window_starts.binary_search(&round) {
            Ok(idx) => idx,
            Err(idx) => idx.saturating_sub(1),
        };

        let next_window_start = self
            .window_starts
            .get(current_idx + 1)
            .copied()
            .unwrap_or(self.num_rounds);

        next_window_start - round
    }
}

#[derive(Debug, Clone)]
pub struct LinearOnlySchedule {
    num_rounds: usize,
}

impl LinearOnlySchedule {
    pub fn new(num_rounds: usize) -> Self {
        Self { num_rounds }
    }
}

impl StreamingSchedule for LinearOnlySchedule {
    fn switch_over_point(&self) -> usize {
        0
    }

    fn is_window_start(&self, _round: usize) -> bool {
        true
    }

    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn num_unbound_vars(&self, _round: usize) -> usize {
        1
    }
}
