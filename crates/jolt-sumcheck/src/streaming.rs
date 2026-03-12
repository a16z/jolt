//! Streaming sumcheck: memory-efficient two-phase sumcheck prover.
//!
//! The streaming sumcheck processes large witness polynomials without
//! materializing them entirely in memory. It operates in two phases:
//!
//! 1. **Streaming phase** — early rounds compute round polynomials directly
//!    from windowed views of the data (e.g., the execution trace), avoiding
//!    full materialization.
//!
//! 2. **Linear phase** — later rounds operate on materialized polynomial
//!    evaluations after the streaming data structure has been fully bound.
//!
//! The transition point is controlled by a [`StreamingSchedule`], which
//! also determines window sizes for the streaming phase.
//!
//! # Architecture
//!
//! The streaming sumcheck is generic over four strategy types:
//! - `Shared`: shared state accessible to both phases (e.g., eq tables, key data)
//! - `Streaming`: [`StreamingSumcheckWindow`] — computes round polys in streaming mode
//! - `Linear`: [`LinearSumcheckStage`] — computes round polys in linear mode
//! - `S`: [`StreamingSchedule`] — controls phase transition and window sizing

use std::cmp::Ordering;

use jolt_field::{Field, WithChallenge};
use jolt_poly::UnivariatePoly;

use crate::SumcheckCompute;

/// Controls when the streaming sumcheck transitions from streaming to linear
/// phase, and how rounds are partitioned into windows.
///
/// The streaming phase processes data in windows — contiguous groups of
/// sumcheck rounds where the prover can compute round polynomials from
/// a compact representation. At the switch-over point, the prover
/// materializes the remaining polynomial and switches to linear-time
/// per-round evaluation.
pub trait StreamingSchedule: Send + Sync {
    /// The round at which we switch from streaming to linear mode.
    /// Rounds `0..switch_over_point()` use streaming; the rest use linear.
    fn switch_over_point(&self) -> usize;

    /// Whether this round starts a new window.
    ///
    /// At window boundaries, the streaming/linear data structures are
    /// (re)initialized for the upcoming group of rounds.
    fn is_window_start(&self, round: usize) -> bool;

    /// Total number of sumcheck rounds.
    fn num_rounds(&self) -> usize;

    /// Number of unbound variables remaining in the current window at
    /// the given round. For single-round windows, this is always 1.
    fn num_unbound_vars(&self, round: usize) -> usize;
}

/// Cost-optimal half-split schedule: streaming for the first half of rounds
/// with geometrically growing windows, then single-round linear windows
/// for the second half.
///
/// Window sizes follow `w = round(ratio * i)` where
/// `ratio = ln(2) / ln((degree+1)/2)`. This keeps the grid computation
/// cost `(d+1)^w / 2^(w+i)` approximately constant per window.
///
/// For degree 2 (multiquadratic): ratio ~ 1.71, windows at 0, 1, 3, 8, ...
/// For degree 3: ratio ~ 1.0, windows at 0, 1, 2, 3, ...
#[derive(Debug, Clone)]
pub struct HalfSplitSchedule {
    num_rounds: usize,
    switch_over_point: usize,
    window_starts: Vec<usize>,
}

impl HalfSplitSchedule {
    /// Computes the cost-optimal window ratio for a given polynomial degree.
    #[inline]
    pub fn compute_window_ratio(degree: usize) -> f64 {
        let d_plus_1 = (degree + 1) as f64;
        2.0_f64.ln() / (d_plus_1 / 2.0).ln()
    }

    /// Computes the optimal window size for a given round and degree.
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

/// Schedule that disables streaming entirely — all rounds use linear mode.
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

/// Computation strategy for the streaming phase of the sumcheck.
///
/// During streaming rounds, the prover computes round polynomials from
/// a windowed view of the data without materializing the full polynomial.
/// Each window covers a contiguous group of sumcheck variables.
pub trait StreamingSumcheckWindow<F: Field>: Sized + Send + Sync {
    /// Shared state accessible to both streaming and linear phases.
    type Shared;

    /// Initializes a new streaming window for `window_size` upcoming rounds.
    fn initialize(shared: &mut Self::Shared, window_size: usize) -> Self;

    /// Computes the round polynomial for the current streaming round.
    fn compute_message(
        &self,
        shared: &Self::Shared,
        window_size: usize,
        previous_claim: F,
    ) -> UnivariatePoly<F>;

    /// Incorporates the verifier's challenge for the current round.
    fn ingest_challenge(&mut self, shared: &mut Self::Shared, r: F, round: usize);
}

/// Computation strategy for the linear phase of the sumcheck.
///
/// After the streaming phase, the polynomial is materialized in memory
/// and subsequent rounds are computed in linear time per round.
pub trait LinearSumcheckStage<F: Field>: Sized + Send + Sync {
    /// Shared state (same type as the streaming phase).
    type Shared;
    /// The streaming window type that preceded this linear stage.
    type Streaming: StreamingSumcheckWindow<F, Shared = Self::Shared>;

    /// Initializes the linear stage, optionally consuming the final
    /// streaming window state.
    fn initialize(
        streaming: Option<Self::Streaming>,
        shared: &mut Self::Shared,
        window_size: usize,
    ) -> Self;

    /// Advances to the next single-round window within the linear phase.
    fn next_window(&mut self, shared: &mut Self::Shared, window_size: usize);

    /// Computes the round polynomial for the current linear round.
    fn compute_message(
        &self,
        shared: &Self::Shared,
        window_size: usize,
        previous_claim: F,
    ) -> UnivariatePoly<F>;

    /// Incorporates the verifier's challenge for the current round.
    fn ingest_challenge(&mut self, shared: &mut Self::Shared, r: F, round: usize);
}

/// Two-phase streaming sumcheck prover.
///
/// Dispatches between streaming and linear phases based on a
/// [`StreamingSchedule`]. The `Shared` state is accessible to both phases
/// and typically holds eq polynomial tables, key data, etc.
pub struct StreamingSumcheck<F, S, Shared, Streaming, Linear>
where
    F: Field,
    S: StreamingSchedule,
    Shared: Send + Sync,
    Streaming: StreamingSumcheckWindow<F, Shared = Shared>,
    Linear: LinearSumcheckStage<F, Streaming = Streaming, Shared = Shared>,
{
    streaming: Option<Streaming>,
    linear: Option<Linear>,
    shared: Shared,
    schedule: S,
    _phantom: std::marker::PhantomData<F>,
}

impl<F, S, Shared, Streaming, Linear> StreamingSumcheck<F, S, Shared, Streaming, Linear>
where
    F: Field,
    S: StreamingSchedule,
    Shared: Send + Sync,
    Streaming: StreamingSumcheckWindow<F, Shared = Shared>,
    Linear: LinearSumcheckStage<F, Streaming = Streaming, Shared = Shared>,
{
    pub fn new(shared: Shared, schedule: S) -> Self {
        Self {
            streaming: None,
            linear: None,
            shared,
            schedule,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn shared(&self) -> &Shared {
        &self.shared
    }

    pub fn shared_mut(&mut self) -> &mut Shared {
        &mut self.shared
    }

    pub fn schedule(&self) -> &S {
        &self.schedule
    }
}

impl<F, S, Shared, Streaming, Linear> SumcheckCompute<F>
    for StreamingSumcheck<F, S, Shared, Streaming, Linear>
where
    F: WithChallenge,
    S: StreamingSchedule,
    Shared: Send + Sync,
    Streaming: StreamingSumcheckWindow<F, Shared = Shared>,
    Linear: LinearSumcheckStage<F, Streaming = Streaming, Shared = Shared>,
{
    fn round_polynomial(&self) -> UnivariatePoly<F> {
        // This method is not used by the streaming sumcheck — it uses
        // `compute_round` instead. Provided to satisfy the SumcheckCompute
        // trait for compatibility with the standard prover.
        panic!("StreamingSumcheck must be driven via compute_round(), not round_polynomial()")
    }

    fn bind(&mut self, challenge: F::Challenge) {
        let challenge_f: F = challenge.into();
        if let Some(streaming) = &mut self.streaming {
            streaming.ingest_challenge(&mut self.shared, challenge_f, 0);
        } else if let Some(linear) = &mut self.linear {
            linear.ingest_challenge(&mut self.shared, challenge_f, 0);
        }
    }
}

impl<F, S, Shared, Streaming, Linear> StreamingSumcheck<F, S, Shared, Streaming, Linear>
where
    F: Field,
    S: StreamingSchedule,
    Shared: Send + Sync,
    Streaming: StreamingSumcheckWindow<F, Shared = Shared>,
    Linear: LinearSumcheckStage<F, Streaming = Streaming, Shared = Shared>,
{
    /// Computes the round polynomial for the given round, managing
    /// phase transitions and window initialization automatically.
    pub fn compute_round(&mut self, round: usize, previous_claim: F) -> UnivariatePoly<F> {
        let num_unbound_vars = self.schedule.num_unbound_vars(round);
        let switch_over = self.schedule.switch_over_point();

        match round.cmp(&switch_over) {
            Ordering::Less => {
                if self.schedule.is_window_start(round) {
                    self.streaming =
                        Some(Streaming::initialize(&mut self.shared, num_unbound_vars));
                }
            }
            Ordering::Equal => {
                assert!(
                    self.schedule.is_window_start(round),
                    "switch-over is not a window start"
                );
                self.linear = Some(Linear::initialize(
                    self.streaming.take(),
                    &mut self.shared,
                    num_unbound_vars,
                ));
            }
            Ordering::Greater => {
                if self.schedule.is_window_start(round) {
                    self.linear
                        .as_mut()
                        .expect("linear stage not initialized")
                        .next_window(&mut self.shared, num_unbound_vars);
                }
            }
        }

        if let Some(streaming) = &self.streaming {
            streaming.compute_message(&self.shared, num_unbound_vars, previous_claim)
        } else if let Some(linear) = &self.linear {
            linear.compute_message(&self.shared, num_unbound_vars, previous_claim)
        } else {
            unreachable!("neither streaming nor linear stage active")
        }
    }

    /// Incorporates the verifier's challenge for the given round.
    pub fn ingest_challenge(&mut self, r: F, round: usize) {
        if let Some(streaming) = &mut self.streaming {
            streaming.ingest_challenge(&mut self.shared, r, round);
        } else if let Some(linear) = &mut self.linear {
            linear.ingest_challenge(&mut self.shared, r, round);
        } else {
            unreachable!("neither streaming nor linear stage active");
        }
    }
}

/// A sumcheck prover that processes witness evaluations in streaming
/// fashion, enabling proofs over polynomials that do not fit in memory.
///
/// This is the chunk-based API — for the two-phase window/linear API,
/// see [`StreamingSumcheck`].
pub trait StreamingSumcheckProver<F: Field>: Send + Sync {
    /// Resets internal accumulators for a new round.
    fn begin_round(&mut self);

    /// Ingests a contiguous slice of evaluations and updates accumulators.
    fn process_chunk(&mut self, chunk: &[F]);

    /// Finalizes the current round's accumulators and returns the round
    /// polynomial $s_i(X)$.
    fn finish_round(&mut self) -> UnivariatePoly<F>;

    /// Fixes the current leading variable to `challenge`.
    fn bind(&mut self, challenge: F);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_split_window_ratio() {
        assert!((HalfSplitSchedule::compute_window_ratio(2) - 1.71).abs() < 0.01);
        assert!((HalfSplitSchedule::compute_window_ratio(3) - 1.0).abs() < 0.01);
    }

    #[test]
    fn half_split_optimal_window_size() {
        assert_eq!(HalfSplitSchedule::optimal_window_size(0, 2), 1);
        assert_eq!(HalfSplitSchedule::optimal_window_size(1, 2), 2);
        assert_eq!(HalfSplitSchedule::optimal_window_size(3, 2), 5);
    }

    #[test]
    fn half_split_schedule_construction() {
        let schedule = HalfSplitSchedule::new(20, 2);
        assert_eq!(schedule.num_rounds(), 20);
        assert_eq!(schedule.switch_over_point(), 10);

        assert!(schedule.is_window_start(0));
        assert!(schedule.is_window_start(1));
        assert!(schedule.is_window_start(3));
        assert!(!schedule.is_window_start(2));
        assert!(!schedule.is_window_start(5));

        // Unbound vars within a window
        assert_eq!(schedule.num_unbound_vars(3), 5);
        assert_eq!(schedule.num_unbound_vars(7), 1);

        // After switch-over: single-round windows
        for round in 10..20 {
            assert_eq!(schedule.num_unbound_vars(round), 1);
            assert!(schedule.is_window_start(round));
        }
    }

    #[test]
    fn half_split_small_schedule() {
        let small = HalfSplitSchedule::new(2, 2);
        assert_eq!(small.switch_over_point(), 1);
        assert_eq!(small.num_rounds(), 2);
    }

    #[test]
    fn half_split_cost_model() {
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

    #[test]
    fn linear_only_schedule() {
        let schedule = LinearOnlySchedule::new(10);
        assert_eq!(schedule.switch_over_point(), 0);
        assert_eq!(schedule.num_rounds(), 10);
        for round in 0..10 {
            assert!(schedule.is_window_start(round));
            assert_eq!(schedule.num_unbound_vars(round), 1);
        }
    }
}
