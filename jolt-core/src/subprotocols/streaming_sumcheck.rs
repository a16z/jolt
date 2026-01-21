use std::cmp::Ordering;

use allocative::Allocative;

use crate::field::{JoltField, MaybeAllocative};
use crate::poly::opening_proof::ProverOpeningAccumulator;
use crate::poly::unipoly::UniPoly;
use crate::subprotocols::streaming_schedule::StreamingSchedule;
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::transcripts::Transcript;

pub trait StreamingSumcheckWindow<F: JoltField>: Sized + MaybeAllocative + Send + Sync {
    type Shared;

    fn initialize(shared: &mut Self::Shared, window_size: usize) -> Self;

    fn compute_message(
        &self,
        shared: &Self::Shared,
        window_size: usize,
        previous_claim: F,
    ) -> UniPoly<F>;

    fn ingest_challenge(&mut self, shared: &mut Self::Shared, r: F::Challenge, round: usize);
}

pub trait LinearSumcheckStage<F: JoltField>: Sized + MaybeAllocative + Send + Sync {
    type Shared;
    type Streaming: StreamingSumcheckWindow<F>;

    fn initialize(
        streaming: Option<Self::Streaming>,
        shared: &mut Self::Shared,
        window_size: usize,
    ) -> Self;

    fn next_window(&mut self, shared: &mut Self::Shared, window_size: usize);

    fn compute_message(
        &self,
        shared: &Self::Shared,
        window_size: usize,
        previous_claim: F,
    ) -> UniPoly<F>;

    fn ingest_challenge(&mut self, shared: &mut Self::Shared, r: F::Challenge, round: usize);

    fn cache_openings<T: Transcript>(
        &self,
        shared: &Self::Shared,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    );
}

pub trait SharedStreamingSumcheckState<F: JoltField>:
    Sized + MaybeAllocative + Send + Sync
{
    fn degree(&self) -> usize;
    fn num_rounds(&self) -> usize;
    fn input_claim(&self, accumulator: &ProverOpeningAccumulator<F>) -> F;
}

#[derive(Allocative)]
pub struct StreamingSumcheck<
    F: JoltField,
    S: StreamingSchedule,
    Shared: SharedStreamingSumcheckState<F>,
    Streaming: StreamingSumcheckWindow<F, Shared = Shared>,
    Linear: LinearSumcheckStage<F, Streaming = Streaming, Shared = Shared>,
> {
    streaming: Option<Streaming>,
    linear: Option<Linear>,
    shared: Shared,
    schedule: S,
    phantom: std::marker::PhantomData<F>,
}

impl<F, S, Shared, Streaming, Linear> StreamingSumcheck<F, S, Shared, Streaming, Linear>
where
    F: JoltField,
    S: StreamingSchedule,
    Shared: SharedStreamingSumcheckState<F>,
    Streaming: StreamingSumcheckWindow<F, Shared = Shared>,
    Linear: LinearSumcheckStage<F, Streaming = Streaming, Shared = Shared>,
{
    pub fn new(shared: Shared, schedule: S) -> Self {
        Self {
            streaming: None,
            linear: None,
            shared,
            schedule,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<F, T, S, Shared, Streaming, Linear> SumcheckInstanceProver<F, T>
    for StreamingSumcheck<F, S, Shared, Streaming, Linear>
where
    F: JoltField,
    T: Transcript,
    S: StreamingSchedule,
    Shared: SharedStreamingSumcheckState<F>,
    Streaming: StreamingSumcheckWindow<F, Shared = Shared>,
    Linear: LinearSumcheckStage<F, Streaming = Streaming, Shared = Shared>,
{
    fn degree(&self) -> usize {
        self.shared.degree()
    }

    fn num_rounds(&self) -> usize {
        self.shared.num_rounds()
    }

    fn input_claim(&self, accumulator: &ProverOpeningAccumulator<F>) -> F {
        self.shared.input_claim(accumulator)
    }

    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        let num_unbound_vars = self.schedule.num_unbound_vars(round);
        let switch_over = self.schedule.switch_over_point();

        match round.cmp(&switch_over) {
            Ordering::Less => {
                // STREAMING MODE
                if self.schedule.is_window_start(round) {
                    self.streaming =
                        Some(Streaming::initialize(&mut self.shared, num_unbound_vars));
                }
            }
            Ordering::Equal => {
                // SWITCHING TO LINEAR MODE
                assert!(
                    self.schedule.is_window_start(round),
                    "switch over is not a window start"
                );
                self.linear = Some(Linear::initialize(
                    self.streaming.take(),
                    &mut self.shared,
                    num_unbound_vars,
                ));
            }
            Ordering::Greater => {
                // LINEAR MODE
                assert!(
                    self.schedule.is_window_start(round),
                    "round is not a window start in linear mode"
                );
                self.linear
                    .as_mut()
                    .expect("no linear")
                    .next_window(&mut self.shared, num_unbound_vars);
            }
        }

        if let Some(streaming) = &mut self.streaming {
            streaming.compute_message(&self.shared, num_unbound_vars, previous_claim)
        } else if let Some(linear) = &mut self.linear {
            linear.compute_message(&self.shared, num_unbound_vars, previous_claim)
        } else {
            unreachable!()
        }
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        if let Some(streaming) = &mut self.streaming {
            streaming.ingest_challenge(&mut self.shared, r_j, round);
        } else if let Some(linear) = &mut self.linear {
            linear.ingest_challenge(&mut self.shared, r_j, round);
        } else {
            unreachable!()
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        self.linear.as_ref().expect("no linear").cache_openings(
            &self.shared,
            accumulator,
            transcript,
            sumcheck_challenges,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}
