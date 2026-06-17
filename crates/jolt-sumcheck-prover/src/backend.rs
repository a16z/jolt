use jolt_field::Field;
use jolt_poly::UnivariatePoly;

use crate::error::BackendError;
use crate::spec::BatchedSumcheckSpec;

/// Round-polynomial oracle for one batched sumcheck invocation.
///
/// The handler owns Fiat–Shamir (claim absorption, batching coefficients, round
/// challenges). Backends only materialize per-instance round polynomials and
/// apply private bind state.
pub trait SumcheckBackend<F: Field> {
    type State;

    fn start(&mut self, spec: &BatchedSumcheckSpec<F>) -> Result<Self::State, BackendError>;

    /// Returns one round polynomial per entry in `active` (same order).
    ///
    /// `claims[i]` is the running claim for `active[i]` at the start of `round`.
    fn round_polynomials(
        &mut self,
        state: &Self::State,
        round: usize,
        active: &[usize],
        claims: &[F],
    ) -> Result<Vec<UnivariatePoly<F>>, BackendError>;

    fn bind(
        &mut self,
        state: &mut Self::State,
        round: usize,
        instance: usize,
        challenge: F,
    ) -> Result<(), BackendError>;

    fn finish(&mut self, state: Self::State) -> Result<Vec<F>, BackendError> {
        let _ = state;
        Ok(Vec::new())
    }
}
