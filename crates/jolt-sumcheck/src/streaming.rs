//! Streaming sumcheck prover trait for memory-constrained environments.
//!
//! When the witness polynomial is too large to hold in memory, a streaming
//! prover processes it in chunks. Each round proceeds in three phases:
//!
//! 1. [`begin_round`](StreamingSumcheckProver::begin_round) -- resets
//!    accumulators for the new round.
//! 2. [`process_chunk`](StreamingSumcheckProver::process_chunk) -- ingests
//!    a slice of evaluations and accumulates partial sums.
//! 3. [`finish_round`](StreamingSumcheckProver::finish_round) -- finalizes
//!    the round polynomial from the accumulated state.
//!
//! After the round polynomial is sent and a challenge is received,
//! [`bind`](StreamingSumcheckProver::bind) fixes the current variable.

use jolt_field::Field;
use jolt_poly::UnivariatePoly;

/// A sumcheck prover that processes witness evaluations in streaming
/// fashion, enabling proofs over polynomials that do not fit in memory.
///
/// Implementors maintain internal accumulators that are updated as chunks
/// of the evaluation table are streamed in. The protocol driver calls the
/// methods in the order: `begin_round`, then one or more `process_chunk`,
/// then `finish_round`, then `bind`, and repeats for the next round.
pub trait StreamingSumcheckProver<F: Field>: Send + Sync {
    /// Resets internal accumulators for a new round.
    ///
    /// Must be called before any [`process_chunk`](Self::process_chunk)
    /// calls for the round.
    fn begin_round(&mut self);

    /// Ingests a contiguous slice of evaluations and updates accumulators.
    ///
    /// The `chunk` contains a contiguous segment of the remaining evaluation
    /// table. Chunks need not be equal-sized, but together they must cover
    /// the entire table exactly once per round.
    fn process_chunk(&mut self, chunk: &[F]);

    /// Finalizes the current round's accumulators and returns the round
    /// polynomial $s_i(X)$.
    ///
    /// Must be called exactly once per round, after all chunks have been
    /// processed.
    fn finish_round(&mut self) -> UnivariatePoly<F>;

    /// Fixes the current leading variable to `challenge`, preparing the
    /// prover for the next round with one fewer variable.
    fn bind(&mut self, challenge: F);
}
