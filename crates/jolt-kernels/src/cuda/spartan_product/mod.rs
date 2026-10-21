//! CUDA Spartan product column packing, pairwise reductions and remainder rounds.
//!
//! The device reduces the nine pairwise products of three native left and right
//! operand lanes against split equality weights. Optimized instead evaluates
//! extended-domain products directly from typed rows. CUDA retains signed wide
//! operands as magnitudes with separate sign bits and sums window matrices on
//! the host.
//!
//! CUDA materializes challenge-weighted operand tables before the first remainder
//! round; optimized fuses that materialization with round zero. Each device folds
//! its tables against split equality weights before the short tail is gathered
//! onto GPU 0. Native columns stay resident for the individual output claims.

mod columns;
mod remainder;
mod uniskip;
mod witness;
