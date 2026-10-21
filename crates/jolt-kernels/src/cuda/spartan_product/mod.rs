//! CUDA Spartan product column packing and pairwise reductions.
//!
//! The device reduces the nine pairwise products of three native left and right
//! operand lanes against split equality weights. Optimized instead evaluates
//! extended-domain products directly from typed rows. CUDA retains signed wide
//! operands as magnitudes with separate sign bits and sums window matrices on
//! the host.

mod columns;
mod uniskip;
mod witness;
